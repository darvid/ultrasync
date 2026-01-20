"""Server state management."""

from __future__ import annotations

import asyncio
import fcntl
import os
from pathlib import Path
from typing import IO, TYPE_CHECKING

from ultrasync_mcp.file_registry import FileRegistry
from ultrasync_mcp.logging_config import get_logger
from ultrasync_mcp.paths import get_data_dir
from ultrasync_mcp.patterns import PatternSetManager
from ultrasync_mcp.server.config import (
    DEFAULT_EMBEDDING_MODEL,
    ENV_CLIENT_ROOT,
)
from ultrasync_mcp.server.utils import (
    detect_coding_agent,
    get_transcript_parser,
)
from ultrasync_mcp.threads import ThreadManager
from ultrasync_mcp.transcript_watcher import TranscriptWatcher, WatcherStats

if TYPE_CHECKING:
    from ultrasync_index import GlobalIndex
    from ultrasync_mcp.embeddings import EmbeddingProvider
    from ultrasync_mcp.jit.manager import JITIndexManager
    from ultrasync_mcp.jit.session_threads import PersistentThreadManager
    from ultrasync_mcp.sync_client import (
        SyncClient,
        SyncManager,
        SyncManagerStats,
    )

logger = get_logger("mcp_server")


class ServerState:
    """Core server state with lazy initialization and lifecycle management."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        root: Path | None = None,
        jit_data_dir: Path | None = None,
        aot_index_path: Path | None = None,
        aot_blob_path: Path | None = None,
        watch_transcripts: bool = False,
        agent: str | None = None,
        enable_learning: bool = True,
    ) -> None:
        self._model_name = model_name
        self._root = root
        self._jit_data_dir = jit_data_dir or get_data_dir(root)
        self._aot_index_path = aot_index_path
        self._aot_blob_path = aot_blob_path
        self._embedder: EmbeddingProvider | None = None
        self._thread_manager: ThreadManager | None = None
        self._file_registry: FileRegistry | None = None
        self._jit_manager: JITIndexManager | None = None
        self._aot_index: GlobalIndex | None = None
        self._aot_checked = False
        self._pattern_manager: PatternSetManager | None = None
        self._watch_transcripts = watch_transcripts
        self._agent = agent
        self._enable_learning = enable_learning
        self._watcher: TranscriptWatcher | None = None
        self._watcher_started = False
        self._watcher_lock_fd: IO[bytes] | None = None
        self._persistent_thread_manager: PersistentThreadManager | None = None
        self._init_task: asyncio.Task[None] | None = None
        self._sync_manager: SyncManager | None = None
        self._init_lock: asyncio.Lock | None = None

        env_client_root = os.environ.get(ENV_CLIENT_ROOT)
        if env_client_root:
            self._client_root = env_client_root
            logger.info("client root from env: %s", env_client_root)
        else:
            self._client_root = None
            if root is not None:
                logger.warning(
                    "running with --directory=%s but no CLIENT_ROOT set. "
                    "sync uses this dir initially. set ULTRASYNC_CLIENT_ROOT "
                    "to the client project path for project isolation.",
                    root,
                )

    # -------------------------------------------------------------------------
    # Lazy initialization
    # -------------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        if self._embedder is None:
            from ultrasync_mcp.embeddings import SentenceTransformerProvider

            self._embedder = SentenceTransformerProvider(self._model_name)
            self._thread_manager = ThreadManager(self._embedder)
            self._file_registry = FileRegistry(
                root=self._root, embedder=self._embedder
            )

    def _ensure_jit_initialized(self) -> None:
        self._ensure_initialized()
        if self._jit_manager is None:
            from ultrasync_mcp.jit.manager import JITIndexManager

            assert self._embedder is not None
            self._jit_manager = JITIndexManager(
                data_dir=self._jit_data_dir,
                embedding_provider=self._embedder,
            )

    async def _init_index_manager_async(self) -> None:
        """Initialize index manager without blocking event loop."""
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()

        async with self._init_lock:
            if self._embedder is None or self._jit_manager is None:
                await asyncio.to_thread(self._ensure_jit_initialized)

            if self._embedder is not None and not hasattr(self, "_warmup_done"):
                import time

                logger.info("warming up embedding model...")
                start = time.perf_counter()
                await asyncio.to_thread(self._embedder.embed, "warmup")
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.info(
                    "embedding model ready (warmup took %.0fms)", elapsed_ms
                )
                self._warmup_done = True

    def _ensure_aot_initialized(self) -> None:
        """Try to load AOT GlobalIndex if it exists."""
        if self._aot_checked:
            return
        self._aot_checked = True

        index_path = self._aot_index_path
        blob_path = self._aot_blob_path

        if index_path is None:
            default_index = self._jit_data_dir / "index.dat"
            if default_index.exists():
                index_path = default_index
        if blob_path is None:
            default_blob = self._jit_data_dir / "aot_blob.dat"
            if default_blob.exists():
                blob_path = default_blob

        if (
            index_path
            and blob_path
            and index_path.exists()
            and blob_path.exists()
        ):
            try:
                from ultrasync_index import GlobalIndex as RustGlobalIndex

                self._aot_index = RustGlobalIndex(
                    str(index_path), str(blob_path)
                )
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def embedder(self) -> EmbeddingProvider:
        self._ensure_initialized()
        assert self._embedder is not None
        return self._embedder

    @property
    def thread_manager(self) -> ThreadManager:
        self._ensure_initialized()
        assert self._thread_manager is not None
        return self._thread_manager

    @property
    def file_registry(self) -> FileRegistry:
        self._ensure_initialized()
        assert self._file_registry is not None
        return self._file_registry

    @property
    def jit_manager(self) -> JITIndexManager:
        self._ensure_jit_initialized()
        assert self._jit_manager is not None
        return self._jit_manager

    async def get_jit_manager_async(self) -> JITIndexManager:
        """Get JIT manager, waiting for background init if in progress."""
        if self._init_task is not None and not self._init_task.done():
            await self._init_task
        if self._jit_manager is None:
            self._ensure_jit_initialized()
        assert self._jit_manager is not None
        return self._jit_manager

    @property
    def aot_index(self) -> GlobalIndex | None:
        """Get AOT GlobalIndex if available, None otherwise."""
        self._ensure_aot_initialized()
        return self._aot_index

    @property
    def pattern_manager(self) -> PatternSetManager:
        """Get the PatternSetManager, initializing if needed."""
        if self._pattern_manager is None:
            self._pattern_manager = PatternSetManager(self._jit_data_dir)
        return self._pattern_manager

    @property
    def root(self) -> Path | None:
        return self._root

    @property
    def watcher(self) -> TranscriptWatcher | None:
        return self._watcher

    @property
    def persistent_thread_manager(self) -> PersistentThreadManager | None:
        return self._persistent_thread_manager

    @property
    def client_root(self) -> str | None:
        return self._client_root

    @property
    def sync_manager(self) -> SyncManager | None:
        return self._sync_manager

    @property
    def sync_client(self) -> SyncClient | None:
        if self._sync_manager:
            return self._sync_manager.client
        return None

    # -------------------------------------------------------------------------
    # Watcher management
    # -------------------------------------------------------------------------

    def _try_acquire_watcher_lock(self) -> bool:
        """Try to acquire exclusive lock for transcript watching."""
        lock_path = self._jit_data_dir / "watcher.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._watcher_lock_fd = open(lock_path, "wb")
            fd = self._watcher_lock_fd.fileno()
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._watcher_lock_fd.write(f"{os.getpid()}\n".encode())
            self._watcher_lock_fd.flush()
            logger.info(
                "acquired watcher lock (leader): pid=%d path=%s",
                os.getpid(),
                lock_path,
            )
            return True
        except OSError:
            if self._watcher_lock_fd:
                self._watcher_lock_fd.close()
                self._watcher_lock_fd = None
            logger.info(
                "watcher lock held by another process, skipping transcript "
                "watching (follower mode): path=%s",
                lock_path,
            )
            return False

    def _release_watcher_lock(self) -> None:
        """Release the watcher lock if we hold it."""
        if self._watcher_lock_fd:
            try:
                fcntl.flock(self._watcher_lock_fd.fileno(), fcntl.LOCK_UN)
                self._watcher_lock_fd.close()
            except OSError:
                pass
            self._watcher_lock_fd = None
            logger.info("released watcher lock")

    async def start_watcher(self) -> None:
        """Start the transcript watcher if configured."""
        if not self._watch_transcripts or self._watcher_started:
            return

        project_root = self._root or Path.cwd()
        parser = get_transcript_parser(self._agent)

        if parser is None:
            detected = detect_coding_agent()
            logger.warning(
                "transcript watching enabled but no parser available "
                "(agent=%s, detected=%s)",
                self._agent,
                detected,
            )
            return

        if not self._try_acquire_watcher_lock():
            return

        await self._init_index_manager_async()
        assert self._jit_manager is not None
        assert self._embedder is not None

        from ultrasync_mcp.jit.session_threads import PersistentThreadManager

        self._persistent_thread_manager = PersistentThreadManager(
            tracker=self._jit_manager.tracker,
            embedder=self._embedder,
            vector_cache=self._jit_manager.vector_cache,
        )

        self._watcher = TranscriptWatcher(
            project_root=project_root,
            jit_manager=self._jit_manager,
            parser=parser,
            enable_learning=self._enable_learning,
            thread_manager=self._persistent_thread_manager,
        )
        await self._watcher.start()
        self._watcher_started = True

        logger.info(
            "transcript watcher started: agent=%s project=%s threads=enabled",
            parser.agent_name,
            project_root,
        )

    async def stop_watcher(self) -> None:
        """Stop the transcript watcher if running."""
        if self._watcher:
            await self._watcher.stop()
            self._watcher = None
            self._watcher_started = False
        self._release_watcher_lock()

    def get_watcher_stats(self) -> WatcherStats | None:
        """Get transcript watcher statistics."""
        if self._watcher:
            return self._watcher.get_stats()
        return None

    # -------------------------------------------------------------------------
    # Sync management
    # -------------------------------------------------------------------------

    async def start_sync_manager(
        self,
        client_root: str | None = None,
        wait_for_root: bool = True,
        wait_timeout: float = 3.0,
    ) -> bool:
        """Start the sync manager if configured."""
        from ultrasync_mcp.sync_client import (
            SyncConfig,
            SyncManager,
            is_remote_sync_enabled,
        )

        if self._sync_manager is not None:
            logger.warning("sync manager already running")
            return True

        if not is_remote_sync_enabled():
            logger.debug("remote sync not enabled")
            return False

        if not client_root and not self._client_root and wait_for_root:
            logger.debug(
                "waiting %.1fs for client root detection...", wait_timeout
            )
            waited = 0.0
            poll_interval = 0.1
            while waited < wait_timeout and not self._client_root:
                await asyncio.sleep(poll_interval)
                waited += poll_interval
            if self._client_root:
                logger.info(
                    "client root detected after %.1fs: %s",
                    waited,
                    self._client_root,
                )
            else:
                logger.warning(
                    "no client root detected after %.1fs, using MCP cwd",
                    wait_timeout,
                )

        config = SyncConfig()
        logger.info(
            "SyncConfig created: git_remote=%s project_name=%s",
            config.git_remote,
            config.project_name,
        )
        if not config.is_configured:
            logger.warning(
                "sync enabled but not configured - set "
                "ULTRASYNC_SYNC_URL and ULTRASYNC_SYNC_TOKEN"
            )
            return False

        if client_root:
            logger.info("updating config from client_root: %s", client_root)
            config.update_from_client_root(client_root)
        elif self._client_root:
            logger.info(
                "updating config from _client_root: %s", self._client_root
            )
            config.update_from_client_root(self._client_root)

        logger.info(
            "SyncConfig after update: git_remote=%s project_name=%s",
            config.git_remote,
            config.project_name,
        )

        await self._init_index_manager_async()
        assert self._jit_manager is not None

        def on_team_memory(payload: dict) -> None:
            if self._jit_manager is None:
                return
            memory_manager = self._jit_manager.memory
            if memory_manager is None:
                return
            try:
                memory_manager.import_memory(
                    memory_id=payload.get("id", ""),
                    text=payload.get("text", ""),
                    task=payload.get("task"),
                    insights=payload.get("insights"),
                    context=payload.get("context"),
                    tags=payload.get("tags"),
                    owner_id=payload.get("owner_id"),
                    created_at=payload.get("created_at"),
                )
            except Exception as e:
                logger.exception("failed to import team memory: %s", e)

        self._sync_manager = SyncManager(
            tracker=self._jit_manager.tracker,
            config=config,
            resync_interval=300,
            batch_size=50,
            on_team_memory=on_team_memory,
            graph_memory=self._jit_manager.graph,
            jit_manager=self._jit_manager,
        )

        started = await self._sync_manager.start()
        if started:
            logger.info(
                "sync manager started: project=%s git_remote=%s",
                config.project_name,
                config.git_remote,
            )
        return started

    async def stop_sync_manager(self) -> None:
        """Stop the sync manager if running."""
        if self._sync_manager:
            await self._sync_manager.stop()
            self._sync_manager = None

    async def _reconnect_sync_manager(self) -> None:
        """Stop and restart sync manager with updated config."""
        if not self._sync_manager:
            return

        try:
            old_config = self._sync_manager.config
            logger.info(
                "restarting sync manager for project switch: %s",
                old_config.git_remote,
            )
            await self._sync_manager.stop()
            self._sync_manager = None
            await asyncio.sleep(0.5)
            logger.info("starting fresh sync manager...")
            await self.start_sync_manager()
        except Exception as e:
            logger.error("failed to restart sync manager: %s", e)

    def set_client_root(self, root: str) -> None:
        """Set the client workspace root from MCP list_roots()."""
        if self._client_root == root:
            return

        old_root = self._client_root
        logger.info("client root detected: %s (previous: %s)", root, old_root)
        self._client_root = root

        new_jit_data_dir = get_data_dir(Path(root))
        if new_jit_data_dir != self._jit_data_dir:
            old_jit_dir = self._jit_data_dir
            self._jit_data_dir = new_jit_data_dir
            logger.info(
                "jit data dir changed: %s -> %s", old_jit_dir, new_jit_data_dir
            )

            if self._jit_manager is not None:
                logger.info(
                    "reinitializing jit_manager for new project data directory"
                )
                if hasattr(self._jit_manager, "close"):
                    try:
                        self._jit_manager.close()
                    except Exception as e:
                        logger.warning("error closing old jit_manager: %s", e)
                self._jit_manager = None

        if self._sync_manager and self._sync_manager.config:
            old_remote = self._sync_manager.config.git_remote
            self._sync_manager.config.update_from_client_root(root)
            new_remote = self._sync_manager.config.git_remote
            if old_remote != new_remote:
                logger.warning(
                    "sync manager git_remote changed after connect! "
                    "old=%s new=%s - triggering reconnect",
                    old_remote,
                    new_remote,
                )
                asyncio.create_task(self._reconnect_sync_manager())

    def get_sync_stats(self) -> SyncManagerStats | None:
        """Get sync manager statistics."""
        if self._sync_manager:
            return self._sync_manager.get_stats()
        return None

    # -------------------------------------------------------------------------
    # Compaction
    # -------------------------------------------------------------------------

    async def start_compaction_loop(
        self,
        interval_seconds: int = 3600,
        initial_delay: int = 300,
    ) -> None:
        """Start background compaction loop."""
        self._compaction_stop_event = asyncio.Event()

        stop = self._compaction_stop_event
        while self._jit_manager is None and not stop.is_set():
            await asyncio.sleep(1)

        if self._compaction_stop_event.is_set():
            return

        try:
            await asyncio.wait_for(
                self._compaction_stop_event.wait(),
                timeout=initial_delay,
            )
            return
        except asyncio.TimeoutError:
            pass

        logger.info("compaction loop started, interval=%ds", interval_seconds)

        while not self._compaction_stop_event.is_set():
            try:
                if self._jit_manager:
                    result = self._jit_manager.maybe_compact()
                    if result.get("errors"):
                        for err in result["errors"]:
                            logger.warning("compaction error: %s", err)
            except Exception as e:
                logger.exception("compaction loop error: %s", e)

            try:
                await asyncio.wait_for(
                    self._compaction_stop_event.wait(),
                    timeout=interval_seconds,
                )
                break
            except asyncio.TimeoutError:
                pass

        logger.info("compaction loop stopped")

    async def stop_compaction_loop(self) -> None:
        """Stop the compaction loop."""
        if hasattr(self, "_compaction_stop_event"):
            self._compaction_stop_event.set()
