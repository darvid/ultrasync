"""Async Socket.IO client for syncing with ultrasync.web server.

This module provides a background sync client that connects to the
hub-and-spoke sync server and pushes index/memory changes as ops.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import socketio

from ultrasync.logging_config import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger("sync_client")


def _debug_log(msg: str) -> None:
    """Write debug messages to a file for troubleshooting sync issues."""
    import datetime

    with open("/tmp/ultrasync_sync_debug.log", "a") as f:
        ts = datetime.datetime.now().isoformat()
        f.write(f"[{ts}] {msg}\n")
        f.flush()


# env vars for sync configuration
# primary gate - set to "true" to enable remote sync
ENV_REMOTE_SYNC = "ULTRASYNC_REMOTE_SYNC"

# sync server connection details
ENV_SYNC_URL = "ULTRASYNC_SYNC_URL"
ENV_SYNC_TOKEN = "ULTRASYNC_SYNC_TOKEN"
ENV_SYNC_PROJECT_NAME = "ULTRASYNC_SYNC_PROJECT_NAME"

# deprecated - org_id and project_id are now derived from token
ENV_SYNC_ORG_ID = "ULTRASYNC_SYNC_ORG_ID"  # deprecated
ENV_SYNC_PROJECT_ID = "ULTRASYNC_SYNC_PROJECT_ID"  # deprecated
ENV_SYNC_CLERK_USER_ID = "ULTRASYNC_SYNC_CLERK_USER_ID"


def is_remote_sync_enabled() -> bool:
    """Check if remote sync is enabled via env var.

    Returns True if ULTRASYNC_REMOTE_SYNC is set to a truthy value
    (true, 1, yes, on - case insensitive).
    """
    val = os.environ.get(ENV_REMOTE_SYNC, "").lower()
    return val in ("true", "1", "yes", "on")


def _get_project_name() -> str:
    """Get project name from env or auto-detect from git.

    Returns the project name for sync isolation. Uses:
    1. ULTRASYNC_SYNC_PROJECT_NAME env var if set
    2. Git remote origin URL (extracts repo name)
    3. Current directory name as fallback
    """
    # explicit env var takes precedence
    env_name = os.environ.get(ENV_SYNC_PROJECT_NAME, "")
    if env_name:
        return env_name

    # try to get from git remote
    try:
        import subprocess

        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            # extract repo name from git URL
            # handles: git@github.com:user/repo.git, https://github.com/user/repo.git
            if url.endswith(".git"):
                url = url[:-4]
            name = url.split("/")[-1].split(":")[-1]
            if name:
                return name
    except Exception:
        pass

    # fallback to current directory name
    return os.path.basename(os.getcwd())


@dataclass
class SyncConfig:
    """Configuration for sync client.

    Environment variables:
        ULTRASYNC_REMOTE_SYNC: Set to "true" to enable sync (gate)
        ULTRASYNC_SYNC_URL: Sync server URL (e.g., "http://localhost:5000")
        ULTRASYNC_SYNC_TOKEN: Auth token (required for authentication)
        ULTRASYNC_SYNC_PROJECT_NAME: Project/repo name for isolation
            (auto-detected from git if not set)

    The server derives org_id and project_id from the token claims and
    project_name, so you don't need to provide UUIDs explicitly.

    Example MCP config (~/.claude/.claude.json):
        {
          "mcpServers": {
            "ultrasync": {
              "command": "uv",
              "args": ["--directory", "/path/to/ultrasync", "run", "ultrasync"],
              "env": {
                "ULTRASYNC_REMOTE_SYNC": "true",
                "ULTRASYNC_SYNC_URL": "https://sync.example.com",
                "ULTRASYNC_SYNC_TOKEN": "your-auth-token"
              }
            }
          }
        }
    """

    url: str = field(default_factory=lambda: os.environ.get(ENV_SYNC_URL, ""))
    token: str = field(
        default_factory=lambda: os.environ.get(ENV_SYNC_TOKEN, "")
    )
    project_name: str = field(default_factory=_get_project_name)
    actor_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    client_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    clerk_user_id: str | None = field(
        default_factory=lambda: os.environ.get(ENV_SYNC_CLERK_USER_ID)
    )

    @property
    def is_enabled(self) -> bool:
        """Check if sync is enabled via ULTRASYNC_REMOTE_SYNC."""
        return is_remote_sync_enabled()

    @property
    def is_configured(self) -> bool:
        """Check if sync is properly configured (url and token required)."""
        return bool(self.url and self.token)


@dataclass
class BulkSyncProgress:
    """Progress tracking for bulk sync operations."""

    state: str = "idle"  # idle, syncing, error, complete
    total: int = 0
    synced: int = 0
    current_batch: int = 0
    total_batches: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None


class SyncClient:
    """Async Socket.IO client for syncing with ultrasync.web server.

    Usage:
        client = SyncClient(config)
        await client.connect()

        # push an op
        await client.push_op(
            namespace="index",
            key="file:/path/to/file.py",
            op_type="set",
            payload={"symbols": [...], "embedding": [...]}
        )

        await client.disconnect()
    """

    def __init__(
        self,
        config: SyncConfig | None = None,
        on_ops: Callable[[list[dict]], None] | None = None,
    ) -> None:
        self.config = config or SyncConfig()
        self._sio = socketio.AsyncClient(logger=False, engineio_logger=False)
        self._connected = False
        self._last_seq = 0
        self._pending_acks: dict[str, asyncio.Future] = {}
        self._on_ops = on_ops

        # register handlers
        self._sio.on("connect", self._on_connect)
        self._sio.on("disconnect", self._on_disconnect)
        self._sio.on("ops", self._on_ops_received)
        self._sio.on("ack", self._on_ack)
        self._sio.on("reject", self._on_reject)
        self._sio.on("error", self._on_error)

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self, timeout: int = 5) -> bool:
        """Connect to the sync server.

        Returns:
            True if connected successfully, False otherwise
        """
        if not self.config.is_configured:
            logger.warning("sync not configured, skipping connect")
            return False

        try:
            await asyncio.wait_for(
                self._sio.connect(self.config.url, wait_timeout=timeout),
                timeout=timeout + 1,
            )

            # wait for _on_connect callback to fire (sets _connected = True)
            # this prevents race conditions where _sync_loop starts before
            # the connection is fully established
            for _ in range(50):  # 50 * 0.1s = 5s max wait
                if self._connected:
                    break
                await asyncio.sleep(0.1)

            if not self._connected:
                logger.error("connected but _on_connect callback never fired")
                return False

            # send hello with token-based auth
            # server derives org_id/project_id from token + project_name
            hello = {
                "client_id": self.config.client_id,
                "token": self.config.token,
                "project_name": self.config.project_name,
                "last_server_seq": self._last_seq,
                "clerk_user_id": self.config.clerk_user_id,
            }
            await self._sio.emit("hello", hello)
            logger.info(
                "sync client connected to %s (project: %s)",
                self.config.url,
                self.config.project_name,
            )
            return True

        except Exception as e:
            logger.error("failed to connect to sync server: %s", e)
            return False

    async def disconnect(self) -> None:
        """Disconnect from the sync server."""
        if self._sio.connected:
            await self._sio.disconnect()
        self._connected = False

    async def push_op(
        self,
        namespace: str,
        key: str,
        op_type: str,
        payload: dict[str, Any],
        timeout: float = 5.0,
    ) -> dict | None:
        """Push an operation to the sync server.

        Args:
            namespace: One of presence, settings, index, collab, metadata
            key: Unique key within the namespace
            op_type: set, del, or patch
            payload: Operation payload
            timeout: Seconds to wait for ack

        Returns:
            The ack response if successful, None otherwise
        """
        if not self._connected:
            logger.debug("not connected, skipping push_op")
            return None

        command_id = str(uuid.uuid4())
        command = {
            "command_id": command_id,
            "namespace": namespace,
            "key": key,
            "op_type": op_type,
            "payload": payload,
        }

        # create future for ack
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_acks[command_id] = future

        try:
            await self._sio.emit("command", command)
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            logger.warning("timeout waiting for ack on command %s", command_id)
            return None
        finally:
            self._pending_acks.pop(command_id, None)

    async def push_file_indexed(
        self,
        path: str,
        symbols: list[dict],
        file_hash: str | None = None,
    ) -> dict | None:
        """Push a file indexing event.

        Args:
            path: File path
            symbols: List of symbol dicts with name, kind, lines, etc.
            file_hash: Optional content hash for dedup
        """
        return await self.push_op(
            namespace="index",
            key=f"file:{path}",
            op_type="set",
            payload={
                "path": path,
                "symbols": symbols,
                "file_hash": file_hash,
            },
        )

    async def push_memory(
        self,
        memory_id: str,
        text: str,
        task: str | None = None,
        insights: list[str] | None = None,
        context: list[str] | None = None,
    ) -> dict | None:
        """Push a memory entry.

        Args:
            memory_id: Unique memory ID (e.g., "mem:abc123")
            text: Memory content
            task: Task type
            insights: Insight tags
            context: Context tags
        """
        return await self.push_op(
            namespace="metadata",
            key=f"memory:{memory_id}",
            op_type="set",
            payload={
                "id": memory_id,
                "text": text,
                "task": task,
                "insights": insights or [],
                "context": context or [],
            },
        )

    async def push_presence(
        self,
        cursor_file: str | None = None,
        cursor_line: int | None = None,
        activity: str | None = None,
    ) -> dict | None:
        """Push presence/cursor info.

        Args:
            cursor_file: Current file being viewed
            cursor_line: Current line number
            activity: Current activity (editing, searching, etc.)
        """
        return await self.push_op(
            namespace="presence",
            key=f"cursor:{self.config.actor_id}",
            op_type="set",
            payload={
                "file": cursor_file,
                "line": cursor_line,
                "activity": activity,
            },
        )

    async def bulk_sync(
        self,
        items: list[dict[str, Any]],
        batch_size: int = 50,
        on_progress: Callable[[BulkSyncProgress], None] | None = None,
    ) -> BulkSyncProgress:
        """Bulk sync multiple items via HTTP endpoint.

        Uses the /api/sync/bulk endpoint for efficient batch uploads
        instead of individual WebSocket ops.

        Args:
            items: List of items to sync, each with:
                - namespace: "index" | "metadata" | "presence"
                - key: string key (e.g., file path)
                - op_type: "set" | "del"
                - payload: dict payload
            batch_size: Number of items per batch (default 50)
            on_progress: Optional callback for progress updates

        Returns:
            BulkSyncProgress with final state
        """
        import hashlib
        from datetime import datetime, timezone

        import aiohttp

        progress = BulkSyncProgress(
            state="syncing",
            total=len(items),
            synced=0,
            current_batch=0,
            total_batches=(len(items) + batch_size - 1) // batch_size,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        if on_progress:
            on_progress(progress)

        if not items:
            progress.state = "complete"
            progress.completed_at = datetime.now(timezone.utc).isoformat()
            if on_progress:
                on_progress(progress)
            return progress

        batches = [
            items[i : i + batch_size] for i in range(0, len(items), batch_size)
        ]

        base_url = self.config.url.rstrip("/")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}/api/tokens/verify",
                    headers={"Authorization": f"Bearer {self.config.token}"},
                ) as resp:
                    if resp.status != 200:
                        progress.state = "error"
                        progress.errors.append(
                            {
                                "index": -1,
                                "error": f"token verify failed: {resp.status}",
                            }
                        )
                        if on_progress:
                            on_progress(progress)
                        return progress

                    token_data = await resp.json()
                    org_id = token_data.get("org_id")
                    project_id = token_data.get("project_id")

                    if self.config.project_name:
                        clerk_user_id = token_data.get("clerk_user_id", "")
                        combined = f"{clerk_user_id}:{self.config.project_name}"
                        hash_bytes = bytearray(
                            hashlib.sha256(combined.encode()).digest()[:16]
                        )
                        # set UUID version 4 and variant bits to match server
                        hash_bytes[6] = (hash_bytes[6] & 0x0F) | 0x40
                        hash_bytes[8] = (hash_bytes[8] & 0x3F) | 0x80
                        project_id = str(uuid.UUID(bytes=bytes(hash_bytes)))

                bulk_url = f"{base_url}/api/sync/bulk/{org_id}/{project_id}"
                logger.info(
                    "starting bulk sync: %d items in %d batches",
                    len(items),
                    len(batches),
                )

                for i, batch in enumerate(batches):
                    progress.current_batch = i + 1

                    if on_progress:
                        on_progress(progress)

                    async with session.post(
                        bulk_url,
                        json={"items": batch},
                        headers={
                            "Authorization": f"Bearer {self.config.token}",
                            "Content-Type": "application/json",
                        },
                    ) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            err = f"{resp.status}: {error_text[:50]}"
                            progress.errors.append(
                                {"index": i * batch_size, "error": err}
                            )
                            logger.warning(
                                "bulk sync batch %d failed: %s",
                                i + 1,
                                error_text[:100],
                            )
                            continue

                        result = await resp.json()
                        progress.synced += result.get("synced", 0)

                        for err in result.get("errors", []):
                            progress.errors.append(
                                {
                                    "index": err["index"] + i * batch_size,
                                    "error": err["error"],
                                }
                            )

                        server_seq = result.get("server_seq", 0)
                        if server_seq > self._last_seq:
                            self._last_seq = server_seq

                        logger.debug(
                            "bulk sync batch %d/%d: synced %d items, seq=%d",
                            i + 1,
                            len(batches),
                            result.get("synced", 0),
                            server_seq,
                        )

                    if on_progress:
                        on_progress(progress)

            progress.state = "complete" if not progress.errors else "error"
            progress.completed_at = datetime.now(timezone.utc).isoformat()
            logger.info(
                "bulk sync complete: %d/%d synced, %d errors",
                progress.synced,
                progress.total,
                len(progress.errors),
            )

        except Exception as e:
            progress.state = "error"
            progress.errors.append({"index": -1, "error": str(e)})
            logger.exception("bulk sync failed: %s", e)

        if on_progress:
            on_progress(progress)

        return progress

    async def bulk_sync_streaming(
        self,
        item_iterator: Iterator[tuple[dict[str, Any], str]],
        batch_size: int = 50,
        on_progress: Callable[[BulkSyncProgress], None] | None = None,
    ) -> BulkSyncProgress:
        """Stream bulk sync from an iterator - memory efficient for monorepos.

        Unlike bulk_sync which requires all items upfront, this method
        processes items from a generator in batches, never holding more
        than batch_size items in memory at once.

        Args:
            item_iterator: Generator yielding (item_dict, item_type) tuples
                where item_type is 'file', 'memory', or 'metadata'
            batch_size: Number of items per batch (default 50)
            on_progress: Optional callback for progress updates

        Returns:
            BulkSyncProgress with final state and item counts
        """
        import hashlib
        from datetime import datetime, timezone

        import aiohttp

        progress = BulkSyncProgress(
            state="syncing",
            total=0,  # unknown upfront with streaming
            synced=0,
            current_batch=0,
            total_batches=0,  # unknown upfront
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        # track counts by type
        files_count = 0
        memories_count = 0

        if on_progress:
            on_progress(progress)

        base_url = self.config.url.rstrip("/")

        try:
            async with aiohttp.ClientSession() as session:
                # verify token and get org/project IDs
                async with session.post(
                    f"{base_url}/api/tokens/verify",
                    headers={"Authorization": f"Bearer {self.config.token}"},
                ) as resp:
                    if resp.status != 200:
                        progress.state = "error"
                        progress.errors.append(
                            {
                                "index": -1,
                                "error": f"token verify failed: {resp.status}",
                            }
                        )
                        if on_progress:
                            on_progress(progress)
                        return progress

                    token_data = await resp.json()
                    org_id = token_data.get("org_id")
                    project_id = token_data.get("project_id")

                    if self.config.project_name:
                        clerk_user_id = token_data.get("clerk_user_id", "")
                        combined = f"{clerk_user_id}:{self.config.project_name}"
                        hash_bytes = bytearray(
                            hashlib.sha256(combined.encode()).digest()[:16]
                        )
                        hash_bytes[6] = (hash_bytes[6] & 0x0F) | 0x40
                        hash_bytes[8] = (hash_bytes[8] & 0x3F) | 0x80
                        project_id = str(uuid.UUID(bytes=bytes(hash_bytes)))

                bulk_url = f"{base_url}/api/sync/bulk/{org_id}/{project_id}"
                logger.info("starting streaming bulk sync to %s", bulk_url)

                # process iterator in batches
                batch: list[dict[str, Any]] = []
                batch_item_types: list[str] = []

                for item, item_type in item_iterator:
                    batch.append(item)
                    batch_item_types.append(item_type)

                    if len(batch) >= batch_size:
                        # send this batch
                        progress.current_batch += 1
                        result = await self._send_batch(
                            session, bulk_url, batch, progress
                        )

                        if result:
                            # count by type
                            for t in batch_item_types:
                                if t == "file":
                                    files_count += 1
                                elif t == "memory":
                                    memories_count += 1

                        batch = []
                        batch_item_types = []

                        if on_progress:
                            # attach counts to progress for callback
                            progress.files_count = files_count  # type: ignore
                            progress.memories_count = memories_count  # type: ignore
                            on_progress(progress)

                # send remaining items
                if batch:
                    progress.current_batch += 1
                    result = await self._send_batch(
                        session, bulk_url, batch, progress
                    )
                    if result:
                        for t in batch_item_types:
                            if t == "file":
                                files_count += 1
                            elif t == "memory":
                                memories_count += 1

            progress.state = "complete" if not progress.errors else "error"
            progress.completed_at = datetime.now(timezone.utc).isoformat()
            progress.total = progress.synced + len(progress.errors)
            progress.total_batches = progress.current_batch

            # attach final counts
            progress.files_count = files_count  # type: ignore
            progress.memories_count = memories_count  # type: ignore

            logger.info(
                "streaming sync complete: %d synced, %d files, %d memories",
                progress.synced,
                files_count,
                memories_count,
            )

        except Exception as e:
            progress.state = "error"
            progress.errors.append({"index": -1, "error": str(e)})
            logger.exception("streaming bulk sync failed: %s", e)

        if on_progress:
            on_progress(progress)

        return progress

    async def _send_batch(
        self,
        session: Any,  # aiohttp.ClientSession
        url: str,
        batch: list[dict[str, Any]],
        progress: BulkSyncProgress,
    ) -> bool:
        """Send a single batch to the sync endpoint.

        Returns True if successful, False otherwise.
        Updates progress.synced and progress.errors in place.
        """
        try:
            async with session.post(
                url,
                json={"items": batch},
                headers={
                    "Authorization": f"Bearer {self.config.token}",
                    "Content-Type": "application/json",
                },
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    progress.errors.append(
                        {
                            "batch": progress.current_batch,
                            "error": f"HTTP {resp.status}: {error_text[:100]}",
                        }
                    )
                    logger.warning(
                        "batch %d failed: HTTP %s",
                        progress.current_batch,
                        resp.status,
                    )
                    return False

                result = await resp.json()
                progress.synced += result.get("synced", 0)

                for err in result.get("errors", []):
                    progress.errors.append(
                        {
                            "batch": progress.current_batch,
                            "error": err.get("error", "unknown"),
                        }
                    )

                server_seq = result.get("server_seq", 0)
                if server_seq > self._last_seq:
                    self._last_seq = server_seq

                logger.debug(
                    "batch %d: synced %d items, seq=%d",
                    progress.current_batch,
                    result.get("synced", 0),
                    server_seq,
                )
                return True

        except Exception as e:
            progress.errors.append(
                {"batch": progress.current_batch, "error": str(e)}
            )
            logger.warning("batch %d failed: %s", progress.current_batch, e)
            return False

    # Socket.IO event handlers

    def _on_connect(self) -> None:
        self._connected = True
        logger.debug("socket.io connected")

    def _on_disconnect(self) -> None:
        self._connected = False
        logger.debug("socket.io disconnected")

    def _on_ops_received(self, data: dict) -> None:
        """Handle incoming ops from server."""
        ops = data.get("ops", [])
        seq_start = data.get("seq_start", 0)

        if ops:
            logger.debug(
                "received %d ops starting at seq %d", len(ops), seq_start
            )
            # update our last seen seq
            for op in ops:
                server_seq = op.get("server_seq", 0)
                if server_seq > self._last_seq:
                    self._last_seq = server_seq

            # call user callback if set
            if self._on_ops:
                self._on_ops(ops)

    def _on_ack(self, data: dict) -> None:
        """Handle ack for our commands."""
        command_id = data.get("command_id")
        server_seq = data.get("server_seq", 0)

        if server_seq > self._last_seq:
            self._last_seq = server_seq

        if command_id and command_id in self._pending_acks:
            self._pending_acks[command_id].set_result(data)
            logger.debug("ack for command %s, seq=%d", command_id, server_seq)

    def _on_reject(self, data: dict) -> None:
        """Handle rejection of our commands."""
        command_id = data.get("command_id")
        reason = data.get("reason", "unknown")

        if command_id and command_id in self._pending_acks:
            self._pending_acks[command_id].set_exception(
                Exception(f"command rejected: {reason}")
            )
            logger.warning("command %s rejected: %s", command_id, reason)

    def _on_error(self, data: dict) -> None:
        """Handle errors from server."""
        code = data.get("code", "unknown")
        message = data.get("message", "")
        logger.error("sync server error: %s - %s", code, message)


_sync_client: SyncClient | None = None


def get_sync_client() -> SyncClient | None:
    """Get the global sync client instance.

    Returns None if sync is not configured.
    """
    global _sync_client
    if _sync_client is None:
        config = SyncConfig()
        if config.is_configured:
            _sync_client = SyncClient(config)
    return _sync_client


async def ensure_connected() -> SyncClient | None:
    """Ensure the sync client is connected.

    Returns the client if connected, None otherwise.
    """
    client = get_sync_client()
    if client and not client.connected:
        await client.connect()
    return client if client and client.connected else None


@dataclass
class SyncManagerStats:
    """Statistics about SyncManager activity."""

    running: bool = False
    connected: bool = False
    project_name: str = ""
    sync_url: str = ""
    # sync stats
    initial_sync_done: bool = False
    last_sync_at: str | None = None
    files_synced: int = 0
    memories_synced: int = 0
    total_syncs: int = 0
    # error tracking
    errors: list[str] = field(default_factory=list)
    # resync config
    resync_interval_seconds: int = 0


class SyncManager:
    """Background sync manager for ultrasync.web integration.

    Similar to TranscriptWatcher, this class manages a background asyncio
    task that handles:
    1. Initial full sync on connect (all indexed files + memories)
    2. Periodic re-syncs to catch any missed updates
    3. Maintaining WebSocket connection for real-time ops

    Usage:
        manager = SyncManager(
            tracker=file_tracker,
            config=SyncConfig(),
            resync_interval=300,  # 5 minutes
        )
        await manager.start()
        # ... later ...
        await manager.stop()
    """

    def __init__(
        self,
        tracker: Any,  # FileTracker - avoid circular import
        config: SyncConfig | None = None,
        resync_interval: int = 300,  # seconds between periodic resyncs
        batch_size: int = 50,
    ) -> None:
        """Initialize the sync manager.

        Args:
            tracker: FileTracker instance for accessing indexed files/symbols
            config: Sync configuration (defaults to env-based config)
            resync_interval: Seconds between periodic resyncs (0 to disable)
            batch_size: Number of items per bulk sync batch
        """
        self.tracker = tracker
        self.config = config or SyncConfig()
        self.resync_interval = resync_interval
        self.batch_size = batch_size

        self._client: SyncClient | None = None
        self._sync_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

        self.stats = SyncManagerStats(
            project_name=self.config.project_name,
            sync_url=self.config.url,
            resync_interval_seconds=resync_interval,
        )

        logger.info(
            "sync manager initialized: project=%s url=%s resync=%ds",
            self.config.project_name,
            self.config.url,
            resync_interval,
        )

    @property
    def client(self) -> SyncClient | None:
        """Get the underlying sync client."""
        return self._client

    @property
    def connected(self) -> bool:
        """Check if currently connected to sync server."""
        return self._client is not None and self._client.connected

    async def start(self) -> bool:
        """Start the sync manager background task.

        Returns:
            True if started successfully, False if not configured/enabled
        """
        _debug_log("SyncManager.start() called")

        if self._sync_task is not None:
            _debug_log("SyncManager.start() - already running")
            logger.warning("sync manager already running")
            return True

        if not self.config.is_enabled:
            _debug_log("SyncManager.start() - not enabled")
            logger.info(
                "remote sync not enabled (ULTRASYNC_REMOTE_SYNC != true)"
            )
            return False

        if not self.config.is_configured:
            _debug_log("SyncManager.start() - not configured")
            logger.warning(
                "sync enabled but not configured - set ULTRASYNC_SYNC_URL "
                "and ULTRASYNC_SYNC_TOKEN"
            )
            return False

        _debug_log(f"SyncManager.start() - connecting to {self.config.url}")
        self._stop_event.clear()
        self.stats.running = True

        # create client and connect
        self._client = SyncClient(self.config)
        connected = await self._client.connect()
        _debug_log(f"SyncManager.start() - connected={connected}")

        if not connected:
            _debug_log("SyncManager.start() - connection FAILED")
            logger.error("failed to connect to sync server")
            self.stats.errors.append("initial connection failed")
            self.stats.running = False
            return False

        self.stats.connected = True
        logger.info(
            "sync manager connected: project=%s",
            self.config.project_name,
        )

        _debug_log("SyncManager.start() - launching _sync_loop task")
        # launch background sync loop
        self._sync_task = asyncio.create_task(
            self._sync_loop(), name="sync-manager"
        )

        # add done callback to catch silent failures
        def _on_sync_task_done(task: asyncio.Task) -> None:
            if task.cancelled():
                logger.debug("sync loop task was cancelled")
            elif task.exception():
                logger.error("sync loop task crashed: %s", task.exception())
                self.stats.errors.append(
                    f"sync loop crashed: {task.exception()}"
                )
            else:
                logger.debug("sync loop task completed normally")

        self._sync_task.add_done_callback(_on_sync_task_done)

        return True

    async def stop(self) -> None:
        """Stop the sync manager and disconnect."""
        if self._sync_task is None:
            return

        logger.info("stopping sync manager...")
        self._stop_event.set()
        self.stats.running = False

        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None

        if self._client:
            await self._client.disconnect()
            self._client = None

        self.stats.connected = False
        logger.info("sync manager stopped")

    async def _sync_loop(self) -> None:
        """Main sync loop - handles initial sync and periodic resyncs."""
        _debug_log("_sync_loop STARTED")
        logger.info("_sync_loop started, beginning initial full sync...")
        try:
            # do initial full sync
            _debug_log("calling _do_full_sync...")
            progress = await self._do_full_sync()
            _debug_log(
                f"_do_full_sync returned: state={progress.state} "
                f"synced={progress.synced}/{progress.total}"
            )
            logger.info(
                "_sync_loop initial sync complete: state=%s synced=%d/%d",
                progress.state,
                progress.synced,
                progress.total,
            )
            self.stats.initial_sync_done = True

            # if resync disabled, just keep connection alive
            if self.resync_interval <= 0:
                logger.info("periodic resync disabled, waiting for stop signal")
                await self._stop_event.wait()
                return

            # periodic resync loop
            while not self._stop_event.is_set():
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.resync_interval,
                    )
                    break  # stop event was set
                except asyncio.TimeoutError:
                    # time for a resync
                    if self.connected:
                        logger.info("periodic resync starting...")
                        await self._do_full_sync()
                    else:
                        # try to reconnect
                        logger.warning("disconnected, attempting reconnect...")
                        await self._reconnect()

        except asyncio.CancelledError:
            _debug_log("_sync_loop CANCELLED")
            raise
        except Exception as e:
            error_msg = f"sync loop error: {e}"
            _debug_log(f"_sync_loop EXCEPTION: {e}")
            logger.exception(error_msg)
            self.stats.errors.append(error_msg)
            if len(self.stats.errors) > 100:
                self.stats.errors = self.stats.errors[-50:]

    async def _reconnect(self) -> bool:
        """Attempt to reconnect to sync server."""
        if self._client is None:
            self._client = SyncClient(self.config)

        connected = await self._client.connect()
        self.stats.connected = connected

        if connected:
            logger.info("reconnected to sync server")
            # do a full sync after reconnect
            await self._do_full_sync()
        else:
            self.stats.errors.append("reconnect failed")

        return connected

    def _iter_sync_items(
        self,
    ) -> Iterator[tuple[dict[str, Any], str]]:
        """Generator that yields sync items without loading all into memory.

        Yields:
            Tuple of (item_dict, item_type) where item_type is 'file' or
            'memory' for counting purposes.

        This is memory-efficient for large monorepos - only holds one item
        at a time rather than loading everything into a list.
        """
        import json as json_mod
        from datetime import datetime, timezone
        from pathlib import Path

        # yield project metadata first
        yield (
            {
                "namespace": "metadata",
                "key": "project:info",
                "op_type": "set",
                "payload": {
                    "project_name": self.config.project_name,
                    "synced_at": datetime.now(timezone.utc).isoformat(),
                },
            },
            "metadata",
        )

        # yield files one at a time
        for file_record in self.tracker.iter_files():
            path = file_record.path
            if not path:
                continue

            symbols = self.tracker.get_symbols(Path(path))
            symbol_data = [
                {
                    "name": s.name,
                    "kind": s.kind,
                    "line_start": s.line_start,
                    "line_end": s.line_end,
                }
                for s in symbols
            ]

            # parse detected contexts from JSON string
            detected_contexts: list[str] = []
            if file_record.detected_contexts:
                try:
                    detected_contexts = json_mod.loads(
                        file_record.detected_contexts
                    )
                except (json_mod.JSONDecodeError, TypeError):
                    pass

            yield (
                {
                    "namespace": "index",
                    "key": f"file:{path}",
                    "op_type": "set",
                    "payload": {
                        "path": path,
                        "symbols": symbol_data,
                        "size": file_record.size,
                        "indexed_at": file_record.indexed_at,
                        "detected_contexts": detected_contexts,
                    },
                },
                "file",
            )

        # yield memories one at a time
        for memory in self.tracker.iter_memories():
            try:
                insights = (
                    json_mod.loads(memory.insights) if memory.insights else []
                )
            except (json_mod.JSONDecodeError, TypeError):
                insights = []
            try:
                context = (
                    json_mod.loads(memory.context) if memory.context else []
                )
            except (json_mod.JSONDecodeError, TypeError):
                context = []

            yield (
                {
                    "namespace": "metadata",
                    "key": f"memory:{memory.id}",
                    "op_type": "set",
                    "payload": {
                        "id": memory.id,
                        "text": memory.text,
                        "task": memory.task,
                        "insights": insights,
                        "context": context,
                        "created_at": memory.created_at,
                    },
                },
                "memory",
            )

    async def _do_full_sync(self) -> BulkSyncProgress:
        """Perform a full sync of all indexed files and memories.

        Uses streaming to handle large monorepos efficiently - never loads
        all items into memory at once. Processes in batches of batch_size.
        """
        from datetime import datetime, timezone

        _debug_log(
            f"_do_full_sync called: client={self._client is not None} "
            f"connected={self._client.connected if self._client else False}"
        )

        if not self._client or not self._client.connected:
            _debug_log("_do_full_sync BAILING - not connected!")
            logger.warning("cannot sync - not connected")
            return BulkSyncProgress(state="error")

        # stream sync with batching - memory efficient for large repos
        progress = await self._client.bulk_sync_streaming(
            item_iterator=self._iter_sync_items(),
            batch_size=self.batch_size,
            on_progress=lambda p: logger.debug(
                "sync progress: %d synced, batch %d, files=%d memories=%d",
                p.synced,
                p.current_batch,
                getattr(p, "files_count", 0),
                getattr(p, "memories_count", 0),
            ),
        )

        # update stats from progress
        self.stats.last_sync_at = datetime.now(timezone.utc).isoformat()
        self.stats.files_synced = getattr(progress, "files_count", 0)
        self.stats.memories_synced = getattr(progress, "memories_count", 0)
        self.stats.total_syncs += 1

        if progress.errors:
            for err in progress.errors[:5]:
                self.stats.errors.append(str(err))

        _debug_log(
            f"full sync complete: synced={progress.synced} "
            f"files={self.stats.files_synced} "
            f"memories={self.stats.memories_synced}"
        )
        logger.info(
            "full sync complete: %d synced, %d files, %d memories, %d errors",
            progress.synced,
            self.stats.files_synced,
            self.stats.memories_synced,
            len(progress.errors),
        )

        return progress

    async def sync_file(self, path: str, symbols: list[dict]) -> bool:
        """Sync a single file immediately (for real-time updates).

        Args:
            path: File path
            symbols: List of symbol dicts

        Returns:
            True if synced successfully
        """
        if not self._client or not self._client.connected:
            return False

        result = await self._client.push_file_indexed(path, symbols)
        return result is not None

    async def sync_memory(
        self,
        memory_id: str,
        text: str,
        task: str | None = None,
        insights: list[str] | None = None,
        context: list[str] | None = None,
    ) -> bool:
        """Sync a single memory immediately.

        Returns:
            True if synced successfully
        """
        if not self._client or not self._client.connected:
            return False

        result = await self._client.push_memory(
            memory_id, text, task, insights, context
        )
        return result is not None

    def get_stats(self) -> SyncManagerStats:
        """Get current sync manager statistics."""
        # update connected status
        self.stats.connected = self.connected
        return self.stats
