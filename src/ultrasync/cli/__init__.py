"""ultrasync CLI - index and query codebases.

Uses tyro for type-driven CLI generation from dataclasses.
"""

from __future__ import annotations

from typing import Annotated

import tyro

from ultrasync.cli.commands.anchors import (
    AnchorsFind,
    AnchorsFindAll,
    AnchorsList,
    AnchorsScan,
    AnchorsShow,
)
from ultrasync.cli.commands.callgraph import Callgraph
from ultrasync.cli.commands.compact import Compact
from ultrasync.cli.commands.conventions import (
    ConventionsAdd,
    ConventionsCheck,
    ConventionsDelete,
    ConventionsDiscover,
    ConventionsExport,
    ConventionsGeneratePrompt,
    ConventionsImport,
    ConventionsList,
    ConventionsSearch,
    ConventionsShow,
    ConventionsStats,
)
from ultrasync.cli.commands.delete import Delete
from ultrasync.cli.commands.enrich import Enrich, EnrichClear, EnrichList
from ultrasync.cli.commands.graph import (
    GraphBootstrap,
    GraphDiff,
    GraphDot,
    GraphEdges,
    GraphExport,
    GraphGc,
    GraphImportCallgraph,
    GraphKvGet,
    GraphKvList,
    GraphKvSet,
    GraphNodes,
    GraphRelations,
    GraphStats,
)
from ultrasync.cli.commands.grep import Grep, Sgrep
from ultrasync.cli.commands.index import Index
from ultrasync.cli.commands.ir import (
    IrEndpoints,
    IrEntities,
    IrExtract,
    IrFlows,
    IrServices,
)
from ultrasync.cli.commands.keys import Keys
from ultrasync.cli.commands.mcp import Mcp
from ultrasync.cli.commands.memory import (
    MemoryDelete,
    MemoryList,
    MemorySearch,
    MemoryShow,
    MemoryStats,
    MemoryWrite,
)
from ultrasync.cli.commands.patterns import (
    PatternsList,
    PatternsLoad,
    PatternsScan,
    PatternsShow,
)
from ultrasync.cli.commands.query import Query
from ultrasync.cli.commands.repl import Repl
from ultrasync.cli.commands.show import Show, Symbols
from ultrasync.cli.commands.source import GetSource
from ultrasync.cli.commands.stack import StackComponents, StackDetect, StackHash
from ultrasync.cli.commands.stats import Stats
from ultrasync.cli.commands.threads import (
    ThreadsForFile,
    ThreadsList,
    ThreadsSearch,
    ThreadsShow,
    ThreadsStats,
)
from ultrasync.cli.commands.voyager import Voyager
from ultrasync.cli.commands.warm import Warm

# Type aliases for subcommand annotations
_Index = Annotated[Index, tyro.conf.subcommand("index")]
_Query = Annotated[Query, tyro.conf.subcommand("query")]
_Search = Annotated[Query, tyro.conf.subcommand("search")]  # alias
_Grep = Annotated[Grep, tyro.conf.subcommand("grep")]
_Sgrep = Annotated[Sgrep, tyro.conf.subcommand("sgrep")]
_Show = Annotated[Show, tyro.conf.subcommand("show")]
_Symbols = Annotated[Symbols, tyro.conf.subcommand("symbols")]
_GetSource = Annotated[GetSource, tyro.conf.subcommand("get-source")]
_Delete = Annotated[Delete, tyro.conf.subcommand("delete")]
_Callgraph = Annotated[Callgraph, tyro.conf.subcommand("callgraph")]
_Voyager = Annotated[Voyager, tyro.conf.subcommand("voyager")]
_Mcp = Annotated[Mcp, tyro.conf.subcommand("mcp")]
_Stats = Annotated[Stats, tyro.conf.subcommand("stats")]
_Keys = Annotated[Keys, tyro.conf.subcommand("keys")]
_Warm = Annotated[Warm, tyro.conf.subcommand("warm")]
_Compact = Annotated[Compact, tyro.conf.subcommand("compact")]
_Repl = Annotated[Repl, tyro.conf.subcommand("repl")]

# Patterns subcommands
_PatternsList = Annotated[PatternsList, tyro.conf.subcommand("patterns:list")]
_PatternsShow = Annotated[PatternsShow, tyro.conf.subcommand("patterns:show")]
_PatternsLoad = Annotated[PatternsLoad, tyro.conf.subcommand("patterns:load")]
_PatternsScan = Annotated[PatternsScan, tyro.conf.subcommand("patterns:scan")]

# Anchors subcommands
_AnchorsList = Annotated[AnchorsList, tyro.conf.subcommand("anchors:list")]
_AnchorsShow = Annotated[AnchorsShow, tyro.conf.subcommand("anchors:show")]
_AnchorsScan = Annotated[AnchorsScan, tyro.conf.subcommand("anchors:scan")]
_AnchorsFind = Annotated[AnchorsFind, tyro.conf.subcommand("anchors:find")]
_AnchorsFindAll = Annotated[
    AnchorsFindAll, tyro.conf.subcommand("anchors:find-all")
]

# Threads subcommands
_ThreadsList = Annotated[ThreadsList, tyro.conf.subcommand("threads:list")]
_ThreadsShow = Annotated[ThreadsShow, tyro.conf.subcommand("threads:show")]
_ThreadsStats = Annotated[ThreadsStats, tyro.conf.subcommand("threads:stats")]
_ThreadsForFile = Annotated[
    ThreadsForFile, tyro.conf.subcommand("threads:for-file")
]
_ThreadsSearch = Annotated[
    ThreadsSearch, tyro.conf.subcommand("threads:search")
]

# Memory subcommands
_MemoryList = Annotated[MemoryList, tyro.conf.subcommand("memory:list")]
_MemoryShow = Annotated[MemoryShow, tyro.conf.subcommand("memory:show")]
_MemorySearch = Annotated[MemorySearch, tyro.conf.subcommand("memory:search")]
_MemoryStats = Annotated[MemoryStats, tyro.conf.subcommand("memory:stats")]
_MemoryWrite = Annotated[MemoryWrite, tyro.conf.subcommand("memory:write")]
_MemoryDelete = Annotated[MemoryDelete, tyro.conf.subcommand("memory:delete")]

# Graph subcommands
_GraphStats = Annotated[GraphStats, tyro.conf.subcommand("graph:stats")]
_GraphBootstrap = Annotated[
    GraphBootstrap, tyro.conf.subcommand("graph:bootstrap")
]
_GraphNodes = Annotated[GraphNodes, tyro.conf.subcommand("graph:nodes")]
_GraphEdges = Annotated[GraphEdges, tyro.conf.subcommand("graph:edges")]
_GraphKvList = Annotated[GraphKvList, tyro.conf.subcommand("graph:kv:list")]
_GraphKvSet = Annotated[GraphKvSet, tyro.conf.subcommand("graph:kv:set")]
_GraphKvGet = Annotated[GraphKvGet, tyro.conf.subcommand("graph:kv:get")]
_GraphDiff = Annotated[GraphDiff, tyro.conf.subcommand("graph:diff")]
_GraphDot = Annotated[GraphDot, tyro.conf.subcommand("graph:dot")]
_GraphExport = Annotated[GraphExport, tyro.conf.subcommand("graph:export")]
_GraphGc = Annotated[GraphGc, tyro.conf.subcommand("graph:gc")]
_GraphImportCallgraph = Annotated[
    GraphImportCallgraph, tyro.conf.subcommand("graph:import-callgraph")
]
_GraphRelations = Annotated[
    GraphRelations, tyro.conf.subcommand("graph:relations")
]

# IR subcommands
_IrExtract = Annotated[IrExtract, tyro.conf.subcommand("ir:extract")]
_IrEntities = Annotated[IrEntities, tyro.conf.subcommand("ir:entities")]
_IrEndpoints = Annotated[IrEndpoints, tyro.conf.subcommand("ir:endpoints")]
_IrServices = Annotated[IrServices, tyro.conf.subcommand("ir:services")]
_IrFlows = Annotated[IrFlows, tyro.conf.subcommand("ir:flows")]

# Stack subcommands
_StackDetect = Annotated[StackDetect, tyro.conf.subcommand("stack")]
_StackComponents = Annotated[
    StackComponents, tyro.conf.subcommand("stack:components")
]
_StackHash = Annotated[StackHash, tyro.conf.subcommand("stack:hash")]

# Enrich subcommands
_Enrich = Annotated[Enrich, tyro.conf.subcommand("enrich")]
_EnrichList = Annotated[EnrichList, tyro.conf.subcommand("enrich:list")]
_EnrichClear = Annotated[EnrichClear, tyro.conf.subcommand("enrich:clear")]

# Conventions subcommands
_ConventionsList = Annotated[
    ConventionsList, tyro.conf.subcommand("conventions:list")
]
_ConventionsShow = Annotated[
    ConventionsShow, tyro.conf.subcommand("conventions:show")
]
_ConventionsSearch = Annotated[
    ConventionsSearch, tyro.conf.subcommand("conventions:search")
]
_ConventionsAdd = Annotated[
    ConventionsAdd, tyro.conf.subcommand("conventions:add")
]
_ConventionsDelete = Annotated[
    ConventionsDelete, tyro.conf.subcommand("conventions:delete")
]
_ConventionsStats = Annotated[
    ConventionsStats, tyro.conf.subcommand("conventions:stats")
]
_ConventionsDiscover = Annotated[
    ConventionsDiscover, tyro.conf.subcommand("conventions:discover")
]
_ConventionsCheck = Annotated[
    ConventionsCheck, tyro.conf.subcommand("conventions:check")
]
_ConventionsExport = Annotated[
    ConventionsExport, tyro.conf.subcommand("conventions:export")
]
_ConventionsImport = Annotated[
    ConventionsImport, tyro.conf.subcommand("conventions:import")
]
_ConventionsGeneratePrompt = Annotated[
    ConventionsGeneratePrompt,
    tyro.conf.subcommand("conventions:generate-prompt"),
]

# Top-level commands using pipe syntax
Command = (
    _Index
    | _Query
    | _Search
    | _Grep
    | _Sgrep
    | _Show
    | _Symbols
    | _GetSource
    | _Delete
    | _Callgraph
    | _Voyager
    | _Mcp
    | _Stats
    | _Keys
    | _Warm
    | _Compact
    | _Repl
    | _PatternsList
    | _PatternsShow
    | _PatternsLoad
    | _PatternsScan
    | _AnchorsList
    | _AnchorsShow
    | _AnchorsScan
    | _AnchorsFind
    | _AnchorsFindAll
    | _ThreadsList
    | _ThreadsShow
    | _ThreadsStats
    | _ThreadsForFile
    | _ThreadsSearch
    | _MemoryList
    | _MemoryShow
    | _MemorySearch
    | _MemoryStats
    | _MemoryWrite
    | _MemoryDelete
    | _GraphStats
    | _GraphBootstrap
    | _GraphNodes
    | _GraphEdges
    | _GraphKvList
    | _GraphKvSet
    | _GraphKvGet
    | _GraphDiff
    | _GraphDot
    | _GraphExport
    | _GraphGc
    | _GraphImportCallgraph
    | _GraphRelations
    | _IrExtract
    | _IrEntities
    | _IrEndpoints
    | _IrServices
    | _IrFlows
    | _StackDetect
    | _StackComponents
    | _StackHash
    | _Enrich
    | _EnrichList
    | _EnrichClear
    | _ConventionsList
    | _ConventionsShow
    | _ConventionsSearch
    | _ConventionsAdd
    | _ConventionsDelete
    | _ConventionsStats
    | _ConventionsDiscover
    | _ConventionsCheck
    | _ConventionsExport
    | _ConventionsImport
    | _ConventionsGeneratePrompt
)


def main() -> int:
    """Entry point for the CLI."""
    # configure structlog (respects ULTRASYNC_DEBUG env var)
    from ultrasync.logging_config import configure_logging

    configure_logging()

    try:
        cmd = tyro.cli(
            Command,
            prog="ultrasync",
            description="Index and query codebases with semantic search.",
        )
        return cmd.run()
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        from ultrasync import console

        console.error(str(e))
        return 1
