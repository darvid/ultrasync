"""Auto-extract memories from assistant responses.

Extracts meaningful snippets (conclusions, decisions, findings) from
assistant text - NOT full messages. Filters out process narration
and only creates memories when there's a real insight.

Uses Hyperscan for fast multi-pattern matching with a pre-compiled
database containing hundreds of insight patterns.
"""

import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal

import structlog

from ultrasync.hyperscan_search import HyperscanSearch
from ultrasync.keys import hash64_file_key

logger = structlog.get_logger(__name__)

# Environment variable names
ENV_MEMORY_EXTRACTION = "ULTRASYNC_MEMORY_EXTRACTION"
ENV_MEMORY_AGGRESSIVENESS = "ULTRASYNC_MEMORY_AGGRESSIVENESS"

# Aggressiveness levels
AggressivenessLevel = Literal["conservative", "moderate", "aggressive"]

# Minimum confidence to create a memory
MIN_CONFIDENCE_THRESHOLD = 0.3

# Higher confidence threshold for exploration-only turns
# (slightly above base 0.3 to filter low-value exploration noise)
EXPLORATION_CONFIDENCE_THRESHOLD = 0.35

# Max snippet length (chars)
MAX_SNIPPET_LENGTH = 300

# Tools that indicate meaningful work was done
ACTION_TOOLS = {"Write", "Edit", "Bash", "mcp__acp__Write", "mcp__acp__Edit"}

# Tools that indicate exploration/discovery (valuable, need higher bar)
EXPLORATION_TOOLS = {
    "Read",
    "Grep",
    "Glob",
    "mcp__acp__Read",
    "mcp__ultrasync__search",
}

# =============================================================================
# PATTERN DEFINITIONS
# =============================================================================
# Organized by category. Each tuple: (pattern_bytes, insight_type, confidence)
# Patterns use (?i) for case-insensitive matching
# Confidence levels:
#   0.4+ = strong signal, high-value insight
#   0.3  = medium signal, likely valuable
#   0.2  = weak signal, needs corroboration
#   0.15 = very weak, only if multiple match
# =============================================================================

# -----------------------------------------------------------------------------
# DECISION/CHOICE PATTERNS - when a choice was made
# -----------------------------------------------------------------------------
DECISION_PATTERNS: list[tuple[bytes, str, float]] = [
    # Strong decision language
    (rb"(?i)\bdecided to\b", "insight:decision", 0.35),
    (rb"(?i)\bgoing (with|to use)\b", "insight:decision", 0.3),
    (rb"(?i)\bchose to\b", "insight:decision", 0.35),
    (rb"(?i)\bopting for\b", "insight:decision", 0.35),
    (rb"(?i)\bsettled on\b", "insight:decision", 0.35),
    (rb"(?i)\bpicked\b.{0,15}\b(over|instead)\b", "insight:decision", 0.35),
    (rb"(?i)\bwent with\b", "insight:decision", 0.3),
    (
        rb"(?i)\bselected\b.{0,10}\b(approach|method|solution)\b",
        "insight:decision",
        0.3,
    ),
    # Approach/strategy
    (rb"(?i)\bthe approach is\b", "insight:decision", 0.3),
    (rb"(?i)\bthe strategy is\b", "insight:decision", 0.3),
    (rb"(?i)\bthe plan is\b", "insight:decision", 0.25),
    (rb"(?i)\bbest option is\b", "insight:decision", 0.35),
    (rb"(?i)\bpreferred (way|approach|method) is\b", "insight:decision", 0.3),
    (rb"(?i)\busing\b.{0,20}\binstead of\b", "insight:decision", 0.35),
    # Implementation decisions
    (rb"(?i)\bwill (use|implement|go with)\b", "insight:decision", 0.25),
    (rb"(?i)\bimplementing (it )?(with|using|via)\b", "insight:decision", 0.25),
    (rb"(?i)\bthe (right|correct|proper) way\b", "insight:decision", 0.25),
]

# -----------------------------------------------------------------------------
# BUG/PROBLEM PATTERNS - root cause analysis
# -----------------------------------------------------------------------------
BUG_PATTERNS: list[tuple[bytes, str, float]] = [
    # Root cause identification (high value)
    (rb"(?i)\bthe bug (was|is)\b", "insight:decision", 0.45),
    (rb"(?i)\broot cause\b", "insight:decision", 0.45),
    (rb"(?i)\bissue (was|is) (that|caused)\b", "insight:decision", 0.4),
    (rb"(?i)\bproblem (was|is) (that|caused)\b", "insight:decision", 0.4),
    (rb"(?i)\bcaused by\b", "insight:decision", 0.35),
    (
        rb"(?i)\bdue to\b.{0,30}\b(bug|issue|error|problem)\b",
        "insight:decision",
        0.35,
    ),
    (rb"(?i)\bstems from\b", "insight:decision", 0.35),
    (rb"(?i)\bthe culprit (is|was)\b", "insight:decision", 0.4),
    (rb"(?i)\boffending (code|line|function)\b", "insight:decision", 0.35),
    # Error conditions
    (rb"(?i)\b(the )?error occurs when\b", "insight:decision", 0.35),
    (rb"(?i)\bfails when\b", "insight:decision", 0.3),
    (rb"(?i)\bbreaks when\b", "insight:decision", 0.3),
    (rb"(?i)\bcrashes (when|if|because)\b", "insight:decision", 0.35),
    # Discovery of issue
    (rb"(?i)\bfound the (bug|issue|problem)\b", "insight:decision", 0.4),
    (
        rb"(?i)\bidentified the (bug|issue|problem|cause)\b",
        "insight:decision",
        0.4,
    ),
    (
        rb"(?i)\bdiscovered (that|the)\b.{0,20}\b(bug|issue|problem)\b",
        "insight:decision",
        0.35,
    ),
    (rb"(?i)\btracked (it )?down to\b", "insight:decision", 0.35),
    (rb"(?i)\bnarrowed (it )?down to\b", "insight:decision", 0.3),
    # Bug characteristics
    (rb"(?i)\brace condition\b", "insight:decision", 0.35),
    (rb"(?i)\bmemory leak\b", "insight:decision", 0.35),
    (rb"(?i)\bnull pointer\b", "insight:decision", 0.3),
    (rb"(?i)\boff[- ]by[- ]one\b", "insight:decision", 0.35),
    (
        rb"(?i)\bedge case\b.{0,15}\b(bug|issue|problem|fail)\b",
        "insight:decision",
        0.3,
    ),
]

# -----------------------------------------------------------------------------
# SOLUTION/FIX PATTERNS - how something was resolved
# -----------------------------------------------------------------------------
FIX_PATTERNS: list[tuple[bytes, str, float]] = [
    # Direct fix statements
    (rb"(?i)\bthe fix (is|was)\b", "insight:decision", 0.4),
    (rb"(?i)\bfixed (by|it by|this by)\b", "insight:decision", 0.4),
    (rb"(?i)\bsolved (by|it by|this by)\b", "insight:decision", 0.4),
    (rb"(?i)\bresolved (by|it by|this by)\b", "insight:decision", 0.4),
    (rb"(?i)\bthe solution (is|was)\b", "insight:decision", 0.4),
    (rb"(?i)\bworkaround (is|was)\b", "insight:decision", 0.35),
    (rb"(?i)\bcan be fixed by\b", "insight:decision", 0.35),
    (rb"(?i)\bto fix (this|it|the)\b", "insight:decision", 0.25),
    # Resolution confirmation
    (rb"(?i)\bworks now\b", "insight:decision", 0.3),
    (rb"(?i)\bworking (now|after)\b", "insight:decision", 0.25),
    (rb"(?i)\bresolved (the|this)\b", "insight:decision", 0.3),
    (rb"(?i)\bpatched\b", "insight:decision", 0.25),
    (
        rb"(?i)\bcorrected\b.{0,15}\b(issue|bug|error|problem)\b",
        "insight:decision",
        0.3,
    ),
]

# -----------------------------------------------------------------------------
# CONSTRAINT/LIMITATION PATTERNS - what can't be done
# -----------------------------------------------------------------------------
CONSTRAINT_PATTERNS: list[tuple[bytes, str, float]] = [
    # Can't/won't statements
    (rb"(?i)\bcan'?t (be done|do this|use|work)\b", "insight:constraint", 0.3),
    (rb"(?i)\bcannot (be done|do this|use|work)\b", "insight:constraint", 0.3),
    (rb"(?i)\bwon'?t work\b", "insight:constraint", 0.3),
    (rb"(?i)\bdoesn'?t support\b", "insight:constraint", 0.3),
    (
        rb"(?i)\bisn'?t (supported|possible|allowed)\b",
        "insight:constraint",
        0.3,
    ),
    # Limitation language
    (rb"(?i)\blimitation (is|of)\b", "insight:constraint", 0.35),
    (rb"(?i)\brestricted (to|by)\b", "insight:constraint", 0.3),
    (
        rb"(?i)\bnot (possible|supported|allowed|available)\b",
        "insight:constraint",
        0.3,
    ),
    (rb"(?i)\bimpossible to\b", "insight:constraint", 0.35),
    (rb"(?i)\bno way to\b", "insight:constraint", 0.3),
    # Blocking conditions
    (rb"(?i)\bblocked by\b", "insight:constraint", 0.35),
    (rb"(?i)\bprevented by\b", "insight:constraint", 0.3),
    (rb"(?i)\bincompatible with\b", "insight:constraint", 0.35),
    (rb"(?i)\bconflicts with\b", "insight:constraint", 0.3),
    # Requirements
    (rb"(?i)\bonly works with\b", "insight:constraint", 0.3),
    (
        rb"(?i)\brequires\b.{0,20}\b(version|at least|minimum)\b",
        "insight:constraint",
        0.3,
    ),
    (rb"(?i)\bmust (have|use|be)\b", "insight:constraint", 0.25),
    (
        rb"(?i)\bneeds\b.{0,15}\b(to be|first|before)\b",
        "insight:constraint",
        0.25,
    ),
    # Deprecation
    (rb"(?i)\bdeprecated\b", "insight:constraint", 0.35),
    (rb"(?i)\bnot recommended\b", "insight:constraint", 0.3),
    (rb"(?i)\bavoid using\b", "insight:constraint", 0.3),
    (
        rb"(?i)\bdon'?t use\b.{0,20}\b(anymore|instead)\b",
        "insight:constraint",
        0.3,
    ),
    (
        rb"(?i)\blegacy\b.{0,15}\b(api|code|system)\b",
        "insight:constraint",
        0.25,
    ),
    # Rate limits and quotas
    (rb"(?i)\brate limit\b", "insight:constraint", 0.4),
    (
        rb"(?i)\bquota\b.{0,10}\b(limit|exceed|per)\b",
        "insight:constraint",
        0.35,
    ),
    (
        rb"(?i)\b\d+\s*(requests?|calls?)\s*(per|/)\s*(minute|second|hour)\b",
        "insight:constraint",
        0.4,
    ),
    (rb"(?i)\bthrottle\b", "insight:constraint", 0.35),
    (rb"(?i)\bbackoff\b", "insight:constraint", 0.3),
    (rb"(?i)\bretry.{0,10}(limit|after)\b", "insight:constraint", 0.3),
]

# -----------------------------------------------------------------------------
# TRADEOFF PATTERNS - compromises and alternatives
# -----------------------------------------------------------------------------
TRADEOFF_PATTERNS: list[tuple[bytes, str, float]] = [
    # Explicit tradeoff language
    (rb"(?i)\btrade-?off\b", "insight:tradeoff", 0.4),
    (rb"(?i)\bcompromise\b.{0,15}\b(between|is)\b", "insight:tradeoff", 0.35),
    # Alternative choices
    (rb"(?i)\binstead of\b", "insight:tradeoff", 0.25),
    (rb"(?i)\brather than\b", "insight:tradeoff", 0.25),
    (rb"(?i)\bat the (expense|cost) of\b", "insight:tradeoff", 0.4),
    (rb"(?i)\bsacrificing\b", "insight:tradeoff", 0.35),
    (rb"(?i)\bin exchange for\b", "insight:tradeoff", 0.35),
    # Comparison
    (rb"(?i)\bpros and cons\b", "insight:tradeoff", 0.35),
    (rb"(?i)\badvantages and disadvantages\b", "insight:tradeoff", 0.35),
    (rb"(?i)\bbetter for\b.{0,20}\b(but|worse)\b", "insight:tradeoff", 0.35),
    (
        rb"(?i)\bfaster but\b.{0,15}\b(more|less|harder)\b",
        "insight:tradeoff",
        0.3,
    ),
    (
        rb"(?i)\bsimpler but\b.{0,15}\b(more|less|slower)\b",
        "insight:tradeoff",
        0.3,
    ),
    # Balancing
    (rb"(?i)\bbalance between\b", "insight:tradeoff", 0.3),
    (rb"(?i)\bweighing\b.{0,15}\b(against|vs)\b", "insight:tradeoff", 0.3),
]

# -----------------------------------------------------------------------------
# PITFALL/WARNING PATTERNS - things to watch out for
# -----------------------------------------------------------------------------
PITFALL_PATTERNS: list[tuple[bytes, str, float]] = [
    # Explicit warnings
    (rb"(?i)\bbe careful\b", "insight:pitfall", 0.4),
    (rb"(?i)\bwatch out\b", "insight:pitfall", 0.4),
    (rb"(?i)\bgotcha\b", "insight:pitfall", 0.45),
    (rb"(?i)\bcaveat\b", "insight:pitfall", 0.4),
    (rb"(?i)\bwarning\b", "insight:pitfall", 0.3),
    (rb"(?i)\bcaution\b", "insight:pitfall", 0.3),
    # Common mistakes
    (rb"(?i)\bcommon mistake\b", "insight:pitfall", 0.4),
    (rb"(?i)\beasy to miss\b", "insight:pitfall", 0.35),
    (
        rb"(?i)\bsubtle\b.{0,15}\b(bug|issue|problem|error)\b",
        "insight:pitfall",
        0.35,
    ),
    (rb"(?i)\btricky\b.{0,15}\b(part|thing|issue)\b", "insight:pitfall", 0.3),
    (
        rb"(?i)\bhidden\b.{0,15}\b(bug|issue|problem|gotcha)\b",
        "insight:pitfall",
        0.35,
    ),
    # Reminders
    (rb"(?i)\bdon'?t forget\b", "insight:pitfall", 0.3),
    (rb"(?i)\bmake sure (to|you)\b", "insight:pitfall", 0.25),
    (rb"(?i)\bimportant to (note|remember)\b", "insight:pitfall", 0.3),
    (rb"(?i)\bremember (to|that)\b", "insight:pitfall", 0.2),
    # Danger words
    (rb"(?i)\bpitfall\b", "insight:pitfall", 0.45),
    (rb"(?i)\btrap\b.{0,10}\b(is|here|for)\b", "insight:pitfall", 0.35),
    (rb"(?i)\bfootgun\b", "insight:pitfall", 0.45),
    (rb"(?i)\bfoot[- ]?gun\b", "insight:pitfall", 0.45),
    # Unexpected behavior
    (rb"(?i)\bsurprising(ly)?\b", "insight:pitfall", 0.25),
    (rb"(?i)\bunexpected(ly)?\b", "insight:pitfall", 0.25),
    (rb"(?i)\bcounterintuitive\b", "insight:pitfall", 0.35),
    (rb"(?i)\bnon-?obvious\b", "insight:pitfall", 0.3),
    (rb"(?i)\bunintuitive\b", "insight:pitfall", 0.35),
]

# -----------------------------------------------------------------------------
# ASSUMPTION PATTERNS - stated assumptions
# -----------------------------------------------------------------------------
ASSUMPTION_PATTERNS: list[tuple[bytes, str, float]] = [
    (rb"(?i)\bassuming (that)?\b", "insight:assumption", 0.3),
    (rb"(?i)\bassumes (that)?\b", "insight:assumption", 0.3),
    (rb"(?i)\bassumption (is|here|being)\b", "insight:assumption", 0.35),
    (rb"(?i)\bprovided that\b", "insight:assumption", 0.3),
    (rb"(?i)\bgiven that\b", "insight:assumption", 0.25),
    (rb"(?i)\bif\b.{0,30}\bthen\b.{0,30}\bshould\b", "insight:assumption", 0.2),
    (rb"(?i)\bexpecting (that)?\b", "insight:assumption", 0.25),
    (rb"(?i)\bpresume\b", "insight:assumption", 0.3),
    (rb"(?i)\btaking for granted\b", "insight:assumption", 0.35),
]

# -----------------------------------------------------------------------------
# DISCOVERY/LOCATION PATTERNS - where things are
# -----------------------------------------------------------------------------
DISCOVERY_PATTERNS: list[tuple[bytes, str, float]] = [
    # Location statements
    (
        rb"(?i)\bis (located |defined |implemented )?in\b.{0,30}"
        rb"\.(py|ts|js|tsx|jsx|rs|go|java|cpp|c|rb)\b",
        "insight:discovery",
        0.35,
    ),
    (
        rb"(?i)\blives in\b.{0,20}\.(py|ts|js|tsx|jsx|rs|go)\b",
        "insight:discovery",
        0.35,
    ),
    (
        rb"(?i)\bfound in\b.{0,20}\.(py|ts|js|tsx|jsx|rs|go)\b",
        "insight:discovery",
        0.3,
    ),
    (rb"(?i)\bresides in\b", "insight:discovery", 0.3),
    (rb"(?i)\bstored in\b", "insight:discovery", 0.25),
    (
        rb"(?i)\bthe\b.{0,15}\b(is|are) (at|in)\b.{0,20}/",
        "insight:discovery",
        0.25,
    ),
    (rb"(?i)\bcan be found (at|in)\b", "insight:discovery", 0.3),
    # Implementation discovery
    (rb"(?i)\bimplemented in\b", "insight:discovery", 0.3),
    (rb"(?i)\bhandled by\b", "insight:discovery", 0.3),
    (rb"(?i)\bmanaged by\b", "insight:discovery", 0.3),
    (rb"(?i)\bowned by\b", "insight:discovery", 0.25),
    (rb"(?i)\bresponsible for\b", "insight:discovery", 0.3),
    # Code structure
    (
        rb"(?i)\b(the|this) (code|logic|implementation|handler)\b"
        rb".{0,20}\b(uses?|implements?|handles?)\b",
        "insight:discovery",
        0.3,
    ),
    (
        rb"(?i)\b(the|this) (function|method|class)\b"
        rb".{0,15}\b(does|handles|manages)\b",
        "insight:discovery",
        0.25,
    ),
]

# -----------------------------------------------------------------------------
# ARCHITECTURE/STRUCTURE PATTERNS - how things are organized
# -----------------------------------------------------------------------------
ARCHITECTURE_PATTERNS: list[tuple[bytes, str, float]] = [
    # Architecture language
    (rb"(?i)\barchitecture (is|uses|follows)\b", "insight:discovery", 0.35),
    (rb"(?i)\bdesign pattern\b", "insight:discovery", 0.35),
    (rb"(?i)\b(the|this) pattern (is|used)\b", "insight:discovery", 0.3),
    (rb"(?i)\bstructured (as|like|around)\b", "insight:discovery", 0.3),
    # Architecture types (catch "uses a X architecture")
    (
        rb"(?i)\b(layered|microservice|monolith|hexagonal|clean|mvc|mvvm)\b"
        rb".{0,10}\barchitecture\b",
        "insight:discovery",
        0.4,
    ),
    (
        rb"(?i)\barchitecture\b.{0,10}"
        rb"\b(layered|microservice|monolith|hexagonal|clean)\b",
        "insight:discovery",
        0.4,
    ),
    # Code organization patterns
    (rb"(?i)\bcontrollers?\b.{0,15}\bservices?\b", "insight:discovery", 0.3),
    (rb"(?i)\bservices?\b.{0,15}\brepositories?\b", "insight:discovery", 0.3),
    (
        rb"(?i)\b(uses|follows|implements)\b.{0,10}\bpattern\b",
        "insight:discovery",
        0.3,
    ),
    # Flow descriptions
    (rb"(?i)\bdata ?flow\b", "insight:discovery", 0.3),
    (rb"(?i)\bcontrol ?flow\b", "insight:discovery", 0.3),
    (rb"(?i)\bcall (graph|chain|stack)\b", "insight:discovery", 0.3),
    (rb"(?i)\brequest flow\b", "insight:discovery", 0.3),
    (rb"(?i)\bexecution (path|flow)\b", "insight:discovery", 0.3),
    # Layers and components
    (rb"(?i)\b(the|this) layer\b", "insight:discovery", 0.25),
    (
        rb"(?i)\b(the|this) module\b.{0,15}\b(handles|manages|is)\b",
        "insight:discovery",
        0.25,
    ),
    (
        rb"(?i)\b(the|this) component\b.{0,15}\b(handles|manages|is)\b",
        "insight:discovery",
        0.25,
    ),
    (
        rb"(?i)\b(the|this) service\b.{0,15}\b(handles|manages|is)\b",
        "insight:discovery",
        0.25,
    ),
    # Separation of concerns
    (rb"(?i)\bseparation of concerns\b", "insight:discovery", 0.35),
    (rb"(?i)\bsingle responsibility\b", "insight:discovery", 0.35),
    (rb"(?i)\bdependency injection\b", "insight:discovery", 0.35),
]

# -----------------------------------------------------------------------------
# RELATIONSHIP PATTERNS - how things connect
# -----------------------------------------------------------------------------
RELATIONSHIP_PATTERNS: list[tuple[bytes, str, float]] = [
    # Calls and invocations
    (rb"(?i)\bcalls\b.{0,15}\b(to|into|from)\b", "insight:discovery", 0.25),
    (rb"(?i)\binvokes\b", "insight:discovery", 0.25),
    (rb"(?i)\btriggers\b.{0,10}\b(the|a|an)\b", "insight:discovery", 0.25),
    # Inheritance and implementation
    (rb"(?i)\bextends\b.{0,15}\b(the|from)\b", "insight:discovery", 0.25),
    (
        rb"(?i)\bimplements\b.{0,15}\b(the|interface)\b",
        "insight:discovery",
        0.25,
    ),
    (rb"(?i)\binherits from\b", "insight:discovery", 0.3),
    # Dependencies
    (rb"(?i)\bdepends on\b", "insight:discovery", 0.3),
    (rb"(?i)\bimports?\b.{0,20}\bfrom\b", "insight:discovery", 0.2),
    (rb"(?i)\brequires\b.{0,10}\b(the|a|an)\b", "insight:discovery", 0.2),
    # Communication
    (rb"(?i)\bcommunicates with\b", "insight:discovery", 0.3),
    (rb"(?i)\btalks to\b", "insight:discovery", 0.25),
    (rb"(?i)\bsends (to|data to)\b", "insight:discovery", 0.25),
    (rb"(?i)\breceives from\b", "insight:discovery", 0.25),
]

# -----------------------------------------------------------------------------
# SUMMARY/CONCLUSION PATTERNS - wrapping up findings
# -----------------------------------------------------------------------------
SUMMARY_PATTERNS: list[tuple[bytes, str, float]] = [
    # Explicit summary
    (rb"(?i)\bin summary\b", "insight:discovery", 0.4),
    (rb"(?i)\bto summarize\b", "insight:discovery", 0.4),
    (rb"(?i)\bin conclusion\b", "insight:discovery", 0.4),
    (rb"(?i)\bsumming up\b", "insight:discovery", 0.35),
    # Key points
    (
        rb"(?i)\bthe (main|key) (point|takeaway|thing)\b",
        "insight:discovery",
        0.35,
    ),
    (rb"(?i)\bbottom line\b", "insight:discovery", 0.35),
    (rb"(?i)\btl;?dr\b", "insight:discovery", 0.4),
    (rb"(?i)\bin short\b", "insight:discovery", 0.3),
    # Overall statements
    (rb"(?i)\boverall,\b", "insight:discovery", 0.25),
    (rb"(?i)\bessentially,\b", "insight:discovery", 0.25),
    (rb"(?i)\bbasically,\b", "insight:discovery", 0.2),
    (rb"(?i)\bfundamentally\b", "insight:discovery", 0.25),
]

# -----------------------------------------------------------------------------
# REALIZATION/UNDERSTANDING PATTERNS - aha moments
# -----------------------------------------------------------------------------
REALIZATION_PATTERNS: list[tuple[bytes, str, float]] = [
    (rb"(?i)\brealized (that)?\b", "insight:decision", 0.35),
    (rb"(?i)\bunderstood (that|why)\b", "insight:decision", 0.3),
    (rb"(?i)\bfigured out\b", "insight:decision", 0.35),
    (rb"(?i)\bturns out\b", "insight:decision", 0.35),
    (rb"(?i)\bit appears (that)?\b", "insight:decision", 0.25),
    (rb"(?i)\bseems like\b", "insight:assumption", 0.2),
    (rb"(?i)\bnow I (see|understand)\b", "insight:decision", 0.3),
    (rb"(?i)\bmakes sense now\b", "insight:decision", 0.3),
    (rb"(?i)\baha\b", "insight:decision", 0.3),
    (rb"(?i)\bthe (key|trick|secret) (is|was)\b", "insight:decision", 0.35),
]

# -----------------------------------------------------------------------------
# RECOMMENDATION PATTERNS - what should be done
# -----------------------------------------------------------------------------
RECOMMENDATION_PATTERNS: list[tuple[bytes, str, float]] = [
    (rb"(?i)\brecommend (that|using)?\b", "insight:decision", 0.35),
    (rb"(?i)\bsuggest (that|using)?\b", "insight:decision", 0.3),
    (rb"(?i)\badvise (that|using)?\b", "insight:decision", 0.35),
    (rb"(?i)\bshould (use|consider|try)\b", "insight:decision", 0.25),
    (rb"(?i)\bbetter to\b", "insight:decision", 0.25),
    (rb"(?i)\bprefer\b.{0,10}\b(to|over)\b", "insight:decision", 0.25),
    (rb"(?i)\bbest practice\b", "insight:decision", 0.35),
    (rb"(?i)\bidiomatic (way)?\b", "insight:decision", 0.3),
    (rb"(?i)\bconventional (way|approach)\b", "insight:decision", 0.3),
    (rb"(?i)\bthe (right|proper|correct) way\b", "insight:decision", 0.3),
]

# -----------------------------------------------------------------------------
# VERIFICATION/CONFIRMATION PATTERNS - validated findings
# -----------------------------------------------------------------------------
VERIFICATION_PATTERNS: list[tuple[bytes, str, float]] = [
    (rb"(?i)\bconfirmed (that)?\b", "insight:decision", 0.35),
    (rb"(?i)\bverified (that)?\b", "insight:decision", 0.35),
    (rb"(?i)\bvalidated\b", "insight:decision", 0.3),
    (rb"(?i)\btested and\b", "insight:decision", 0.25),
    (rb"(?i)\bworks as expected\b", "insight:decision", 0.3),
    (rb"(?i)\bbehaves correctly\b", "insight:decision", 0.3),
    (
        rb"(?i)\bmatches\b.{0,15}\b(expected|spec|requirement)\b",
        "insight:decision",
        0.25,
    ),
    (rb"(?i)\baligns with\b", "insight:discovery", 0.25),
    (rb"(?i)\bconsistent with\b", "insight:discovery", 0.25),
]

# -----------------------------------------------------------------------------
# EXPLANATION/REASON PATTERNS - why something is
# -----------------------------------------------------------------------------
EXPLANATION_PATTERNS: list[tuple[bytes, str, float]] = [
    (rb"(?i)\bthe reason (is|was|being)\b", "insight:decision", 0.35),
    (rb"(?i)\bthis is (because|why|due to)\b", "insight:decision", 0.3),
    (rb"(?i)\bthat'?s (because|why)\b", "insight:decision", 0.3),
    (rb"(?i)\bexplains why\b", "insight:decision", 0.35),
    (rb"(?i)\bdue to the fact\b", "insight:decision", 0.3),
    (rb"(?i)\bowing to\b", "insight:decision", 0.25),
    (rb"(?i)\bthe cause (is|was)\b", "insight:decision", 0.35),
    (rb"(?i)\bwhat'?s happening (is|here)\b", "insight:decision", 0.3),
]

# -----------------------------------------------------------------------------
# OUTCOME PATTERNS - task completion signals
# -----------------------------------------------------------------------------
OUTCOME_PATTERNS: list[tuple[bytes, str, float]] = [
    (rb"(?i)\bdone!?\b", "insight:decision", 0.2),
    (rb"(?i)\bfinished!?\b", "insight:decision", 0.2),
    (rb"(?i)\bcompleted!?\b", "insight:decision", 0.2),
    (rb"(?i)\bimplemented!?\b", "insight:decision", 0.2),
    (rb"(?i)\bfixed!?\b", "insight:decision", 0.25),
    (rb"(?i)\bresolved!?\b", "insight:decision", 0.25),
    (rb"(?i)\bsolved!?\b", "insight:decision", 0.25),
    (rb"(?i)\bworks!?\b", "insight:decision", 0.2),
    (rb"(?i)\bworking( now)?!?\b", "insight:decision", 0.2),
    (rb"(?i)\bcommitted\b", "insight:decision", 0.25),
    (rb"(?i)\bmerged\b", "insight:decision", 0.25),
    (rb"(?i)\bdeployed\b", "insight:decision", 0.25),
    (rb"(?i)\bpassed\b.{0,10}\b(tests?|checks?)\b", "insight:decision", 0.3),
    (rb"(?i)\bsucceeded\b", "insight:decision", 0.25),
    (rb"(?i)\bsuccessful(ly)?\b", "insight:decision", 0.2),
]

# -----------------------------------------------------------------------------
# COMPARISON PATTERNS - comparing options
# -----------------------------------------------------------------------------
COMPARISON_PATTERNS: list[tuple[bytes, str, float]] = [
    (rb"(?i)\bcompared to\b", "insight:tradeoff", 0.25),
    (rb"(?i)\bin comparison\b", "insight:tradeoff", 0.25),
    (rb"(?i)\bvs\.?\b", "insight:tradeoff", 0.2),
    (rb"(?i)\bversus\b", "insight:tradeoff", 0.2),
    (rb"(?i)\bdifference (is|between)\b", "insight:tradeoff", 0.25),
    (rb"(?i)\bunlike\b", "insight:tradeoff", 0.2),
    (rb"(?i)\bwhereas\b", "insight:tradeoff", 0.2),
    (rb"(?i)\bwhile\b.{0,20}\binstead\b", "insight:tradeoff", 0.25),
]

# -----------------------------------------------------------------------------
# TECHNICAL INSIGHT PATTERNS - specific technical findings
# -----------------------------------------------------------------------------
TECHNICAL_PATTERNS: list[tuple[bytes, str, float]] = [
    # Performance
    (
        rb"(?i)\bperformance (issue|problem|bottleneck)\b",
        "insight:decision",
        0.35,
    ),
    (rb"(?i)\bslow (because|due to)\b", "insight:decision", 0.3),
    (rb"(?i)\boptimized (by|using)\b", "insight:decision", 0.3),
    (rb"(?i)\bN\+1 (query|problem)\b", "insight:pitfall", 0.4),
    (rb"(?i)\bbig O\b", "insight:decision", 0.25),
    (rb"(?i)\btime complexity\b", "insight:decision", 0.3),
    (rb"(?i)\bspace complexity\b", "insight:decision", 0.3),
    # Security
    (rb"(?i)\bsecurity (issue|vulnerability|risk)\b", "insight:pitfall", 0.4),
    (
        rb"(?i)\binjection\b.{0,10}\b(attack|vulnerability)\b",
        "insight:pitfall",
        0.4,
    ),
    (rb"(?i)\bXSS\b", "insight:pitfall", 0.35),
    (rb"(?i)\bCSRF\b", "insight:pitfall", 0.35),
    (
        rb"(?i)\bauthentication\b.{0,15}\b(bug|issue|problem)\b",
        "insight:pitfall",
        0.35,
    ),
    (
        rb"(?i)\bauthorization\b.{0,15}\b(bug|issue|problem)\b",
        "insight:pitfall",
        0.35,
    ),
    # Concurrency
    (rb"(?i)\bthread[- ]?safe\b", "insight:decision", 0.3),
    (rb"(?i)\bdeadlock\b", "insight:pitfall", 0.4),
    (rb"(?i)\brace condition\b", "insight:pitfall", 0.4),
    (rb"(?i)\bconcurrency (issue|bug|problem)\b", "insight:pitfall", 0.35),
    # Types
    (rb"(?i)\btype (error|mismatch)\b", "insight:decision", 0.3),
    (rb"(?i)\btype[- ]?safe\b", "insight:decision", 0.25),
    (rb"(?i)\bnull (check|safety|pointer)\b", "insight:decision", 0.25),
]

# =============================================================================
# COMBINE ALL PATTERNS
# =============================================================================
ALL_PATTERN_DEFS: list[tuple[bytes, str, float]] = (
    DECISION_PATTERNS
    + BUG_PATTERNS
    + FIX_PATTERNS
    + CONSTRAINT_PATTERNS
    + TRADEOFF_PATTERNS
    + PITFALL_PATTERNS
    + ASSUMPTION_PATTERNS
    + DISCOVERY_PATTERNS
    + ARCHITECTURE_PATTERNS
    + RELATIONSHIP_PATTERNS
    + SUMMARY_PATTERNS
    + REALIZATION_PATTERNS
    + RECOMMENDATION_PATTERNS
    + VERIFICATION_PATTERNS
    + EXPLANATION_PATTERNS
    + OUTCOME_PATTERNS
    + COMPARISON_PATTERNS
    + TECHNICAL_PATTERNS
)

# =============================================================================
# PRE-COMPILED HYPERSCAN DATABASE (SINGLETON)
# =============================================================================
# Built once at module load time for maximum performance

_SCANNER: HyperscanSearch | None = None
_PATTERN_METADATA: list[tuple[str, float]] = []


def _build_scanner() -> tuple[HyperscanSearch, list[tuple[str, float]]]:
    """Build the hyperscan scanner and metadata list."""
    metadata = [
        (insight_type, confidence)
        for _, insight_type, confidence in ALL_PATTERN_DEFS
    ]
    pattern_bytes = [p for p, _, _ in ALL_PATTERN_DEFS]
    scanner = HyperscanSearch(pattern_bytes)
    return scanner, metadata


def get_scanner() -> tuple[HyperscanSearch, list[tuple[str, float]]]:
    """Get the pre-compiled scanner singleton."""
    global _SCANNER, _PATTERN_METADATA
    if _SCANNER is None:
        _SCANNER, _PATTERN_METADATA = _build_scanner()
        logger.info(
            "built hyperscan memory extractor with %d patterns",
            len(ALL_PATTERN_DEFS),
        )
    return _SCANNER, _PATTERN_METADATA


# =============================================================================
# PROCESS NARRATION FILTER (python re - start-of-text only)
# =============================================================================
# These patterns indicate PURE process narration with no insight content.
# "found it!" followed by actual content is fine - we strip the prefix.
PROCESS_NARRATION_PATTERNS = [
    re.compile(r"^let me\b", re.I),
    re.compile(r"^now (let me|I'll|I'm going to)\b", re.I),
    re.compile(r"^(ok|okay|alright)[\s,]+(so|let me|now)\b", re.I),
    re.compile(r"^checking\b", re.I),
    re.compile(r"^looking (at|for|into)\b", re.I),
    re.compile(r"^searching\b", re.I),
    re.compile(r"^reading\b", re.I),
    re.compile(r"^I('ll| will| am going to)\b", re.I),
    re.compile(r"^first,?\s+(let me|I'll)\b", re.I),
    re.compile(r"^lemme\b", re.I),
    re.compile(r"^on it\b", re.I),
    re.compile(r"^running\b", re.I),
    re.compile(r"^executing\b", re.I),
    re.compile(r"^starting\b", re.I),
]

# Prefixes that can be stripped if followed by actual content
STRIPPABLE_PREFIXES = [
    re.compile(r"^found it!?\s*", re.I),
    re.compile(r"^here's (the|what)[^.!?]*[.!?]\s*", re.I),
    re.compile(r"^(ok|okay|alright)[\s,]+so\s+", re.I),
    re.compile(r"^(ok|okay|alright)[,.]?\s+", re.I),
    re.compile(r"^(ah|aha|ahh)[,!]?\s+", re.I),
    re.compile(r"^(oh|ohh)[,!]?\s+", re.I),
    re.compile(r"^(yep|yup|yeah)[,.]?\s+", re.I),
    re.compile(r"^nice[,!]?\s+", re.I),
]

# =============================================================================
# TASK AND CONTEXT INFERENCE
# =============================================================================
TOOL_TO_TASK: dict[str, str] = {
    "Bash": "task:debug",
    "mcp__acp__Bash": "task:debug",
    "Write": "task:implement_feature",
    "Edit": "task:implement_feature",
    "mcp__acp__Write": "task:implement_feature",
    "mcp__acp__Edit": "task:implement_feature",
    "Read": "task:review",
    "Grep": "task:review",
    "Glob": "task:review",
    "mcp__acp__Read": "task:review",
    "mcp__ultrasync__search": "task:review",
}

CONTEXT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\.(tsx|jsx)$", re.I), "context:frontend"),
    (re.compile(r"components?/", re.I), "context:frontend"),
    (re.compile(r"(pages|views|screens)/", re.I), "context:frontend"),
    (re.compile(r"(server|api|routes|handlers)/", re.I), "context:backend"),
    (re.compile(r"(controllers?|services?)/", re.I), "context:backend"),
    (re.compile(r"(api|endpoints?|routes?)/", re.I), "context:api"),
    (re.compile(r"\.(graphql|proto)$", re.I), "context:api"),
    (re.compile(r"(auth|login|session|oauth|jwt)/", re.I), "context:auth"),
    (re.compile(r"(auth|login|session|password)", re.I), "context:auth"),
    (re.compile(r"(models?|schemas?|migrations?)/", re.I), "context:data"),
    (re.compile(r"(database|db|orm)/", re.I), "context:data"),
    (re.compile(r"test[s_]?/", re.I), "context:testing"),
    (re.compile(r"(test_|_test\.|\.test\.|\.spec\.)", re.I), "context:testing"),
    (re.compile(r"(docker|k8s|kubernetes|helm)/", re.I), "context:infra"),
    (re.compile(r"(Dockerfile|docker-compose)", re.I), "context:infra"),
    (re.compile(r"\.(tf|tfvars)$", re.I), "context:infra"),
    (
        re.compile(r"(\.github|\.gitlab|jenkins|circleci)/", re.I),
        "context:cicd",
    ),
    (re.compile(r"(workflows?|pipelines?)/", re.I), "context:cicd"),
]

# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class MemoryExtractionConfig:
    """Configuration for auto-memory extraction."""

    enabled: bool = True
    aggressiveness: AggressivenessLevel = "moderate"
    min_text_length: int = 50
    dedup_window: int = 100
    patterns_enabled: bool = True
    turn_summary_enabled: bool = True
    tool_context_enabled: bool = True
    exploration_higher_bar: bool = True

    @classmethod
    def from_env(cls) -> "MemoryExtractionConfig":
        """Load config from environment variables."""
        enabled_str = os.environ.get(ENV_MEMORY_EXTRACTION, "true")
        enabled = enabled_str.lower() in ("true", "1", "yes")

        aggressiveness = os.environ.get(ENV_MEMORY_AGGRESSIVENESS, "moderate")
        if aggressiveness not in ("conservative", "moderate", "aggressive"):
            aggressiveness = "moderate"

        return cls(
            enabled=enabled,
            aggressiveness=aggressiveness,  # type: ignore
        )

    @classmethod
    def conservative(cls) -> "MemoryExtractionConfig":
        """Conservative preset - only high-signal patterns."""
        return cls(
            aggressiveness="conservative",
            min_text_length=100,
            turn_summary_enabled=False,
        )

    @classmethod
    def moderate(cls) -> "MemoryExtractionConfig":
        """Moderate preset - include reasoning with file context."""
        return cls(aggressiveness="moderate", min_text_length=75)

    @classmethod
    def aggressive(cls) -> "MemoryExtractionConfig":
        """Aggressive preset - capture most substantive responses."""
        return cls(
            aggressiveness="aggressive",
            min_text_length=50,
            exploration_higher_bar=False,
        )


@dataclass
class PatternMatch:
    """A pattern match with location info."""

    pattern_id: int
    insight_type: str
    confidence: float
    start: int
    end: int


@dataclass
class ExtractionResult:
    """Result of memory extraction from assistant text."""

    should_create: bool
    text: str
    original_text: str
    task: str | None = None
    insights: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)
    symbol_keys: list[int] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.0
    skip_reason: str | None = None


class DeduplicationCache:
    """LRU cache for hash-based deduplication."""

    def __init__(self, max_size: int = 100):
        self._cache: OrderedDict[int, float] = OrderedDict()
        self._max_size = max_size

    def _normalize(self, text: str) -> str:
        normalized = text.lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = re.sub(r"[.,!?;:]+", "", normalized)
        return normalized

    def is_duplicate(self, text: str) -> bool:
        normalized = self._normalize(text)
        text_hash = hash(normalized)

        if text_hash in self._cache:
            self._cache.move_to_end(text_hash)
            return True

        self._cache[text_hash] = time.time()

        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

        return False

    def clear(self) -> None:
        self._cache.clear()


# =============================================================================
# MEMORY EXTRACTOR
# =============================================================================


class MemoryExtractor:
    """Extract meaningful snippets from assistant responses.

    Uses a pre-compiled Hyperscan database with hundreds of patterns
    for fast multi-pattern matching. Extracts focused snippets around
    matches rather than storing full messages.
    """

    def __init__(self, config: MemoryExtractionConfig | None = None):
        self.config = config or MemoryExtractionConfig.from_env()
        self._dedup_cache = DeduplicationCache(self.config.dedup_window)
        # Use singleton scanner
        self._scanner, self._pattern_metadata = get_scanner()

    def extract(
        self,
        text: str,
        tools_used: list[str] | None = None,
        files_accessed: list[str] | None = None,
    ) -> ExtractionResult:
        """Extract memory from assistant text."""
        tools_used = tools_used or []
        files_accessed = files_accessed or []

        if not self.config.enabled:
            return ExtractionResult(
                should_create=False,
                text="",
                original_text=text,
                skip_reason="extraction_disabled",
            )

        if len(text.strip()) < self.config.min_text_length:
            return ExtractionResult(
                should_create=False,
                text="",
                original_text=text,
                skip_reason="text_too_short",
            )

        # Determine if exploration-only
        has_action = any(t in ACTION_TOOLS for t in tools_used)
        is_exploration_only = not has_action and any(
            t in EXPLORATION_TOOLS for t in tools_used
        )

        # Find pattern matches
        matches = self._find_pattern_matches(text)

        if not matches:
            return ExtractionResult(
                should_create=False,
                text="",
                original_text=text,
                skip_reason="no_patterns_matched",
            )

        # Calculate confidence
        confidence = min(1.0, sum(m.confidence for m in matches))

        # Apply threshold
        if is_exploration_only and self.config.exploration_higher_bar:
            threshold = EXPLORATION_CONFIDENCE_THRESHOLD
        else:
            threshold = MIN_CONFIDENCE_THRESHOLD

        if confidence < threshold:
            return ExtractionResult(
                should_create=False,
                text="",
                original_text=text,
                skip_reason="below_confidence_threshold",
                confidence=confidence,
            )

        # Extract snippet and clean it
        snippet = self._extract_snippet(text, matches)
        snippet = self._strip_narration_prefix(snippet)

        # Filter process narration
        if self._is_process_narration(snippet):
            return ExtractionResult(
                should_create=False,
                text="",
                original_text=text,
                skip_reason="process_narration",
                confidence=confidence,
            )

        # Check deduplication
        if self._dedup_cache.is_duplicate(snippet):
            return ExtractionResult(
                should_create=False,
                text="",
                original_text=text,
                skip_reason="duplicate",
            )

        insights = list(set(m.insight_type for m in matches))
        task = self._infer_task(tools_used)
        context = self._infer_context(files_accessed)
        symbol_keys = [hash64_file_key(f) for f in files_accessed]

        return ExtractionResult(
            should_create=True,
            text=snippet,
            original_text=text,
            task=task,
            insights=insights,
            context=context,
            symbol_keys=symbol_keys,
            tags=["auto-extracted"],
            confidence=confidence,
        )

    def _find_pattern_matches(self, text: str) -> list[PatternMatch]:
        """Find pattern matches using hyperscan."""
        text_bytes = text.encode("utf-8")

        try:
            raw_matches = self._scanner.scan(text_bytes, match_limit=1000)
        except Exception as e:
            logger.warning("hyperscan scan failed: %s", e)
            return []

        matches: list[PatternMatch] = []
        for pattern_id, start, end in raw_matches:
            insight_type, confidence = self._pattern_metadata[pattern_id - 1]
            matches.append(
                PatternMatch(
                    pattern_id=pattern_id,
                    insight_type=insight_type,
                    confidence=confidence,
                    start=start,
                    end=end,
                )
            )

        return matches

    def _extract_snippet(self, text: str, matches: list[PatternMatch]) -> str:
        """Extract focused snippet around best match."""
        if not matches:
            return ""

        best_match = max(matches, key=lambda m: m.confidence)
        sentences = self._split_sentences(text)

        if not sentences:
            return text[:MAX_SNIPPET_LENGTH]

        # Find sentence containing match
        target_idx = 0
        char_pos = 0
        for i, sentence in enumerate(sentences):
            sentence_end = char_pos + len(sentence)
            if char_pos <= best_match.start < sentence_end:
                target_idx = i
                break
            char_pos = sentence_end

        # Get target + context
        start_idx = max(0, target_idx - 1)
        end_idx = min(len(sentences), target_idx + 2)

        snippet_sentences = sentences[start_idx:end_idx]
        snippet = " ".join(s.strip() for s in snippet_sentences if s.strip())

        if len(snippet) > MAX_SNIPPET_LENGTH:
            snippet = snippet[:MAX_SNIPPET_LENGTH].rsplit(" ", 1)[0] + "..."

        return snippet.strip()

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        pattern = r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$"
        sentences = re.split(pattern, text)
        return [s for s in sentences if s.strip()]

    def _strip_narration_prefix(self, text: str) -> str:
        """Strip common narration prefixes like 'found it!' from text."""
        result = text.strip()
        for pattern in STRIPPABLE_PREFIXES:
            result = pattern.sub("", result)
        return result.strip()

    def _is_process_narration(self, text: str) -> bool:
        """Check if text is pure process narration with no insight."""
        # First strip any acceptable prefixes
        text_stripped = self._strip_narration_prefix(text)

        # If nothing left after stripping, it was pure narration
        if len(text_stripped) < 30:
            return True

        # Check if what remains starts with process patterns
        for pattern in PROCESS_NARRATION_PATTERNS:
            if pattern.match(text_stripped):
                return True
        return False

    def _infer_task(self, tools_used: list[str]) -> str | None:
        """Infer task type from tools used."""
        if not self.config.tool_context_enabled:
            return None

        task_scores: dict[str, int] = {}
        for tool in tools_used:
            task = TOOL_TO_TASK.get(tool)
            if task:
                task_scores[task] = task_scores.get(task, 0) + 1

        if not task_scores:
            return "task:general"

        return max(task_scores, key=lambda k: task_scores[k])

    def _infer_context(self, files: list[str]) -> list[str]:
        """Infer context types from file paths."""
        if not self.config.tool_context_enabled:
            return []

        contexts: set[str] = set()
        for file_path in files:
            for pattern, context_type in CONTEXT_PATTERNS:
                if pattern.search(file_path):
                    contexts.add(context_type)

        return list(contexts)

    def reset_dedup_cache(self) -> None:
        """Reset the deduplication cache."""
        self._dedup_cache.clear()
        logger.debug("memory extractor dedup cache cleared")
