"""
RHEN Engine Facade.

Clean facade exporting core Rhen decision engines, gates, verdicts, and primitives.
"""

from .rhen import (
    # Core Engines & Gates
    RhenEngine,
    AdvancedRhenEngine,
    LatentSearchEngine,
    GeometricAbstentionGate,
    # Abstention Constants & Verdict
    ANSWER,
    ABSTAIN,
    NEED_CLARIFICATION,
    AbstentionVerdict,
    RhenAbstentionError,
    # High-level ergonomic APIs
    decide,
    evaluate,
    get_default_engine,
    reset_default_engine,
    # Decision Primitives & Results
    Noul,
    NoulResult,
    Choice,
    ChoiceResult,
    Score,
    ScoreResult,
    MultiChoice,
    MultiChoiceResult,
    Range,
    RangeResult,
    Extract,
    ExtractResult,
    Cascade,
    CascadeResult,
    # Schema & Latent structures
    SchemaMutation,
    SCHEMA_MUTATION_TYPES,
    LatentSearchResult,
    LatentRolloutResult,
)

__all__ = [
    "RhenEngine",
    "AdvancedRhenEngine",
    "LatentSearchEngine",
    "GeometricAbstentionGate",
    "ANSWER",
    "ABSTAIN",
    "NEED_CLARIFICATION",
    "AbstentionVerdict",
    "RhenAbstentionError",
    "decide",
    "evaluate",
    "get_default_engine",
    "reset_default_engine",
    "Noul",
    "NoulResult",
    "Choice",
    "ChoiceResult",
    "Score",
    "ScoreResult",
    "MultiChoice",
    "MultiChoiceResult",
    "Range",
    "RangeResult",
    "Extract",
    "ExtractResult",
    "Cascade",
    "CascadeResult",
    "SchemaMutation",
    "SCHEMA_MUTATION_TYPES",
    "LatentSearchResult",
    "LatentRolloutResult",
]
