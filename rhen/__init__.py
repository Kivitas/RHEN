"""
RHEN: Non-Autoregressive JEV-Type Decision Engine for Agentic AI.

RHEN replaces brittle, slow autoregressive text generation and error-prone JSON
parsing with a single forward-pass probabilistic decision evaluation over
structured state.

Core Primitives:
- Noul: Binary semantic truth / proposition evaluation
- Choice: Categorical 1-of-K selection with type-safe confidence
- Score: Ordered continuous spectrum (ordinal levels)
- MultiChoice: Independent subset selection
- Range: Bounded continuous interval regression (fitted Beta)
- Extract: Non-autoregressive span pointer extraction
- Cascade: Staged & conditional decision sequences
"""

__version__ = "1.0.0"

# 1. Primitives & Results & High-Level APIs
from .rhen import (
    Noul, NoulResult,
    Choice, ChoiceResult,
    Score, ScoreResult,
    MultiChoice, MultiChoiceResult,
    Range, RangeResult,
    Extract, ExtractResult,
    Cascade, CascadeResult,
    AbstentionVerdict,
    LatentSearchResult,
    LatentRolloutResult,
    SchemaMutation,
    SCHEMA_MUTATION_TYPES,
    RhenEngine,
    AdvancedRhenEngine,
    LatentSearchEngine,
    GeometricAbstentionGate,
    ANSWER,
    ABSTAIN,
    NEED_CLARIFICATION,
    RhenAbstentionError,
    decide,
    evaluate,
    get_default_engine,
    reset_default_engine,
)

# 2. Sequence Packing & Masks
from .rhen_packer import (
    PackedRhenSequence,
    RhenPacker,
    get_rhen_packer,
    sanitize_delimiter_text,
    compute_structural_depth_ids,
)

RhenSequencePacker = RhenPacker

# 3. Model & Architecture
from .config import RhenConfig, EOSConfig, RhenEngineConfig
from .model import (
    RhenModel,
    EOSModel,
    EOSLoopedLLM,
    MultiHeadRhenPointerHead,
    RhenLatentDecisionEngine,
    ContinuousLatentAdapter,
    LatentTransitionOperator,
    ProspectiveValueHead,
    ModernHopfieldEpisodicBank,
    SchemaMutationHead,
)
from .adapter import HuggingFaceEncoderAdapter, RhenEncoderAdapter, MockEncoder

# 4. Context Building
from .context_builder import ContextBuilder

# 5. Agent Tools & Cryptographic Action Cards
from .agent_tools import (
    ActionCard,
    ActionStagingEngine,
    MerkleActionLedger,
    BFTConsensusEngine,
    canonical_payload_json,
)

# 6. Conformal Calibration & Meta-Cognition
from .meta_cognitive_monitor import (
    ConformalPredictionEngine,
    AdaptiveConformalEngine,
    EnergyOODDetector,
    MetaCognitiveMonitor,
    RhenCalibrator,
)

# 7. Verification & Out-of-Band Audit
from .rhen_verify import (
    OutOfBandAuditLedger,
    SocketAuditSink,
    serve_audit_socket,
    verify_sealed_records,
    GeometrySelectiveClassifier,
    CheapVerifier,
)

# 8. Pillars & Orchestration
from .rhen_pillars import RhenPillarConfig, PillarStack
from .unified_orchestrator import UnifiedAIOrchestrator, RhenOrchestrator

# 9. Cognition Extensions
from .rhen_cognition import (
    RLCDOptimizer,
    DurationThresholdingHead,
    HierarchicalChoiceRouter,
    OODRegressionGuard,
    EvidenceVerificationLattice,
    MultilingualDecisionRouter,
)

# 10. Chunking & Long Context
from .rhen_chunker import (
    SemanticBoundaryChunker,
    SalienceGatedChunkRouter,
    ChunkedSpanPointer,
    ChunkedRhenState,
    evidential_logsumexp_fuse,
)

# 11. Credit Assignment
from .rhen_credit import (
    TrajectoryGraph,
    SegmentAdvantage,
    SchemaContrastiveAdvantage,
    CascadeSearch,
    search_vs_rl,
)

# 12. Tokenizer
from .tokenizer import (
    EOSTokenizer,
    RhenTokenizer,
    get_tokenizer,
    get_rhen_tokenizer,
    RHEN_SPECIAL_TOKENS,
)

__all__ = [
    "Noul", "NoulResult",
    "Choice", "ChoiceResult",
    "Score", "ScoreResult",
    "MultiChoice", "MultiChoiceResult",
    "Range", "RangeResult",
    "Extract", "ExtractResult",
    "Cascade", "CascadeResult",
    "AbstentionVerdict",
    "LatentSearchResult",
    "LatentRolloutResult",
    "SchemaMutation",
    "SCHEMA_MUTATION_TYPES",
    "RhenEngine",
    "AdvancedRhenEngine",
    "LatentSearchEngine",
    "GeometricAbstentionGate",
    "ANSWER", "ABSTAIN", "NEED_CLARIFICATION",
    "RhenAbstentionError",
    "decide", "evaluate", "get_default_engine", "reset_default_engine",
    "PackedRhenSequence",
    "RhenPacker",
    "RhenSequencePacker",
    "get_rhen_packer",
    "sanitize_delimiter_text",
    "compute_structural_depth_ids",
    "RhenConfig", "EOSConfig", "RhenEngineConfig",
    "RhenModel", "EOSModel", "EOSLoopedLLM",
    "MultiHeadRhenPointerHead",
    "RhenLatentDecisionEngine",
    "ContinuousLatentAdapter",
    "LatentTransitionOperator",
    "ProspectiveValueHead",
    "ModernHopfieldEpisodicBank",
    "SchemaMutationHead",
    "HuggingFaceEncoderAdapter",
    "RhenEncoderAdapter",
    "MockEncoder",
    "ContextBuilder",
    "ActionCard",
    "ActionStagingEngine",
    "MerkleActionLedger",
    "BFTConsensusEngine",
    "canonical_payload_json",
    "ConformalPredictionEngine",
    "AdaptiveConformalEngine",
    "EnergyOODDetector",
    "MetaCognitiveMonitor",
    "RhenCalibrator",
    "OutOfBandAuditLedger",
    "SocketAuditSink",
    "serve_audit_socket",
    "verify_sealed_records",
    "GeometrySelectiveClassifier",
    "CheapVerifier",
    "RhenPillarConfig",
    "PillarStack",
    "UnifiedAIOrchestrator",
    "RhenOrchestrator",
    "RLCDOptimizer",
    "DurationThresholdingHead",
    "HierarchicalChoiceRouter",
    "OODRegressionGuard",
    "EvidenceVerificationLattice",
    "MultilingualDecisionRouter",
    "SemanticBoundaryChunker",
    "SalienceGatedChunkRouter",
    "ChunkedSpanPointer",
    "ChunkedRhenState",
    "evidential_logsumexp_fuse",
    "TrajectoryGraph",
    "SegmentAdvantage",
    "SchemaContrastiveAdvantage",
    "CascadeSearch",
    "search_vs_rl",
    "EOSTokenizer",
    "RhenTokenizer",
    "get_tokenizer",
    "get_rhen_tokenizer",
    "RHEN_SPECIAL_TOKENS",
]
