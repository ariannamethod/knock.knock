"""
CLOUD v3.1 — Pre-Semantic Sonar (Fully Async)

"Something fires BEFORE meaning arrives"

Architecture:
    - Resonance Layer (weightless geometry)
    - Chamber MLPs (4 × 8.5K params + cross-fire)
    - Meta-Observer (15K params)
    - User Cloud (temporal fingerprint)
    - Anomaly Detection (0 params, heuristic)
    - Feedback Loop (0 params, closed-loop learning)

Total: ~50K params

Async Pattern:
    All components have async wrappers with field lock discipline.
    Based on HAZE's async architecture for coherence.

Usage:
    from cloud import Cloud, AsyncCloud

    # Sync usage
    cloud = Cloud.random_init()
    response = cloud.ping_sync("I'm feeling anxious")

    # Async usage (recommended)
    async with AsyncCloud.create() as cloud:
        response = await cloud.ping("I'm feeling anxious")
        print(f"Primary: {response.primary}, Secondary: {response.secondary}")
"""

from .cloud import Cloud, CloudResponse, AsyncCloud
from .chambers import CrossFireSystem, ChamberMLP, AsyncCrossFireSystem
from .observer import MetaObserver, AsyncMetaObserver
from .resonance import SimpleResonanceLayer, ResonanceLayer
from .user_cloud import UserCloud, EmotionEvent, AsyncUserCloud
from .anomaly import detect_anomalies, AnomalyReport
from .feedback import measure_coherence, update_coupling
from .anchors import (
    EMOTION_ANCHORS,
    CHAMBER_NAMES,
    COUPLING_MATRIX,
    get_all_anchors,
    get_chamber_ranges,
)

__version__ = "3.1.0"

__all__ = [
    # Main classes
    "Cloud",
    "CloudResponse",
    "AsyncCloud",
    
    # Components (sync)
    "CrossFireSystem",
    "ChamberMLP",
    "MetaObserver",
    "SimpleResonanceLayer",
    "ResonanceLayer",
    "UserCloud",
    "EmotionEvent",
    
    # Components (async) — HAZE-style discipline
    "AsyncCrossFireSystem",
    "AsyncMetaObserver",
    "AsyncUserCloud",
    
    # Anomaly detection
    "detect_anomalies",
    "AnomalyReport",
    
    # Feedback loop
    "measure_coherence",
    "update_coupling",
    
    # Anchors
    "EMOTION_ANCHORS",
    "CHAMBER_NAMES",
    "COUPLING_MATRIX",
    "get_all_anchors",
    "get_chamber_ranges",
]
