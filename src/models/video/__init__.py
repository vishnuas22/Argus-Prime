"""Video detection models."""

from .forensic_flow import (
    ForensicFlow,
    ForensicFlowResult,
    create_forensic_flow,
    register_forensic_flow,
)

from .distil_dire import (
    DistilDIRE,
    DIREResult,
    create_distil_dire,
    register_distil_dire,
)

from .kan_detector import (
    KANDetector,
    KANVideoDetector,
    KANResult,
    create_kan_detector,
    register_kan,
)

__all__ = [
    "ForensicFlow",
    "ForensicFlowResult",
    "create_forensic_flow",
    "register_forensic_flow",
    "DistilDIRE",
    "DIREResult",
    "create_distil_dire",
    "register_distil_dire",
    "KANDetector",
    "KANVideoDetector",
    "KANResult",
    "create_kan_detector",
    "register_kan",
]
