"""Cradle: a framework for raising a newborn intellect.

Public surface (v1):
    LabelRegistry, RegistryFull   - dynamic class-name <-> id mapping
    Oracle, RuleBasedOracle, Label - utterance -> structured label
    Camera, CameraConfig          - webcam ring buffer
    TeacherRuntime, RuntimeConfig - SPELA trainer wired for online teaching
"""
from .oracle import Label, LLMOracle, Oracle, RuleBasedOracle
from .perception import Camera, CameraConfig
from .registry import LabelRegistry, RegistryFull
from .runtime import RuntimeConfig, TeacherRuntime

__all__ = [
    "Camera",
    "CameraConfig",
    "Label",
    "LabelRegistry",
    "LLMOracle",
    "Oracle",
    "RegistryFull",
    "RuleBasedOracle",
    "RuntimeConfig",
    "TeacherRuntime",
]

__version__ = "0.1.0"
