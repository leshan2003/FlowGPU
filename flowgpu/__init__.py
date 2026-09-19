"""FlowGPU -- a simulator for heterogeneous many-core dataflow / brain-inspired
chip + GPU large-model inference."""

__version__ = "0.1.0"

from . import units
from .hardware import make_device, list_devices
from .workload import get_model, list_models
from .system import System, DevicePool, Parallelism, build_system, homogeneous
from .mapping import PlacementPolicy, named as placement
from .sim import ServingSimulator, WorkloadSpec, SLO, execute
from .power import calibrate_system

__all__ = ["units", "make_device", "list_devices", "get_model", "list_models",
           "System", "DevicePool", "Parallelism", "build_system",
           "homogeneous", "PlacementPolicy", "placement", "ServingSimulator",
           "WorkloadSpec", "SLO", "execute", "calibrate_system"]
