"""
ctypes FFI bindings to the aerofusion-ffi Rust cdylib.
"""

import ctypes
import ctypes.util
from ctypes import c_float, c_uint64, c_int32, c_char_p, Structure, POINTER
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple


class AerofusionConfigC(Structure):
    _fields_ = [
        ("gyro_noise", c_float),
        ("gyro_bias_noise", c_float),
        ("accel_noise", c_float),
        ("mag_noise", c_float),
        ("init_att_cov", c_float),
        ("init_bias_cov", c_float),
        ("gravity_magnitude", c_float),
    ]


class AerofusionTelemetryC(Structure):
    _fields_ = [
        ("qw", c_float),
        ("qx", c_float),
        ("qy", c_float),
        ("qz", c_float),
        ("roll_rad", c_float),
        ("pitch_rad", c_float),
        ("yaw_rad", c_float),
        ("roll_deg", c_float),
        ("pitch_deg", c_float),
        ("yaw_deg", c_float),
        ("bias_x", c_float),
        ("bias_y", c_float),
        ("bias_z", c_float),
        ("att_std_x", c_float),
        ("att_std_y", c_float),
        ("att_std_z", c_float),
        ("bias_std_x", c_float),
        ("bias_std_y", c_float),
        ("bias_std_z", c_float),
        ("step_count", c_uint64),
    ]


@dataclass
class Telemetry:
    qw: float
    qx: float
    qy: float
    qz: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    bias_x: float
    bias_y: float
    bias_z: float
    roll_std_deg: float
    pitch_std_deg: float
    yaw_std_deg: float
    step_count: int


def _find_library() -> Path:
    base_dir = Path(__file__).resolve().parent.parent.parent
    candidates = [
        base_dir / "target" / "release" / "aerofusion_ffi.dll",
        base_dir / "target" / "debug" / "aerofusion_ffi.dll",
        base_dir / "cpp_demo" / "build" / "aerofusion_ffi.dll",
        base_dir / "target" / "release" / "libaerofusion_ffi.so",
        base_dir / "target" / "debug" / "libaerofusion_ffi.so",
        base_dir / "target" / "release" / "libaerofusion_ffi.dylib",
        base_dir / "target" / "debug" / "libaerofusion_ffi.dylib",
    ]
    for p in candidates:
        if p.exists():
            return p

    # Fallback to system loader
    sys_lib = ctypes.util.find_library("aerofusion_ffi")
    if sys_lib:
        return Path(sys_lib)

    raise FileNotFoundError(
        f"Could not locate aerofusion_ffi binary. Searched in: {[str(c) for c in candidates]}"
    )


class AerofusionEngine:
    """Python high-level wrapper around the Rust Aerofusion EKF core."""

    def __init__(self, config: Optional[AerofusionConfigC] = None):
        lib_path = _find_library()
        self._lib = ctypes.CDLL(str(lib_path))
        self._setup_prototypes()

        if config is not None:
            self._handle = self._lib.aerofusion_create_with_config(ctypes.byref(config))
        else:
            self._handle = self._lib.aerofusion_create()

        if not self._handle:
            raise RuntimeError("Failed to create AerofusionEngine instance in Rust core")

    def _setup_prototypes(self):
        self._lib.aerofusion_version.restype = c_char_p
        self._lib.aerofusion_version.argtypes = []

        self._lib.aerofusion_create.restype = ctypes.c_void_p
        self._lib.aerofusion_create.argtypes = []

        self._lib.aerofusion_create_with_config.restype = ctypes.c_void_p
        self._lib.aerofusion_create_with_config.argtypes = [POINTER(AerofusionConfigC)]

        self._lib.aerofusion_destroy.restype = None
        self._lib.aerofusion_destroy.argtypes = [ctypes.c_void_p]

        self._lib.aerofusion_reset.restype = c_int32
        self._lib.aerofusion_reset.argtypes = [ctypes.c_void_p]

        self._lib.aerofusion_step.restype = c_int32
        self._lib.aerofusion_step.argtypes = [
            ctypes.c_void_p,
            c_float, c_float, c_float,
            c_float, c_float, c_float,
            c_float,
        ]

        self._lib.aerofusion_step_with_mag.restype = c_int32
        self._lib.aerofusion_step_with_mag.argtypes = [
            ctypes.c_void_p,
            c_float, c_float, c_float,
            c_float, c_float, c_float,
            c_float, c_float, c_float,
            c_float,
        ]

        self._lib.aerofusion_get_telemetry.restype = c_int32
        self._lib.aerofusion_get_telemetry.argtypes = [ctypes.c_void_p, POINTER(AerofusionTelemetryC)]

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            self._lib.aerofusion_destroy(self._handle)
            self._handle = None

    def reset(self):
        res = self._lib.aerofusion_reset(self._handle)
        if res != 0:
            raise RuntimeError(f"aerofusion_reset failed with code {res}")

    def step(self, gyro: Tuple[float, float, float], accel: Tuple[float, float, float], dt: float):
        res = self._lib.aerofusion_step(
            self._handle,
            c_float(gyro[0]), c_float(gyro[1]), c_float(gyro[2]),
            c_float(accel[0]), c_float(accel[1]), c_float(accel[2]),
            c_float(dt),
        )
        if res != 0:
            raise RuntimeError(f"aerofusion_step failed with code {res}")

    def step_with_mag(
        self,
        gyro: Tuple[float, float, float],
        accel: Tuple[float, float, float],
        mag: Tuple[float, float, float],
        dt: float,
    ):
        res = self._lib.aerofusion_step_with_mag(
            self._handle,
            c_float(gyro[0]), c_float(gyro[1]), c_float(gyro[2]),
            c_float(accel[0]), c_float(accel[1]), c_float(accel[2]),
            c_float(mag[0]), c_float(mag[1]), c_float(mag[2]),
            c_float(dt),
        )
        if res != 0:
            raise RuntimeError(f"aerofusion_step_with_mag failed with code {res}")

    def get_telemetry(self) -> Telemetry:
        raw = AerofusionTelemetryC()
        res = self._lib.aerofusion_get_telemetry(self._handle, ctypes.byref(raw))
        if res != 0:
            raise RuntimeError(f"aerofusion_get_telemetry failed with code {res}")

        rad_to_deg = 57.29577951308232
        return Telemetry(
            qw=float(raw.qw),
            qx=float(raw.qx),
            qy=float(raw.qy),
            qz=float(raw.qz),
            roll_deg=float(raw.roll_deg),
            pitch_deg=float(raw.pitch_deg),
            yaw_deg=float(raw.yaw_deg),
            bias_x=float(raw.bias_x),
            bias_y=float(raw.bias_y),
            bias_z=float(raw.bias_z),
            roll_std_deg=float(raw.att_std_x * rad_to_deg),
            pitch_std_deg=float(raw.att_std_y * rad_to_deg),
            yaw_std_deg=float(raw.att_std_z * rad_to_deg),
            step_count=int(raw.step_count),
        )

    @classmethod
    def version(cls) -> str:
        lib_path = _find_library()
        lib = ctypes.CDLL(str(lib_path))
        lib.aerofusion_version.restype = c_char_p
        return lib.aerofusion_version().decode("utf-8")
