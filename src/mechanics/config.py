"""Configuration schema for the FSDT solver and parameter sweeps."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import yaml
import hashlib
import subprocess
from pathlib import Path


@dataclass
class MaterialConfig:
    name: str
    E1: float
    E2: float
    G23: float
    G13: float
    G12: float
    nu12: float
    rho: float


@dataclass
class LayerConfig:
    material: str  # name reference to a material definition
    thickness: float
    angle: float  # degrees, converted to radians internally


@dataclass
class EdgeConfig:
    type: Literal["free", "simply_supported", "clamped", "elastic"]
    k: list[float] = field(default_factory=list)  # [k1, k2, k3, k4, k5] for elastic


@dataclass
class BoundaryConfig:
    left: EdgeConfig = field(default_factory=lambda: EdgeConfig(type="clamped"))
    right: EdgeConfig = field(default_factory=lambda: EdgeConfig(type="clamped"))
    top: EdgeConfig = field(default_factory=lambda: EdgeConfig(type="clamped"))
    bottom: EdgeConfig = field(default_factory=lambda: EdgeConfig(type="clamped"))


@dataclass
class SweepParamConfig:
    range: list[float]
    log_scale: bool = False


@dataclass
class SolverConfig:
    M: int = 15
    N: int = 15
    basis: Literal["legendre", "trigonometric"] = "legendre"
    grid: list[int] = field(default_factory=lambda: [64, 64])


@dataclass
class PlateConfig:
    L1: float = 0.3
    L2: float = 0.3
    layers: list[LayerConfig] = field(default_factory=list)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)


@dataclass
class OutputConfig:
    format: Literal["zarr", "numpy", "h5py"] = "zarr"
    path: str = "./data/output"


@dataclass
class SensitivityConfig:
    enabled: bool = False
    method: Literal["sobol", "morris"] = "sobol"
    n_base_samples: int = 64
    # Parameters to fix (not sweep) — use when Sobol shows negligible effect
    fixed_params: dict[str, float] = field(default_factory=dict)


@dataclass
class SweepConfig:
    method: Literal["latin_hypercube", "grid", "random", "sensitivity_weighted"] = "latin_hypercube"
    n_samples: int = 10000
    parameters: dict[str, SweepParamConfig] = field(default_factory=dict)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    checkpoint_interval: int = 1000
    store_mode_shapes: bool = True
    mode_shape_grid: list[int] | None = None
    max_workers: int = 0  # 0 = auto (min(cpu_count, 3))


@dataclass
class ExperimentConfig:
    schema_version: int = 1
    solver_version: str = ""
    name: str = ""
    seed: int = 42
    solver: SolverConfig = field(default_factory=SolverConfig)
    plate: PlateConfig = field(default_factory=PlateConfig)
    materials: dict[str, MaterialConfig] = field(default_factory=dict)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    # Aerodynamic params (fixed or swept)
    airflow_velocity: float | None = None
    airflow_angle: float = 0.0
    air_density: float = 1.2
    sound_speed: float = 340.0

    @classmethod
    def from_yaml(cls, path: str) -> ExperimentConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: dict) -> ExperimentConfig:
        materials = {}
        for name, m in d.get("materials", {}).items():
            m = {k: float(v) if isinstance(v, str) else v for k, v in m.items()}
            materials[name] = MaterialConfig(name=name, **m)

        def parse_layer(layer_dict: dict) -> LayerConfig:
            return LayerConfig(**layer_dict)

        def parse_edge(edge_dict: dict) -> EdgeConfig:
            if isinstance(edge_dict, str):
                return EdgeConfig(type=edge_dict)
            return EdgeConfig(**edge_dict)

        boundary_raw = d.get("plate", {}).get("boundary", {})
        boundary = BoundaryConfig(
            left=parse_edge(boundary_raw.get("left", "clamped")),
            right=parse_edge(boundary_raw.get("right", "clamped")),
            top=parse_edge(boundary_raw.get("top", "clamped")),
            bottom=parse_edge(boundary_raw.get("bottom", "clamped")),
        )

        layers = [parse_layer(l) for l in d.get("plate", {}).get("layers", [])]

        sweep_params = {}
        for pname, pconf in d.get("sweep", {}).get("parameters", {}).items():
            sweep_params[pname] = SweepParamConfig(**pconf)

        sens_raw = d.get("sweep", {}).get("sensitivity", {})
        sensitivity = SensitivityConfig(
            enabled=sens_raw.get("enabled", False),
            method=sens_raw.get("method", "sobol"),
            n_base_samples=sens_raw.get("n_base_samples", 64),
            fixed_params=sens_raw.get("fixed_params", {}),
        )

        cfg = cls(
            schema_version=d.get("schema_version", 1),
            solver_version=d.get("solver_version", _get_git_hash()),
            name=d.get("name", ""),
            seed=d.get("seed", 42),
            solver=SolverConfig(**d.get("solver", {})),
            plate=PlateConfig(
                L1=d.get("plate", {}).get("L1", 0.3),
                L2=d.get("plate", {}).get("L2", 0.3),
                layers=layers,
                boundary=boundary,
            ),
            materials=materials,
            sweep=SweepConfig(
                method=d.get("sweep", {}).get("method", "latin_hypercube"),
                n_samples=d.get("sweep", {}).get("n_samples", 10000),
                parameters=sweep_params,
                sensitivity=sensitivity,
                checkpoint_interval=d.get("sweep", {}).get("checkpoint_interval", 1000),
                store_mode_shapes=d.get("sweep", {}).get("store_mode_shapes", False),
                mode_shape_grid=d.get("sweep", {}).get("mode_shape_grid"),
                max_workers=d.get("sweep", {}).get("max_workers", 0),
            ),
            output=OutputConfig(**d.get("output", {})),
            airflow_velocity=d.get("airflow_velocity"),
            airflow_angle=d.get("airflow_angle", 0.0),
            air_density=d.get("air_density", 1.2),
            sound_speed=d.get("sound_speed", 340.0),
        )
        return cfg


def _get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent.parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"
