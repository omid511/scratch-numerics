"""Tests for configuration schema and YAML parsing."""
import os
import tempfile
import numpy as np
from mechanics.config import (
    ExperimentConfig, MaterialConfig, LayerConfig, EdgeConfig,
    BoundaryConfig, SolverConfig, PlateConfig, SweepConfig,
    SweepParamConfig, OutputConfig,
)


def test_minimal_config():
    cfg = ExperimentConfig(name="test")
    assert cfg.schema_version == 1
    assert cfg.name == "test"
    assert cfg.seed == 42


def test_solver_config_defaults():
    cfg = SolverConfig()
    assert cfg.M == 15
    assert cfg.N == 15
    assert cfg.basis == "legendre"
    assert cfg.grid == [64, 64]


def test_plate_config_defaults():
    cfg = PlateConfig()
    assert cfg.L1 == 0.3
    assert cfg.L2 == 0.3
    assert len(cfg.layers) == 0


def test_edge_config_free():
    cfg = EdgeConfig(type="free")
    assert cfg.type == "free"
    assert cfg.k == []


def test_edge_config_elastic():
    cfg = EdgeConfig(type="elastic", k=[1e6, 2e6, 3e6, 4e6, 5e6])
    assert cfg.type == "elastic"
    assert len(cfg.k) == 5


def test_boundary_config_defaults():
    cfg = BoundaryConfig()
    assert cfg.left.type == "clamped"
    assert cfg.right.type == "clamped"
    assert cfg.top.type == "clamped"
    assert cfg.bottom.type == "clamped"


def test_sweep_config_defaults():
    cfg = SweepConfig()
    assert cfg.method == "latin_hypercube"
    assert cfg.n_samples == 10000


def test_yaml_roundtrip():
    """Write a YAML, read it back, verify all fields."""
    yaml_content = """
schema_version: 1
name: test_experiment
seed: 123
solver:
  M: 10
  N: 10
  basis: legendre
  grid: [32, 32]
plate:
  L1: 0.1
  L2: 0.2
  layers:
    - material: aluminium
      thickness: 0.001
      angle: 0
  boundary:
    left: simply_supported
    right:
      type: clamped
    top: free
    bottom:
      type: elastic
      k: [1000, 2000, 3000, 4000, 5000]
materials:
  aluminium:
    E1: 70e9
    E2: 70e9
    G23: 26e9
    G13: 26e9
    G12: 26e9
    nu12: 0.33
    rho: 2710
sweep:
  method: grid
  n_samples: 500
  parameters:
    velocity:
      range: [100, 500]
      log_scale: false
    thickness:
      range: [0.0005, 0.002]
      log_scale: true
output:
  format: numpy
  path: ./data/test
airflow_velocity: 400.0
air_density: 1.225
sound_speed: 343.0
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        path = f.name

    try:
        cfg = ExperimentConfig.from_yaml(path)
        assert cfg.name == "test_experiment"
        assert cfg.seed == 123
        assert cfg.solver.M == 10
        assert cfg.solver.grid == [32, 32]
        assert cfg.plate.L1 == 0.1
        assert cfg.plate.L2 == 0.2
        assert len(cfg.plate.layers) == 1
        assert cfg.plate.layers[0].material == "aluminium"
        assert cfg.plate.layers[0].thickness == 0.001
        assert cfg.plate.boundary.left.type == "simply_supported"
        assert cfg.plate.boundary.right.type == "clamped"
        assert cfg.plate.boundary.top.type == "free"
        assert cfg.plate.boundary.bottom.type == "elastic"
        assert cfg.plate.boundary.bottom.k == [1000, 2000, 3000, 4000, 5000]
        assert "aluminium" in cfg.materials
        assert cfg.materials["aluminium"].E1 == 70e9
        assert cfg.sweep.method == "grid"
        assert cfg.sweep.n_samples == 500
        assert "velocity" in cfg.sweep.parameters
        assert cfg.sweep.parameters["velocity"].range == [100, 500]
        assert cfg.sweep.parameters["thickness"].log_scale is True
        assert cfg.output.format == "numpy"
        assert cfg.airflow_velocity == 400.0
        assert cfg.air_density == 1.225
        assert cfg.sound_speed == 343.0
    finally:
        os.unlink(path)


def test_yaml_minimal():
    yaml_content = """
name: minimal
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        path = f.name
    try:
        cfg = ExperimentConfig.from_yaml(path)
        assert cfg.name == "minimal"
        assert cfg.solver.M == 15  # default
    finally:
        os.unlink(path)


def test_layer_config():
    lc = LayerConfig(material="aluminium", thickness=0.001, angle=45.0)
    assert lc.material == "aluminium"
    assert lc.thickness == 0.001
    assert lc.angle == 45.0


def test_material_config():
    mc = MaterialConfig(
        name="test_mat", E1=100e9, E2=10e9,
        G23=5e9, G13=5e9, G12=5e9,
        nu12=0.3, rho=1500,
    )
    assert mc.E1 == 100e9
    assert mc.rho == 1500
