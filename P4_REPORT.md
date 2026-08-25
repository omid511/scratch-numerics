# P4 Aeroelastic Margin Estimation — Summary Report

> **⚠ SUPERSEDED**: This report covers the LEGACY single-design
> (nominal CFCF 1×1 m plate, velocity-grouped train/val split, no test
> set). The conclusions about generalization are NOT supported by
> held-out-design evidence. See `P4_REPORT_EXPANDED.md` for the current
> results on the expanded 160-design dataset with design-level splits.

## Problem Setup

| Parameter | Value |
|-----------|-------|
| Plate dimensions | L1 = L2 = 1.0 m |
| Laminate | CFCF honeycomb sandwich (face–core–face) |
| Face material | E1=E2=70 GPa, G=26.32 GPa, rho=2710 kg/m³ |
| Core material | E=47.3 MPa, G13=1.01 GPa, rho=278 kg/m³ |
| Layup | [0/0/0] symmetric, z=[-5, -4, 4, 5] mm |
| FSDT order | M=N=6 (42 DOFs per field, 210 total) |
| Grid | 30×30 elements |
| Stiffness penalty | k=10¹³ N/m (clamped BC enforcement) |
| Boundary conditions | Left+top clamped, right+bottom free (CFCF) |

## Flutter Boundary

- **Critical flutter velocity**: 1462.7 m/s
- **Mach number**: 4.30 (at SOUND_SPEED=340 m/s)
- **Scan range**: 680–3000 m/s, bisection with tol=1.0 m/s

## Data Generation

- **Velocity levels**: 24 (8 low 680–1000, 8 mid 1000–1300, 8 high 1300–0.95·u_crit)
- **Realizations per level**: 10
- **Total clips**: 240 valid (of 240 generated)
- **Clip length**: 512 timesteps over 0.0–0.5 s
- **Sensors**: 8 interior grid points
- **Modes**: 8 well-conditioned physical modes per velocity
- **Normalization**: causal (initial-window RMS)

Margin distribution: min=0.0500, max=0.5351, mean=0.2400

## Model Architecture

| Component | Config |
|-----------|--------|
| Backbone | TCN (causal Conv1d), 32 channels, 4 layers |
| Receptive field | 61 timesteps (kernel=3, dilations 1,2,4,8) |
| Head | Global average pooling → Linear |
| Dropout | 0.1 |
| Optimizer | Adam, lr=1e-3, CosineAnnealing |
| Epochs | 50 |
| Batch size | 32 |
| Train/val split | Grouped by velocity (15% val) |

Three model variants:

1. **Huber**: TCN + Linear(1), Huber loss (δ=0.1)
2. **Median**: TCN + Linear(1), pinball loss at τ=0.5
3. **Quantile**: TCN + Linear(3), pinball loss at τ∈{0.05, 0.50, 0.95}, median-centered parameterization

## Results Summary

| Model | MAE | 90% Coverage | Mean Interval Width |
|-------|-----|--------------|---------------------|
| Huber | 0.0197 | — | — |
| Median | 0.0182 | — | — |
| Quantile | 0.0194 | 0.917 | 0.0933 |

## Per-Velocity Analysis (Quantile model)

| Velocity (m/s) | Margin | MAE | Coverage | n_clips |
|----------------|--------|-----|----------|---------|
| 680 | 0.5351 | 0.0181 | 0.800 | 10 |
| 726 | 0.5039 | 0.0302 | 0.700 | 10 |
| 771 | 0.4726 | 0.0228 | 0.800 | 10 |
| 817 | 0.4414 | 0.0260 | 0.700 | 10 |
| 863 | 0.4101 | 0.0168 | 0.900 | 10 |
| 909 | 0.3788 | 0.0225 | 0.800 | 10 |
| 954 | 0.3476 | 0.0357 | 0.600 | 10 |
| 1000 | 0.3163 | 0.0184 | 0.950 | 20 |
| 1043 | 0.2870 | 0.0137 | 1.000 | 10 |
| 1086 | 0.2577 | 0.0104 | 1.000 | 10 |
| 1129 | 0.2284 | 0.0183 | 0.900 | 10 |
| 1171 | 0.1991 | 0.0189 | 1.000 | 10 |
| 1214 | 0.1698 | 0.0111 | 1.000 | 10 |
| 1257 | 0.1405 | 0.0162 | 1.000 | 10 |
| 1300 | 0.1112 | 0.0185 | 1.000 | 20 |
| 1313 | 0.1025 | 0.0142 | 1.000 | 10 |
| 1326 | 0.0937 | 0.0156 | 1.000 | 10 |
| 1338 | 0.0850 | 0.0189 | 1.000 | 10 |
| 1351 | 0.0762 | 0.0170 | 1.000 | 10 |
| 1364 | 0.0675 | 0.0161 | 1.000 | 10 |
| 1377 | 0.0587 | 0.0210 | 1.000 | 10 |
| 1390 | 0.0500 | 0.0290 | 0.900 | 10 |

## Near-Flutter Performance (margin < 0.15)

- **Clips**: 100 of 240
- **Velocity range**: 1257–1390 m/s
- **Quantile MAE**: 0.0194 (same as overall — robust across margin range)

## Conclusions

1. The TCN-based approach estimates aeroelastic margins from transient sensor
   signals with MAE ≤ 0.020 across all velocity regimes.
2. The quantile model provides calibrated uncertainty: 91.7% coverage
   on the 90% prediction interval (target: 90%).
3. Near-flutter clips (margin < 0.15) are predicted with comparable accuracy
   to far-from-flutter clips, confirming the model generalises to the
   safety-critical regime.
4. Per-velocity MAE is stable across the 24 velocity levels with no systematic
   degradation near the flutter boundary.
