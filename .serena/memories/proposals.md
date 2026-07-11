# Research Proposals Overview

## Program Theme
Physics-informed probabilistic machine learning for aeroelastic analysis, uncertainty quantification, structural health monitoring, and robust design of honeycomb sandwich structures.

## Four Proposals

### Proposal 1: Multi-Fidelity Correction Field via Latent-Space Gaussian Process
- **Goal:** Learn physics-consistent correction field for FSDT's ~4% error
- **Architecture:** Encoder/decoder + latent-space GP
- **Data:** 10,000-20,000 LF (FSDT) + 50-100 HF (Abaqus) samples
- **Key:** Boundary conditions enforced structurally, not learned

### Proposal 2: Probabilistic Inverse Damage Identification
- **Goal:** Localize and quantify damage from sparse vibration measurements
- **Architecture:** Latent-space posterior inference with learned decoder
- **Data:** 20,000-50,000 FSDT samples + held-out Abaqus subset
- **Key:** Variable-damage-count problem solved via learned latent representation

### Proposal 3: Robust Design Under Stochastic Boundary Conditions
- **Goal:** Identify robust design meeting reliability targets
- **Architecture:** Gradient-enhanced GP with mode-aware active learning
- **Data:** 200 initial FSDT runs + active learning iterations
- **Key:** Mode veering as transition detection opportunity

### Proposal 4: Continuous Stability-Margin Estimation
- **Goal:** Online structural health monitoring with continuous margin estimation
- **Architecture:** TCN backbone with quantile regression head
- **Data:** Transient response clips from FSDT solver
- **Key:** Domain randomization for sim-to-real transfer

## Cross-Cutting Notes
- Proposals 1 & 2 share encoder/decoder infrastructure
- Proposals 1 & 3 share active-learning machinery
- Proposals 3 & 4 share parameter sweep infrastructure
- Proposal 1's Abaqus runs serve as Proposal 2's held-out test set

## Execution Order
1. Proposal 1 first (Abaqus turnaround is critical path)
2. Proposal 3 in parallel (fast solver, no blocking dependency)
3. Proposal 2 follows (development on FSDT data)
4. Proposal 4 last (reuses Proposal 3's sampling infrastructure)