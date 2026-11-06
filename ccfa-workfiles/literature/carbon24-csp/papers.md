# Literature Search: Improving Carbon-24 CSP

Date: 2026-11-05

This source set supports mechanism choices; none of the papers establishes
90% Match@20 and normalized RMS 0.01 on Carbon-24.

| Paper | Source status | Mechanism relevant to ADITcsp |
| --- | --- | --- |
| [DiffCSP](https://arxiv.org/abs/2309.04475) | NeurIPS 2023 | Periodic E(3)-equivariant joint lattice/coordinate diffusion; fractional coordinates |
| [FlowMM](https://arxiv.org/abs/2406.04713) | arXiv, code linked | Riemannian flow matching for periodic, rotational and permutation structure |
| [CrystalFlow](https://arxiv.org/abs/2412.11693) | arXiv | Equivariant flow generation of lattice, coordinates and atom types |
| [MatterGen](https://arxiv.org/abs/2312.03687) | arXiv | Joint atom/lattice diffusion, symmetry/property conditioning, closer local energy minima |
| [Space Group Conditional Flow Matching](https://arxiv.org/abs/2509.23822) | arXiv preprint | Conditions flow and symmetric base noise on space group/Wyckoff positions |
| [Prototype Dependence Benchmark](https://arxiv.org/abs/2609.26502) | arXiv preprint | Template retrieval strongest single method; generative CSP substantially prototype-dependent |
| [TFMat](https://arxiv.org/abs/2609.01076) | arXiv preprint | Structured composition/symmetry/prototype conditions; abstract reports 92.04% MP-20 Match@20 |
| [AtomSteer](https://arxiv.org/abs/2610.38517) | arXiv preprint | Training-free bounded periodic/symmetry projections; target space-group recovery |
| [MatterSim](https://arxiv.org/abs/2405.04967) | arXiv | Universal atomistic MLIP for relaxation/energy evaluation |
| [CHGNet](https://arxiv.org/abs/2302.14231) | arXiv | Charge-informed universal MLIP for relaxation |
| [MACE-MP-0](https://arxiv.org/abs/2401.00096) | arXiv | Foundation MLIP for stable atomistic simulations |
| [Clari](https://arxiv.org/abs/2606.03199) | arXiv preprint | Flow-based CSP; energy-ranked selection from a larger candidate pool |
| [DiverseFlow](https://arxiv.org/abs/2504.07894) | arXiv preprint | Training-free DPP diversity control for fixed sampling budgets |

## Cautions

- TFMat's 92.04% is an MP-20 abstract result, not a Carbon-24 result.
- Prototype-benchmark and Clari are recent preprints and use different benchmarks.
- Space-group conditioning with ground-truth test symmetry is an oracle experiment;
  a deployable system needs a predicted condition or an additional input.
- MLIP relaxation improves physical consistency, not necessarily movement toward
  the experimental target.

