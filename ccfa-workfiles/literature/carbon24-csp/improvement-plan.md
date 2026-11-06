# Carbon-24: Evidence-Grounded Improvement Plan

Date: 2026-11-05

## Current Evidence

- K=8 standard DiT: 1579/2030 matched (77.78%), 451 misses; mean 3.44 matched
  candidates per target; no target has all 20 candidates matching.
- K=16 CoDeMDT decoder=6: 1573/2030 (77.49%), 457 misses; mean 3.40 matched
  candidates; also no all-20 target.
- Common misses: 363. K=8-only hits: 94; K=16-only hits: 88.
- Candidate RMS: median about 0.42 and p75 about 0.46, with StructureMatcher
  `stol=0.5`. Many candidates are near the acceptance boundary.
- The common misses concentrate in C6/C8 and specific space groups.

## Training-Prototype Coverage

Local Carbon-24 checks found:

- Train has 226 space-group-227 structures; test has 71.
- All 71 SG227 test targets have a same-atom-count training structure that matches
  under `StructureMatcher(stol=0.5, ltol=0.3, angle_tol=10)`.
- 362 of the 363 common missed targets have at least one training record with the
  same atom count and space group. The one exception is in a space group absent
  from train.
- Test space groups 7, 17, 36, 122 and 148 are absent from train, but account for
  only six test structures.

This rules out “the training set has no corresponding prototype” as the
explanation for almost all of the current plateau. The remaining problem is in
the conditional generator/prior/sampling or in the precision of the decoder,
not a complete absence of the relevant Carbon-24 families.

## Solutions

1. **Space-group/Wyckoff condition.** The current code zeros `spacegroup` in both
   training and sampling. Add a condition vector containing space group and
   possibly Wyckoff/lattice information. First run an oracle-symmetry diagnostic;
   for a deployable result, predict the condition from composition or an extra
   input. References: Space Group Conditional Flow Matching, MatterGen.
2. **Train-only prototype retrieval/conditioning.** Build a prototype index from
   train composition, space group, cell, RDF and VAE descriptor. Use the retrieved
   prototype as a condition/initialization. The local SG227 result shows that the
   train split contains matching prototypes for all 71 test targets, so this is a
   high-value route. References: prototype-dependence benchmark, TFMat.
3. **MLIP relaxation and candidate ranking.** Generate 50-150 candidates,
   relax/rank with MatterSim, CHGNet or MACE-MP-0, and evaluate after
   deduplication. This targets RMS/validity. References: Clari, MatterGen,
   MatterSim, CHGNet, MACE-MP-0.
4. **Test-time periodic/symmetry projection.** Use bounded projections for
   predicted/known symmetry and periodic geometry. Reference: AtomSteer.
5. **Periodic equivariant backbone.** Use periodic E(3)-equivariant denoising and
   fractional-coordinate flow for long-term precision. References: DiffCSP,
   FlowMM, CrystalFlow.
6. **DiverseFlow as a secondary sampling add-on.** Current MM-FM candidates are
   already diverse (no all-20 matches), so repulsion cannot address the 363
   common misses by itself. Reference: DiverseFlow.

## Priority

1. Space-group oracle diagnostic, then a predicted symmetry condition.
2. Train-only prototype retrieval/conditioning.
3. MLIP relaxation/ranking for RMS.
4. AtomSteer-style projection and equivariant backbone.
5. DiverseFlow only after coverage is fixed.

The 0.01 RMS target is stricter than the current VAE reconstruction mean RMS of
about 0.022. It requires a separate precision/refinement stage; no cited paper
guarantees 90% Match@20 and 0.01 RMS on Carbon-24.

