# Decision Log

Record protocol and architecture decisions with their rationale, affected experiment cells, and rerun policy.

## DEC-001 — Research protocol v1

| Field | Record |
|---|---|
| Date | 2026-09-29 |
| Status | Approved |
| Owner role | Technical PM/BA |
| Reviewer roles | ML Lead; Evaluation Lead |
| Approval date | 2026-09-29 |
| Decision | Define protocol v1 with three concepts and six unique configurations per concept: `n1-r16`, `n3-r16`, `n5-r16`, `n10-r16`, `n5-r4`, and `n5-r32`. The shared `n5-r16` anchor yields 18 main runs; the contingency floor is 12 runs across two concepts using all six identical configurations. |
| Rationale | This preserves the specified data-size and rank comparisons while sharing their common anchor, holding other training and inference settings fixed, and defining the limits of conclusions before results are observed. |
| Impact | The approved research matrix, controls, evaluation, claim boundary, fallback, and change control are specified in [protocol v1](protocol.md). The README records the protocol freeze. |
| Affected experiment cells | All core cells across concepts: 3 × 6 = 18 main runs; contingency: 2 × 6 = 12 comparable runs. |
| Rerun policy | After approval/freeze, any change requires a new protocol version and decision-log entry, identification of affected run IDs/cells, and consistent reruns of all affected cells. Preserve failed-run evidence and completed artifacts. |
