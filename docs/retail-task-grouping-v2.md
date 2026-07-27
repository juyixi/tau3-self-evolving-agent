# Retail Task Grouping v2

## Decision

All Tau3 Retail train episodes use the single attribution group `retail-v2`.
Maintenance remains a separate lifecycle decision group, `retail-v2:maintenance`.
Future domains use a separate domain-level group, for example `airline-v2`.

## Rationale

OPD-Evolver requires comparisons within a task group for outcome-calibrated
attribution, but it does not prescribe action-set signatures as the group
definition. Retail has one shared tool and policy domain. A domain-level group
provides enough selected-versus-control evidence for Memory attribution, while
still preventing cross-domain Retail/Airline mixing.

## Legacy Rollouts

Earlier Stage 8 events use `retail-actions-v1:<sha256>`. They remain immutable
audit records. The Slow Loop accepts this explicit legacy format and
canonicalizes it to `retail-v2` while building evidence. No rollout, reward,
Memory ID, or snapshot is rewritten, so the on-policy snapshot lineage remains
valid. New Fast Loop runs emit `retail-v2` directly.
