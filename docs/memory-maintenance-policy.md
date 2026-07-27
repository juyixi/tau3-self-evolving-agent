# Memory Maintenance Policy

The Fast Loop constrains memory growth before a maintenance round and then performs
an auditable LLM review.

- Each episode keeps at most two new Tips and one new item for each other tier.
- Maintenance reads at most 100 items per tier, but records a per-tier cursor in
  `maintenance_state.json`; later rounds continue from the next page instead of
  repeatedly inspecting a stable ID prefix.
- Near-duplicate embedded memories are ordered before the regular page so the
  maintenance policy sees likely merge or retirement candidates first.
- The policy returns `reviews` with `keep`, `merge`, or `retire` dispositions,
  alongside executable lookup, merge, or delete commands.
- When active Tips exceed the configured capacity (default: 200), a review and a
  merge/delete command over presented Tips are mandatory. An empty maintenance
  decision is rejected and repaired once.

The cursor is operational state only. It does not alter historical rollout JSONL,
memory IDs, snapshots, or OPD source evidence.
