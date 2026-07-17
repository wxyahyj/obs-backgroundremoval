#!/usr/bin/env bash
set -euo pipefail

# Gates Status — read-only catalog report (does NOT execute gates).
# Reference: scale-engine/docs/workflow/GATES_AND_SCORE.md

JSON=0
[ "${1:-}" = "--json" ] && JSON=1

if [ "$JSON" -eq 1 ]; then
  cat <<'EOF'
{
  "version": "2.1",
  "core_gates": ["G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"],
  "meta_governance_gates": ["G9", "G10", "G11", "G12", "G13", "G14", "G15"],
  "enhanced_gates": ["G16", "G17", "G18", "G19", "G20", "G21", "G22"],
  "gates": [
    {"id":"G0","name":"Build","tier":"core","blocking":true,"phase":"execute"},
    {"id":"G1","name":"Exploration","tier":"core","blocking":true,"phase":"explore"},
    {"id":"G2","name":"Planning","tier":"core","blocking":true,"phase":"plan"},
    {"id":"G3","name":"TDD","tier":"core","blocking":true,"phase":"execute"},
    {"id":"G4","name":"Lint","tier":"core","blocking":true,"phase":"verify"},
    {"id":"G5","name":"Test","tier":"core","blocking":true,"phase":"verify"},
    {"id":"G6","name":"TypeCheck","tier":"core","blocking":true,"phase":"verify"},
    {"id":"G7","name":"Security","tier":"core","blocking":true,"phase":"security"},
    {"id":"G8","name":"NoAISlop","tier":"core","blocking":false,"phase":"consolidate"},
    {"id":"G9","name":"KnowledgeUpdated","tier":"meta","blocking":false,"phase":"consolidate"},
    {"id":"G10","name":"ResourceGovernance","tier":"meta","blocking":false,"phase":"consolidate"},
    {"id":"G11","name":"ContextBudget","tier":"meta","blocking":false,"phase":"plan"},
    {"id":"G12","name":"SessionHealth","tier":"meta","blocking":false,"phase":"consolidate"},
    {"id":"G13","name":"MultiAgentCoordination","tier":"meta","blocking":false,"phase":"execute"},
    {"id":"G14","name":"SkillUsage","tier":"meta","blocking":false,"phase":"execute"},
    {"id":"G15","name":"EngineeringStandards","tier":"meta","blocking":false,"phase":"verify"},
    {"id":"G16","name":"CommitDiscipline","tier":"enhanced","blocking":true,"phase":"consolidate"},
    {"id":"G17","name":"DocumentationHygiene","tier":"enhanced","blocking":false,"phase":"consolidate"},
    {"id":"G18","name":"RuntimeEvidence","tier":"enhanced","blocking":true,"phase":"verify"},
    {"id":"G19","name":"CodeReview","tier":"enhanced","blocking":true,"phase":"review"},
    {"id":"G20","name":"SupplyChain","tier":"enhanced","blocking":true,"phase":"security"},
    {"id":"G21","name":"ContextBudget","tier":"enhanced","blocking":false,"phase":"plan"},
    {"id":"G22","name":"SessionHealth","tier":"enhanced","blocking":false,"phase":"consolidate"}
  ],
  "profiles": {
    "fast-lane": {"gates": ["G0", "G4", "G5"], "skip": ["G1", "G2", "G3", "G6", "G7", "G8", "G9-G22"]},
    "standard":  {"gates": ["G2", "G4", "G5", "G6", "G16"], "skip": ["G1", "G3", "G7", "G8", "G9-G15", "G17-G22"]},
    "full":      {"gates": ["G1", "G2", "G3", "G4", "G5", "G6", "G16", "G17", "G18"], "skip": ["G7", "G8", "G9-G15", "G19-G22"]},
    "comprehensive": {"gates": ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G16", "G17", "G18", "G19", "G20"], "skip": ["G8", "G9-G15", "G21", "G22"]}
  }
}
EOF
  exit 0
fi

cat <<'EOF'
=== SCALE Engine Gate Catalog (v2.1) ===

Core Gates (G0–G8) — workflow verification, preflight, product smoke
  G0  Build                blocking   execute
  G1  Exploration          blocking   explore       (≥3 files + main contradiction)
  G2  Planning             blocking   plan          (5 keyword groups + 3 companion artifacts)
  G3  TDD                  blocking   execute       (test files exist & newer than plan)
  G4  Lint                 blocking   verify
  G5  Test                 blocking   verify
  G6  TypeCheck            blocking   verify
  G7  Security             blocking   security      (secrets / injection / .env)
  G8  NoAISlop             advisory   consolidate

Meta-Governance Gates (G9–G15) — scale meta-governance
  G9  KnowledgeUpdated     advisory   consolidate
  G10 ResourceGovernance   advisory   consolidate
  G11 ContextBudget        advisory   plan
  G12 SessionHealth        advisory   consolidate
  G13 MultiAgentCoordination advisory  execute
  G14 SkillUsage           advisory   execute
  G15 EngineeringStandards advisory   verify

Enhanced Gates (G16–G22) — commit discipline, runtime, review, supply chain
  G16 CommitDiscipline     blocking   consolidate   (>25 files / >180min / >1MB staged / whitespace)
  G17 DocumentationHygiene advisory   consolidate   (broken internal markdown links)
  G18 RuntimeEvidence      blocking   verify         (≥1 exit-code-0 record)
  G19 CodeReview           blocking   review         (L/CRITICAL only — Findings + Residual Risk)
  G20 SupplyChain          blocking   security       (no CRITICAL/HIGH vulns; lock file consistent)
  G21 ContextBudget        advisory   plan           (token usage vs configured budget)
  G22 SessionHealth        advisory   consolidate   (stale worktrees / orphaned state)

Profiles
  fast-lane       G0, G4, G5                                         (S-level: typo / comment / config)
  standard        G2, G4, G5, G6, G16                                (M-level: 2-5 files behavior change)
  full            G1, G2, G3, G4, G5, G6, G16, G17, G18              (L-level: cross-module / refactor)
  comprehensive   G1, G2, G3, G4, G5, G6, G7, G16, G17, G18, G19, G20 (CRITICAL: auth / money / migration)

Usage
  bash scripts/gates/all.sh                          # run all gates (default profile: comprehensive)
  bash scripts/gates/all.sh --profile fast-lane      # S-level fast path
  bash scripts/gates/all.sh --profile standard       # M-level
  bash scripts/gates/all.sh --profile full           # L-level
  bash scripts/gates/all.sh --dry-run                # check which scripts exist, don't run
  bash scripts/gates/G{n}-verify.sh                  # run a single gate
  bash scripts/gates/status.sh                       # this catalog (read-only)
  bash scripts/gates/status.sh --json                # machine-readable catalog

Reference: docs/workflow/GATES_AND_SCORE.md (scale-engine v0.50.x)
EOF
