# Plan - {{TASK_ID}}

Date: {{DATE}}
Level: {{LEVEL}}

## Goal

- Problem / why now:
- Desired outcome (goal-driven, not just a task list):
- Success looks like:

## Preconditions

- Required state before starting:
- Dependencies / blockers:

## Scope / Boundary

- In scope:
- Out of scope / non-goals:

## Approach

1. TBD
2. TBD
3. TBD

## Steps with Gates

| # | Step | Gate it satisfies |
| --- | --- | --- |
| 1 | Explore ≥3 files, record main contradiction | G1 |
| 2 | This plan: scope / exception / rollback / acceptance | G2 |
| 3 | Implement; change `src/` ⇒ change `tests/` | G3 |
| 4 | Run lint / tests / verify | G4 / G5 / G6 |

## Exception / Failure Paths

- Expected failure:
- Error handling:
- Manual recovery:

## Rollback / Fallback

- Rollback:
- Fallback:
- Disable path:

## Acceptance Criteria

- TBD

## Human Confirmation (L / CRITICAL)

- L/CRITICAL requires human confirmation / review before execution.
- Who confirms / when:
- For S/M: write `N/A`.

## Verification

- `make gate-quality`
- `make verify PROFILE=default`
- `git diff --check`
