# . Workflow

Governance mode: standard
Governance pack: standard

## Task Levels

| Level | Use for | Required artifacts |
| --- | --- | --- |
| S | typo, comments, small local edits | relevant validation only |
| M | bug fixes, new APIs, 2-5 files | explore, skill plan, plan, verification, review, summary |
| L | cross-module or architecture changes | full artifacts plus human confirmation |
| CRITICAL | auth, permissions, migrations, production config | rollback plan, security review, full verification |

## Standard Task Directory

```text
.planning/tasks/<yyyy-mm-dd>-<task-slug>/
├── explore.md
├── mini-prd.md
├── plan.md
├── runtime.md
├── reality-check.md
├── resource-cleanup.md
├── verification.md
├── review.md
├── summary.md
├── artifact-manifest.json
└── artifacts/
    ├── index.html
    └── release-report.html
```

## Verification

Use service-aware verification when configured:

```bash
scale preflight --service all
scale preflight --service all --preflight-profile full
scale verify <task-id> --profile default
scale verify <task-id> --service <service-name>
scale verify <task-id> --artifact-gate warn
scale verify <task-id> --artifact-gate block
scale verify <task-id> --require-installed-skills
scale verify <task-id> --profile productSmoke
scale task-artifacts check --dir .planning/tasks/<task-dir> --level L
scale artifact render --task-id <task-dir> --type release-report
scale artifact doctor --task-id <task-dir>
```

Keep `.scale/verification.json` as the source of truth for profiles and service commands.
Keep `.scale/skills.json` as the source of truth for active skill routing policy.
Keep `.scale/output-policy.json` as the source of truth for derived HTML artifact types, source Markdown mapping, security policy, and Git retention behavior.
Keep `.scale/resource-policy.json` and `.scale/assets.json` as the source of truth for generated reports, temporary files, module documentation, media, reusable scripts, and Git retention policy.
Keep `.scale/engineering-standards.json` and `.scale/frameworks.json` as the source of truth for logging, security, ORM, architecture, framework, UI/UX, testing, and coding standard checks.
Keep `.scale/engineering-standards-baseline.json` as the temporary exception list for known legacy standards findings; it must not be used to hide new or changed-file problems.
Use `artifactGate: "warn"` while introducing the workflow, then move M/L/CRITICAL work to `"block"` once templates and local gates are stable.

## Workflow Upgrade

Do not rerun `scale init` as a blind upgrade command. Generated governance files may contain local project adaptations.

Use the guarded upgrade flow:

```bash
scale upgrade check --dir .
scale upgrade plan --dir . --html
scale upgrade apply --dir . --confirm
scale upgrade rollback --dir .
scale tools outdated --dir .
scale skill outdated --dir .
scale preflight --preflight-profile quick
```

Rules:

- `.scale/governance.lock.json` records generated file hashes and pack versions.
- Clean or missing generated files can be planned safely.
- Locally changed generated files require manual review before replacement or merge.
- `scale upgrade apply --confirm` only restores missing generated files and refreshes the lock after writing `.scale/backups/upgrade-*/manifest.json`.
- `scale upgrade rollback` only rolls back the latest SCALE-managed safe apply.
- Third-party skills, MCP servers, browser tools, desktop automation, and external CLI tools are never auto-installed by the upgrade flow.
- Community sources require source, install script, permission, and changelog review. Desktop automation is treated as high risk.

## HTML Artifacts

Markdown remains the editable source of truth for task artifacts. HTML artifacts are derived human-review surfaces for plan comparison, implementation plans, code reviews, status reports, incident reports, and release reports.

Use HTML when a human needs to compare, review, or sign off. Keep source Markdown, manifest metadata, and safety checks in place so the derived HTML stays traceable and does not leak secrets or remote scripts.

## Active Skill Routing

SCALE plans required skills from task description, service selection, and changed files. UI/API work requires a Mini-PRD plus domain evidence such as `ui-spec.md`, `visual-review.md`, or `api-contract.md`. Security and database work require explicit review or rollback artifacts.

Tool orchestration is part of the workflow contract:

- UI/UX work requires `impeccable` as the deterministic anti-pattern gate, and should use `taste-skill`, `awesome-design-md`, `ui-ux-pro-max`, and `frontend-design` as direction/supporting skills alongside browser screenshots, responsive checks, and visual review evidence.
- Web research, logged-in pages, and dynamic browser work require `web-access` evidence, source citations, and browser/network/console evidence when available.
- Browser E2E work should combine `webapp-testing`, Playwright, Agent Browser, web-access, or Chrome DevTools MCP according to the target and record screenshots plus console/network findings.
- Desktop or client-side GUI automation uses CUA/computer-use only with explicit operator-safety notes, desktop screenshots, and a side-effect boundary.
- External agent or CLI orchestration such as Codex, Gemini CLI, OpenCode, WPS, or WeChat automation must record version checks, exact commands, output summaries, and dry-run or safe-mode evidence.

When a task records `servicesTouched`, `scale verify <task-id>` uses those services automatically. You can still override selection with `--service all`, `--service api`, or `--service api,gateway`.

Before M/L work, check whether required workflow skills are physically installed:

```bash
scale skill doctor --json
scale skill check --require-installed --json
```

## Workspace Lifecycle

Before finishing an agent-created branch or deleting a temporary worktree, inspect root and child repository state:

```bash
scale workspace status --json
scale workspace finish --summary
scale workspace finish --json
scale workspace cleanup --dir <temporary-worktree> --dry-run --json
scale workspace cleanup --dir <temporary-worktree> --apply --confirm <branch-or-head> --json
```

Do not remove a temporary worktree while any submodule or nested repository has uncommitted or unpushed work. Child repositories must be committed and reviewed in their own remotes, then the root repository can record any required pointer or governance updates. Cleanup defaults to dry-run. Applying cleanup requires the reported confirmation token, normally the temporary branch name.

Use `scale ship <task-id>` for governed commits. It checks MOE/submodule child repository state before staging reviewed root files, so dirty or unpushed child work cannot be hidden inside a root commit. It also enforces the GitLab Flow branch lifecycle: work happens on short branches, merges target `dev`, production lands on `master`, and release publishing is triggered by user-created `vX.Y.Z` tags. Direct governed commits on `dev`, `master`, `main`, or detached HEAD are blocked. Raw `git add .` is outside the governed path and must not be used for MOE releases.

## Resource Governance

Use asset scanning before committing generated reports, media, temporary scripts, or long-lived documentation changes:

```bash
scale assets scan --json
scale assets doctor --json
scale assets settle --task-id <task-id> --artifact-dir .planning/tasks/<task-dir>
```

Default policy:

- maintained module docs, standards, contracts, ADRs, reusable scripts: commit and keep current.
- task planning, verification, runtime-contract, reality-check, and cleanup artifacts: keep in `.planning/tasks`; promote final truth to maintained docs when useful.
- screenshots, videos, E2E reports, coverage, temporary scripts, and runtime logs: keep out of Git unless explicitly promoted.
- large media: use Git LFS or external artifact storage instead of normal Git history.

## Engineering Standards

Use standards scanning before reviewing or shipping M/L/CRITICAL work:

```bash
scale standards scan --json
scale standards doctor --json
scale standards doctor --changed --json
scale standards doctor --changed-files src/example.ts,src/example.test.ts --json
scale standards baseline --write --artifact-dir .planning/tasks/<task-dir> --task-id <task-id> --json
scale standards settle --task-id <task-id> --artifact-dir .planning/tasks/<task-dir>
scale preflight --preflight-profile full --json
scale verify <task-id> --json
```

Default policy:

- ad-hoc console/output logging is allowed only for CLI/script paths.
- sensitive fields such as token, password, secret, authorization, cookie, and credentials must not be logged.
- hardcoded secret-like assignments are blocked before review or release.
- SQL must use parameterized queries, ORM bind parameters, or safe query builders.
- unsafe HTML sinks, dynamic code execution, empty catch blocks, and type suppressions require remediation before release.
- framework and architecture rules live in `.scale/frameworks.json` and module standards docs.
- `.scale/frameworks.json > bannedImports` blocks direct use of deprecated ORMs, unsafe SDKs, or off-system UI components.
- `.scale/frameworks.json > lastReviewedAt/reviewIntervalDays` warns when module framework decisions need review.
- `.scale/engineering-standards.json > blockingRules` promotes selected warning rule IDs to release-blocking findings.
- `.scale/engineering-standards.json > allowedFindingPatterns` allows narrow rule/path/evidence exceptions without hiding unrelated findings in the same file.
- `.scale/engineering-standards-baseline.json` may hold known legacy findings during rollout, but normal task gates should prefer `--changed` or `--changed-files` so new work is blocked without forcing a whole-repo cleanup.
- `.scale/verification.json > policy.engineeringStandardsGate` controls whether preflight and task verification treat standards as `off`, `warn`, or `block`.
- `.scale/product-smoke.json` defines real product-path probes. Use it to prove a routed user/business flow, not only build, unit tests, or `/health`.
- `.scale/verification.json > policy.productSmokeGate` controls whether missing or failed product smoke evidence warns or blocks M/L/CRITICAL delivery.
- `.scale/verification.json > policy.ecosystemReadinessGate` controls whether CI preflight treats missing governed tools or workflow skills as `off`, `warn`, or `block`; `policy.ecosystemReadinessSkillScope` selects `required`, `recommended`, or `all` skill tiers for blocking.
- Full standards scans are for release readiness, scheduled remediation, and architecture cleanup. Changed-file scans are the default for day-to-day feature and bug branches.
- Use `scale standards baseline --write` only during an explicit rollout or remediation planning task. It writes the machine-readable baseline and a `standards-legacy-debt.md` classification report for staged cleanup.

## Automation Templates

Optional automation templates are generated under `docs/workflow/templates/`:

- `github-actions-scale-preflight.yml`: CI workflow that runs `scale preflight --service all --preflight-profile ci`.
- `pre-push-scale-preflight.sh`: local pre-push hook template that runs the default quick preflight.

Keep these templates advisory until `scale preflight --service all --preflight-profile full` is reliable locally for the project.
