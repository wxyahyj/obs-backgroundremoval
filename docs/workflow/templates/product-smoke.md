# Product Smoke

## Real Product Path

Describe the smallest end-to-end path that proves the change works through the real product boundary.

Example:

```text
UI or client -> gateway/router -> service -> database/storage/queue -> observable result
```

Do not use a green health endpoint as the only proof when the user-facing path depends on routing, authentication, storage, async tasks, browser behavior, or third-party integration.

## Quick Setup

1. Open `.scale/product-smoke.json`.
2. Replace the example command with one real product path command.
3. Set that probe's `enabled` field to `true`.
4. Run `scale preflight --profile productSmoke --json`.
5. Run `scale runtime final-check --level M --json`.

`status: "skipped"` means no real product path was exercised. It does not count as completion evidence.

## Setup

- Base URL:
- Test user or tenant:
- Required fixtures:
- Services that must be running:

## Smoke Commands

| Command | Expected Result | Evidence Artifact |
| --- | --- | --- |
| TBD | TBD | TBD |

## Runtime Evidence

Record at least one runtime evidence item:

```bash
scale runtime record \
  --kind command \
  --title "Product smoke: <flow>" \
  --status passed \
  --command "<exact smoke command>" \
  --exit-code 0 \
  --summary "<business result, task id, status, or observable output>" \
  --artifacts ".agent/logs/<service>/<smoke>.json" \
  --metadata-json '{"productSmoke":true,"realProductPath":true}'
```

## Assertions

- [ ] Request crossed the real product boundary, not only an isolated unit.
- [ ] Authentication or user identity path was exercised when relevant.
- [ ] Persistence/storage/queue side effect was verified when relevant.
- [ ] Async task or eventual state was polled to terminal status when relevant.
- [ ] Failure output is specific enough to diagnose the failing layer.
- [ ] Runtime artifacts are ignored or deliberately promoted according to resource governance.
