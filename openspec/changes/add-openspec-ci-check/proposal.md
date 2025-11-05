# Change: Add CI validation for OpenSpec

## Why
We want to ensure spec correctness and consistency by validating OpenSpec proposals automatically on pull requests. This reduces human error, prevents invalid spec deltas from being merged, and enforces the "specs are truth" principle.

## What Changes
- Add a CI check that runs `openspec validate <change-id> --strict` for proposed changes.
- Add guidance to docs (project.md) recommending CI validation on PRs.
- Provide a sample GitHub Actions workflow that runs `openspec validate` and basic lint/tests.

**BREAKING**: None — this is a tooling/CI enhancement.

## Impact
- Affected specs: none (tooling only)
- Affected code: repository CI configuration (e.g., `.github/workflows/openspec-validate.yml`)
- Developer workflow: PRs will require a passing OpenSpec validation run; authors may need to run `openspec validate` locally before pushing.

## Risks
- If `openspec validate` is not available in CI environment, CI runs will fail; resolution: install `@fission-ai/openspec` in the CI job or use `npx`.

## Rollout Plan
1. Add the proposal and tasks (this change).
2. Add a CI workflow that installs Node and `@fission-ai/openspec` and runs `openspec validate` for changed `openspec/changes/*` dirs.
3. Monitor PRs and iterate on runner image and caching.

***

Please review this proposal; if approved, I'll scaffold `tasks.md`, the `specs/` delta and a sample GitHub Actions workflow file.
