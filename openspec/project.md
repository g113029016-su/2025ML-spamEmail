# Project Context

## Purpose
This repository holds OpenSpec-driven project artifacts and any code/config needed to implement the project's capabilities. It uses OpenSpec to capture requirements, propose changes, and validate spec deltas before implementation. The goal is to keep specs as the single source of truth and to standardize how proposals, tasks, and design decisions are documented.

> Assumption: This workspace is a Node/npm-centric project (you ran `npm install -g @fission-ai/openspec` earlier), so guidance and automation below assume Node.js toolchain and CI integration that supports npm/PowerShell on Windows.

## Tech Stack
- Node.js (LTS)
- npm (global/cli usage)
- OpenSpec (`@fission-ai/openspec`) for spec-driven proposals and validation
- Markdown-based specs and docs under `openspec/`
- (Optional) Recommended developer tools: ESLint, Prettier, Jest for tests

## Project Conventions

### Code Style
- Use Prettier for formatting (single source of truth). If Prettier config is not present, add one at repo root.
- Use ESLint for linting with a shared config (extend recommended rules). Keep lint rules permissive for early-stage projects; tighten before major releases.
- File / symbol naming: kebab-case for change ids and directories (e.g. `add-two-factor-auth`), camelCase for JS/TS symbols, PascalCase for component/classes.

### Architecture Patterns
- Keep capabilities focused per folder under `openspec/specs/<capability>/`.
- Prefer small, single-purpose modules. If the project grows beyond simple scripts, consider a service/module split (e.g., `packages/` monorepo).
- Keep `specs/` as the runtime truth; changes must produce deltas which land in `openspec/changes/` before implementation.

### Testing Strategy
- Unit tests: Jest recommended for Node/TS projects. Place tests alongside source files or under `__tests__`.
- Spec validation: run `openspec validate <change-id> --strict` for any proposal before requesting approval.
- CI should run: lint, tests, and `openspec validate` on PRs.

### Git Workflow
- Main branch: `main` or `master` (use existing repo convention).
- Development: create short-lived feature branches `add-...`, `update-...`, `fix-...` and open PRs for review.
- Commit messages: short prefix + description (e.g., `feat: add openspec CI validation`), or follow Conventional Commits if preferred.

## Domain Context
- Primary domain and capabilities should be captured in `openspec/specs/` as human-readable requirements and scenarios. If you have domain-specific terms or business rules, add them here to help AI assistants and future contributors.

## Important Constraints
- Current environment: Windows with PowerShell (you ran installs from PowerShell). CI or dev scripts should support PowerShell or be documented for cross-shell usage.
- Network or permission constraints: global npm installs may require elevated permissions on some machines. Prefer using npx or project-local installs where practical.

## External Dependencies
- `@fission-ai/openspec` — primary tool for spec validation and scaffolding.
- Node/npm — runtime for tooling and scripts.
- (Optional) GitHub Actions / other CI provider for automating validation and tests.

## Notes & Next Steps
- If you want, I can add a small GitHub Actions workflow to run `openspec validate` on PRs; I've scaffolded a change proposal for that as `add-openspec-ci-check` (see `openspec/changes/add-openspec-ci-check/`).
- If any of the assumptions above are incorrect (e.g., project is not Node-based or you prefer another stack), tell me and I'll revise this file.
