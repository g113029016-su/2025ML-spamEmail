## ADDED Requirements

### Requirement: CI SHALL run OpenSpec validation on proposed changes
The project's continuous integration pipeline SHALL run `openspec validate <change-id> --strict` (or equivalent) for any pull request that modifies files under `openspec/changes/`.

#### Scenario: Pull request touches a proposed change
- **WHEN** a PR includes new or modified files under `openspec/changes/`
- **THEN** the CI pipeline SHALL run `openspec validate` for the impacted change-id
- **THEN** the pipeline SHALL fail the job if validation reports errors

#### Scenario: Local validation
- **WHEN** a developer runs the local validation command (e.g., `npm run openspec-validate`)
- **THEN** `openspec validate <change-id> --strict` SHALL run and return non-zero exit code on validation errors

