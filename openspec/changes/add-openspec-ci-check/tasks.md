## 1. Implementation
- [ ] 1.1 Create CI workflow file (`.github/workflows/openspec-validate.yml`)
- [ ] 1.2 Add `openspec validate` step to existing CI (lint / tests) jobs
- [ ] 1.3 Document the CI requirement in `openspec/project.md` and README
- [ ] 1.4 Add example command for local validation to developer docs (e.g., `npm run openspec-validate`)
- [ ] 1.5 Run validation in a test PR and iterate

## 2. Verification
- [ ] 2.1 Create a test change in `openspec/changes/test-change` and confirm CI validates it
- [ ] 2.2 Confirm developer documentation explains how to run validation locally

## 3. Rollout
- [ ] 3.1 Merge CI workflow behind a feature flag if necessary
- [ ] 3.2 Monitor PR failures and help contributors fix validation issues
