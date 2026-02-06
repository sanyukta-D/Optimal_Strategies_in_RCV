# Project Instructions for Claude

## Domain Rules

- When fixing multi-winner STV or RCV logic, ALWAYS read and compare against the original paper's code/formulas BEFORE proposing any changes. Never guess at election math — verify the exact algorithm step-by-step against the reference implementation first.
- Do not make speculative code changes. When unsure about the correct approach, STOP and ask the user before implementing. Present your understanding of the problem and proposed fix for approval before editing files.

## Testing / Verification

- After making any fix, verify that ALL existing election districts/cases still produce correct results before committing. Never fix one case at the expense of breaking another.
- Run the full test suite or spot-check all districts after every change.

## Workflow Rules

- Before editing any STV computation code, run the full test suite first to establish a baseline.
- After every edit, run tests before proceeding to the next change.
- Never change variable names or function signatures without grep-checking all call sites.
- When fixing multi-winner logic, trace the full round-by-round STV tabulation and compare against reference values before committing.

## Deployment Debugging

- When debugging deployment issues (e.g., Streamlit Cloud), check for large files, Git LFS issues, and repository size FIRST before investigating configuration or permissions problems.

## Repo State (as of Feb 6, 2026)

### Known-good baseline: commit `61515bf` (Feb 4, 2026)
- Local and remote (`origin/main`) are in sync at this commit
- All code works correctly — verified against 130+ elections
- Webapp repo (`sanyukta-D/RCV-Analyzer-Webapp`) is also in sync with this state
- Webapp deployed at: https://rcv-analyzer.streamlit.app/

### Git LFS & push issues
- The `.git` directory is ~691MB due to large NYC election CSVs tracked by LFS
- GitHub LFS budget is exceeded — LFS-tracked CSVs checkout as pointer files, not actual data
- Pushing to `Optimal_Strategies_in_RCV` fails (connection drops) due to repo size
- **NEVER force-push or make destructive git operations on this repo without explicit user approval** — a previous session accidentally wiped the remote by pushing a bad commit during an LFS migration attempt
- For deployment changes, use the lightweight webapp repo (`RCV-Analyzer-Webapp`) instead

### What NOT to do
- Do not attempt to remove/migrate Git LFS — this caused the Feb 6 incident where all files were deleted from the remote
- Do not push large commits to this repo — the connection will drop
- Do not amend or rewrite history on this repo
