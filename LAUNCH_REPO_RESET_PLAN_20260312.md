# Launch Repo Reset Plan (2026-03-12)

## Goal
Recreate `NANOLLM` GitHub repository from a clean launch-ready root while preserving full audit trail in local archive.

## Current local state
- Clean root prepared in `E:\testmob`
- Archived legacy material in `E:\testmob\archive\2026-03-12_launch_cleanup\root_archive`
- Move log in `E:\testmob\archive\2026-03-12_launch_cleanup\MOVE_LOG_root_cleanup.json`
- Launch manifest in `E:\testmob\LAUNCH_ROOT_MANIFEST_20260312.json`

## Launch payload (keep in new repo root)
- `PAPER_NANOLLM.md`
- `RUN_DECISIONS.md`
- `TASKS.md`
- `REPRO_CHANGELOG_20260312_V13C_QNANOLORA.md`
- `KAGGLE_V13C_QNANOLORA_MERGE_CLEAN_CELL.md`
- `KAGGLE_COMPRESSION_METHODS_BENCHMARK_CELL.md`
- `production_candidate_v13c_qnanolora_20260312_094722_UTC.zip`
- `README.md`

## Recommended GitHub reset strategy
1. Archive old remote repo content (or export as backup branch/tag).
2. Create a new empty repository `NANOLLM`.
3. Initialize local git in `E:\testmob` (if not already).
4. Commit only launch payload plus optional `archive/2026-03-12_launch_cleanup` metadata docs (without heavy payloads if not desired).
5. Push as fresh `main` history.
6. Add release tag `v2026.03.12-launch-candidate`.

## Optional release packaging policy
- Keep heavy historical artifacts only in local archive or external object storage.
- Keep in repo only:
  - final candidate zip
  - benchmark JSON summaries
  - reproducibility docs
  - canonical Kaggle cells

## Notes
- During this session, direct connectivity to `github.com` from shell was unavailable, so remote inspection/deletion was not executable from this environment.
