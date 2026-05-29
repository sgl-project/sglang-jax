# Release Process

This document describes the SGLang-JAX release flow, its entry-point workflows, and the recovery paths for failures at each stage.

## Architecture Overview

The release process uses a **two-phase** design:

1. **Tag phase** (`tag.yml`) validates the version, runs build and smoke checks, then creates and pushes an annotated tag from the current ref.
2. **Publish phase** (`publish.yml`) checks out the pushed tag, publishes the wheel to PyPI and the Docker image to Docker Hub in parallel, then creates the GitHub Release after both publish jobs complete.

`release.yml` is the **single user-facing entry point**. It chains the two phases with `workflow_call`, so maintainers do not need to manually jump between workflows for a normal release. `tag.yml` and `publish.yml` still support `workflow_dispatch`, but only for recovery scenarios described below.

## Starting a Release

Normally, dispatch the `Release` workflow and provide:

- `version`: a PEP 440 version without the `v` prefix, such as `0.0.3`, `0.0.3rc0`, or `0.0.3.post1`
- `dry_run`: when enabled, runs validation and build checks only; it does not push a tag or publish artifacts

Use `dry_run` to rehearse the full validation path before merging to the release branch.

## Failure Recovery

External side effects are created in this order: **push tag -> PyPI upload / Docker push in parallel -> GitHub Release**. Recover based on where the failure happened.

| Failure point | External side effects | Recovery |
| --- | --- | --- |
| Tag phase, before the tag is pushed | None | Fix the issue, then dispatch `release.yml` again with the same `version`. |
| Any publish job, after the tag is pushed | The tag exists on origin; either PyPI or Docker may already have succeeded | Dispatch `publish.yml` with the same tag. This checks out the tag and reuses the same release identity. |
| GitHub Release creation | The tag, PyPI package, and Docker image may already exist | Dispatch `publish.yml` with the same tag, or rerun failed jobs from the GitHub UI. |
| Transient infrastructure failure | Depends on the failure point | Within 90 days, use "Re-run failed jobs" in the GitHub UI when appropriate. |

### Non-Recoverable Cases

If the tag was pushed but the version itself is wrong, such as the wrong version number or the tag pointing at the wrong commit, **do not rewrite the tag**. The correct procedure is:

1. Delete the tag and any partially created GitHub Release from the GitHub UI.
2. If the package was already published to PyPI, yank that version. PyPI does not support true deletion.
3. Use a new, incremented version and rerun the pipeline.

## Related Workflow Files

- `.github/workflows/release.yml`: user-facing entry point that chains the two phases.
- `.github/workflows/tag.yml`: validation and tag creation.
- `.github/workflows/publish.yml`: publish orchestration, including metadata, parallel PyPI/Docker publishing, and GitHub Release creation.
- `.github/workflows/release-pypi.yml` and `.github/workflows/release-docker.yml`: concrete publish jobs.
- `scripts/ci/release_metadata.py`: PEP 440 version parsing and `is_prerelease` / `update_latest` flag computation.
