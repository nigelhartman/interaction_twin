# Source-only review and portable checks

The review ZIP contains explicitly selected source files, one original vector diagram, and the named Three.js files with their MIT license. It does not contain recordings, generated catalogs, saved demonstration results, model code/checkpoints, prepared environments, API keys, or the project films. It is a private source review snapshot, not a completed installer or a hosted service.

The clean-source smoke used an existing Python 3.12 environment and an existing Node runtime on the development machine. No dependency installation, checkpoint download, API request, or learned-model inference was performed for these checks. A successful smoke therefore does not prove that a fresh installation or a new-video reconstruction works on another computer.

## Start with an empty local workspace

Prepare the runtime described in [README.md](../README.md). The tested Python environment already had NumPy, Pillow, SciPy, OpenCV, trimesh and jsonschema. This is an observed prerequisite list for the selected checks, not a pinned or independently installed dependency recipe. Model inference needs additional model-specific prerequisites from README.

From the extracted source root, set `IT_PYTHON` to your prepared Python executable. The archive does not supply that environment:

```sh
IT_PYTHON=/absolute/path/to/prepared/python
"$IT_PYTHON" -B scripts/build_workspace.py
"$IT_PYTHON" -B scripts/launch_workspace.py --workspace
```

The empty build should report `cases: 0` and `errors: []`. The first-project screen and New video dialog are usable without saved examples. Hosted-processing consent starts unchecked. Missing model/runtime requirements remain explicit; opening the dialog does not run the pipeline. No saved Jabra/camera/wallet example or movie is bundled.

The launcher may report missing runtime requirements before starting. For source-only UI inspection with an already prepared interpreter, the lower-level loopback server can be started separately:

```sh
"$IT_PYTHON" -B scripts/serve_workspace.py --host 127.0.0.1 --port 18867
```

Open `http://127.0.0.1:18867/workspace/index.html`. Choose an unused port; leave other app instances alone. Saved result cards should report unavailable when their private data is absent. Starting this server does not supply model weights or API access.

## Portable Python subset

These checks use analytic or temporary synthetic fixtures. They test strict procedure and reconstruction contracts, camera/rigid/gauge math, native GLB appearance, inferred-layer boundaries, interval-motion evidence splits, SQLite budget admission, mocked fixed-origin transport, path confinement, upload/consent boundaries, and the single-worker rule. The HTTP tests bind ephemeral loopback ports. No external API or model runtime is called.

Create an empty `data/` directory for temporary fixtures. `PYTHONPATH=.:tests` supports the preserved direct imports between test helpers; it must be set from the source root. Do not copy private fixtures or alter tests to make this subset pass.

```sh
mkdir -p data
PYTHONPATH=.:tests PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 "$IT_PYTHON" -B -m unittest -v \
  tests.test_geometry \
  tests.test_consistency \
  tests.test_relative_gauge \
  tests.test_stage5 \
  tests.test_reconstruction_contract \
  tests.test_native_appearance \
  tests.test_observed_prior_mask_filter \
  tests.test_observed_surface_completion \
  tests.test_observed_depth_completion \
  tests.test_interval_motion \
  tests.test_interval_motion_search \
  tests.test_interval_motion_dense_input \
  tests.test_interval_motion_density \
  tests.test_meta_budget \
  tests.test_meta_api \
  tests.test_upgrade_api \
  tests.test_launch_workspace \
  tests.test_mcp_assessment \
  tests.test_submission_review_artifacts \
  tests.test_pipeline.PipelineContracts.test_static_paths_and_origin \
  tests.test_pipeline.PipelineContracts.test_showcase_static_paths_are_confined_without_changing_workspace \
  tests.test_pipeline.LocalHTTPContracts.test_health_and_config \
  tests.test_pipeline.LocalHTTPContracts.test_cross_origin_write_and_missing_consent_rejected \
  tests.test_pipeline.LocalHTTPContracts.test_bounded_upload_and_single_active_worker
```

The provisional clean snapshot passed **195 tests, zero failures/errors/skips** on 1 November 2026. A later fifth review-artifact copier test is included in the final source selection; the final snapshot receipt records the exact rerun count and source hashes. The first provisional attempt preserved three setup errors (missing test-helper import path and empty `data/`); the setup above resolved them without editing tests.

## Optional Node-backed guards and projection math

With an existing `node` executable on PATH, these tests exercise the shipped app/showcase loading controllers with synthetic DOM/WebGL doubles. The exact two lazy-case methods below avoid the separate private-catalog integrity test, which needs omitted saved cases.

```sh
PYTHONPATH=.:tests PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 "$IT_PYTHON" -B -m unittest -v \
  tests.test_lazy_selection \
  tests.test_scene_texture_lifecycle \
  tests.test_component_selection \
  tests.test_lazy_cases.LazySceneTests.test_derivative_keeps_exact_scene_and_original_source_binding \
  tests.test_lazy_cases.LazySceneTests.test_actual_browser_loader_rejects_changed_or_foreign_data_and_bounds_cache
node tests/geometry_camera_harness.mjs
```

The provisional source snapshot passed **17 guard tests**. The projection harness checked **10,000 actual Three.js projections and four letterbox cases**, with maximum discrepancy below `2.35e-13` pixels. These are numerical/controller checks, not browser GPU pixel tests, physical pose validation, or proof of successful learned reconstruction.

## What cannot be reproduced from the code-only ZIP

- The full private-checkout test suite includes real saved-artifact/hash regressions and is not the portable subset above. Do not describe its count as a source-only result.
- The bounded Muse-directed MCP demo is deliberately pinned to an approved private Jabra source/program/motion version. Its five-tool, two-round result and previous failed attempts are omitted from the source ZIP. Guard tests use fixtures and do not rerun Meta.
- New-video tracking/reconstruction needs the appropriate installed model repositories and checkpoints, a supported runtime, and optional API credentials/explicit consent for hosted stages. Missing back surfaces, scale, assembly, contact and simulator readiness are not established by startup or these tests.
- README file links target explicitly selected source files and current safe documentation. The earlier operational README is mentioned as local history without a link and is omitted. Generated review downloads, source captures and demonstration packages are separate private artifacts.

The recommended reviewer first step is the empty workspace plus the portable checks above. Assess saved demonstrations separately with their original source hashes, recorded model receipts and stated limitations.
