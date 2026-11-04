# Run Interaction Twin

## Browse the saved demo

The [public website](https://nigelhartman.github.io/interaction_twin/) needs no installation or API key. Its five examples are saved results. Filtering, 3D rendering and supported robot-task edits run in the browser.

For an offline static copy:

```sh
git clone --branch gh-pages --single-branch https://github.com/nigelhartman/interaction_twin.git demo-root/interaction_twin
python3 -m http.server 8000 --directory demo-root
```

Open `http://localhost:8000/interaction_twin/`. This version has no upload, inference or hosted MCP backend.

## Work on the application

```sh
git clone https://github.com/nigelhartman/interaction_twin.git
cd interaction_twin/app
```

The application lives together under `app/`: Python modules, scripts, schemas, frontend templates, profiles and tests. Run commands from that directory.

Prepare a Python environment with the dependencies used by your selected workflow. The `requirements-*.lock.txt` files record the development environments: local geometry, media, Meta API, MCP and optional segmentation. They are separate environment snapshots, not one combined installer. The core review build uses NumPy, Pillow, SciPy, OpenCV, trimesh and jsonschema; media rendering also uses FFmpeg. Model workflows need their upstream runtimes and weights under the upstream terms.

Once your environment is prepared:

```sh
python -B scripts/build_workspace.py --no-replays
python -B scripts/serve_workspace.py --host 127.0.0.1 --port 8766
```

Use the local URL printed by the server. The code-only checkout opens an empty library: saved recordings, generated models and owner state are deliberately separate from source. The empty-catalog build was checked after the directory move. A fresh dependency installation has not been reproduced from this release.

## Owner workflow

The local `/admin/` desk provides owner setup and sign-in. There is no shipped password or authenticated session. See [owner access](admin-api.md) and [review, approval and memory](admin-workflow.md).

Opening the app starts no inference. Fresh processing needs explicit owner consent, configured Meta access, spending guards and a GPU worker for SAM 3D Objects. The public source supplies no credentials or working cluster allocation. Keep keys in the documented local configuration, outside Git and frontend code.

## Agent access

See the [MCP guide](mcp.md). The server runs locally over stdio; GitHub Pages does not host it. Configure your client with the prepared MCP interpreter and `<repository>/app/scripts/serve_twin_mcp.py --curated`. The curated catalog must be populated with approved results before it can return twins.

## Checks

Fixture tests can run without saved model outputs:

```sh
python -B -m unittest discover -s tests -p test_agent_memory.py
python -B -m unittest discover -s tests -p test_admin_workflow.py
python -B -m unittest discover -s tests -p test_library_query.py
node --test tests/test_robot_workbench_ui_v2.cjs tests/test_aligned_scene_contract.cjs
```

These checks passed after the source move: 27 Python checks and 27 Node checks, with three saved-data Node checks skipped. Tests that inspect recorded model outputs require their separately supplied exact artifacts. The complete development workspace had broader regression coverage; the small source-only checks are not a reproduction of model inference or a physical-accuracy benchmark.

See [release scope and attribution](PUBLIC_RELEASE.md) and the [technical overview](technical-overview-20261104.md) for the architecture and current limits.
