# Run Interaction Twin

## Browse the saved demo

The [public website](https://nigelhartman.github.io/interaction_twin/) needs no installation or API key. Its five owner-recorded examples are saved results. Filtering, 3D rendering and supported robot-task edits run in the browser.

For an offline static copy:

```sh
git clone --branch gh-pages --single-branch --depth 1 https://github.com/nigelhartman/interaction_twin.git demo-root/interaction_twin
python3 -m http.server 8000 --directory demo-root
```

Open `http://localhost:8000/interaction_twin/`. This version has no upload, inference or hosted MCP backend.

## Review the application source

Use **Python 3.12**, **Node.js 22 or newer**, and macOS or Linux. The local backend uses POSIX process and file-lock APIs; Windows users can use WSL. Fetch only `main` to avoid downloading the separate saved-demo branch.

```sh
git clone --branch main --single-branch --depth 1 https://github.com/nigelhartman/interaction_twin.git
cd interaction_twin/app
python3.12 -m venv .venv-review
source .venv-review/bin/activate
python -m pip install -r requirements-review.txt
python -B scripts/check_source_release.py
```

The pinned [review requirements](../app/requirements-review.txt) contain NumPy, JSON Schema validation and their small supporting packages. They install no model runtimes or weights. The check runs temporary synthetic fixtures, builds an empty catalog in a disposable source copy, starts its own loopback server, requests health and the landing page, and stops that child server. It makes no API/model requests and reads no owner recordings or credentials.

To open your source checkout after checking it:

```sh
python -B scripts/build_workspace.py --no-replays
python -B scripts/serve_workspace.py --host 127.0.0.1 --port 8766
```

Open the URL printed by the server. The code-only checkout has an **empty library**: recordings, generated models and owner state are deliberately separate from source. Use the public or offline static demo above to inspect saved examples. Starting the source server does not import the demo branch or launch model calls.

## What this setup verifies

The review recipe was installed in a fresh isolated CPython 3.12 environment on macOS arm64. The check covers owner approval and repair bounds, feedback-memory scoping, library queries, robot UI contracts, connected-scene math, empty-catalog generation and isolated HTTP startup. The fresh run passed 27 Python checks and 66 Node checks; three saved-data Node checks were explicitly skipped. Release entry-document links also resolve within the checkout. Eighteen memory/query fixtures were also checked with Python site packages disabled (`-I -S`).

An optional [CI recipe](ci/source-review.yml) provides the same command for Linux with Python 3.12 and Node.js 22. Remote CI is **not enabled** in this repository; the reported passes above are local checks. To enable CI later, copy the recipe to `.github/workflows/review.yml` using an appropriately authorized GitHub connection. These checks do not reproduce full model inference or measure physical accuracy.

## Owner processing workflow

The local `/admin/` desk provides owner setup and sign-in. There is no shipped password or authenticated session. See [owner access](admin-api.md) and [review, approval and memory](admin-workflow.md).

Fresh video processing additionally requires FFmpeg, the selected model-specific environment, explicit processing consent, Meta access, spending guards and a configured SAM 3D GPU worker. The `requirements-*.lock.txt` files preserve separate development-environment snapshots; they are not one combined installer. The public source supplies no credentials, model weights, existing cluster allocation or private activation receipts. The lightweight review environment alone cannot reconstruct a new video.

Keep keys outside Git and frontend code. Opening the app starts no inference or remote worker.

## Agent access

See the [MCP guide](mcp.md). MCP runs locally over stdio and needs its separate SDK environment. GitHub Pages does not host it. Configure your client with the MCP interpreter and `<repository>/app/scripts/serve_twin_mcp.py --curated`. A curated catalog with approved artifacts must be populated before that server can return twins; the empty source build supplies none.

See [release scope and attribution](PUBLIC_RELEASE.md) and the [technical overview](technical-overview-20261104.md) for the architecture and current limits.
