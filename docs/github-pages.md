# Static demonstration

The public demo is a separate `gh-pages` branch, published by GitHub Pages from its root. The application source remains on `main`.

The exported site contains selected saved videos, geometry, descriptive artifacts, pinned browser dependencies and saved robot baselines. It does not require Python, API keys, a running GPU or the owner's laptop. Library filtering and rendering happen in the browser. Supported robot edits use the existing bounded local compiler.

`export-manifest.json` records the exact deployment inventory. Saved data bytes and their source hashes are unchanged. Only copied viewer URLs and the filtered public catalog change for the project URL `/interaction_twin/`. The local six-example library remains unchanged; the public five-example export excludes the third-party camera recording.

MCP itself runs locally from the source checkout. The website's agent guide explains that distinction. GitHub Pages does not host the app's HTTP API, uploads, processing or private admin.

The export helper `scripts/build_github_pages_20261104_v1.py` requires the corresponding full curated working snapshot and frozen robot assets. The code-only `main` branch deliberately omits those private inputs; it is not a standalone one-command regeneration of all saved demonstrations. The deployed branch can instead be downloaded and served by any ordinary static host.

To preview the deployed branch locally, use a separate clone and ordinary static server:

```sh
git clone --branch gh-pages --single-branch https://github.com/nigelhartman/interaction_twin.git demo-root/interaction_twin
python3 -m http.server 8000 --directory demo-root
# Open http://localhost:8000/interaction_twin/
```

No backend or model calls are involved. The final external URL must be checked after GitHub completes its deployment.
