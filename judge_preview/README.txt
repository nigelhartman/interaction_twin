INTERACTION TWIN — FROZEN LOCAL JUDGE PREVIEW

Start
  1. Extract the ZIP completely.
  2. Open a terminal in the extracted folder.
  3. Run: python3 preview_server.py
  4. Open the localhost address printed in the terminal.

Python 3.10 or newer is sufficient. The server uses only the standard library.
No installation, API key, GPU, login, cloud service or internet connection is
needed. Keep the terminal open while inspecting the preview; Ctrl-C stops it.
If port8767 is busy, run: python3 preview_server.py --port 0
The isolated standard-library-only launch also works:
  python3 -I -S preview_server.py --port 0

What is included
  - The frozen current curated Project, Library, Robotics lab and agent pages.
  - The current Twin frame brand and 45-second film with English captions.
  - A saved cross-recording phone/charger agent demonstration, with its trace,
    exact source links and retained Muse assessment. Browsing does not infer.
  - Exact approved source videos, evidence/masks, native assets, descriptive
    programs and the saved constrained replay needed by these pages.
  - A small read-only JSON search API with the same public filters and fields:
      GET /api/library
      GET /api/library/facets
      GET /api/library?parts=case_base&backends=sam3d
  - Local render code and the vendored Three.js dependency with its license.
  - Supported Jabra and wallet robot task snapshots and downloadable standalone
    browser starter ZIPs. Subsequent supported instructions compile locally in
    the browser; there is no live server planner, model call or physical robot.
  - The Newton physics bench replays 241 saved states from an actual CPU contact
    experiment with three exact approved native meshes. Its runnable starter ZIP
    includes the meshes, assumptions and a separate Newton runner. Watching the
    replay needs no installation; rerunning Newton requires the starter's stated
    Python version and optional dependencies. This preview never launches it.
  - Declared source-supported geometry/appearance derivatives and 2D-only cable
    observations accompany their original native assets and source evidence.
  - Wallet, phone and camera source-visible surfaces retain exact source RGB,
    saved segmentation and estimated depth. Contour refinement improves visible
    borders; hidden surfaces and hand-covered regions are not completed.
  - A separate laptop source-conditioned motion hypothesis fills checked gaps
    when enabled. Original saved observations, unknown rows and view-change cuts
    remain unchanged; these estimates are not independent physical ground truth.
  - bundle-manifest.json records every included file's SHA-256 and byte length.
    The server verifies the frozen files at startup and before serving them.

What is excluded
  - Owner administration, uploads, feedback memory, pending/working revisions,
    model weights, inference code, API credentials, SSH and GPU configuration.
  - An MCP server/runtime. The optional MCP directions on the project’s agent
    page apply ONLY to a separate full project checkout with its runtime already
    installed. Use this bundle's read-only JSON API for standalone agent access.
  - Approved historical implementation-provenance Python source-text files are
    intentionally omitted under a bounded declared-path policy. The exact list
    and hashes appear in bundle-manifest.json; no fixed count is assumed.
    Original provenance pointers remain unchanged and are not download grants.

Preview scope
  - Generated geometry, metric scale, real-world contact and robot grasps remain
    unverified. The Robotics lab's robot is an authored kinematic rehearsal.
    Newton contact is a separate actual engine run with assumed parameters and
    convex proxies, not certification of those reconstructed objects.
  - Only the included ready task baselines are available in the workbench.
    Other twins can still be inspected in the library and recorded-motion view;
    missing task prerequisites stop the workbench explicitly.
  - Some source/procedure JSON contains exact hashes and paths for historical
    internal compiler inputs. Those are provenance pointers, not downloadable
    files in this bundle. The bundle does not copy private internals to resolve
    them. The local server exposes only its exact frozen allowlist.
  - Approved legacy review HTML is included as original bytes for provenance,
    but is served as a sandboxed attachment, not as an executable viewer. The
    current Project/Library use the included modern native-model viewer instead.
  - Nothing is deployed or submitted by this bundle. The server binds only to
    127.0.0.1 and has no write or execution endpoints.

Designed and built by Nigel Hartman.
