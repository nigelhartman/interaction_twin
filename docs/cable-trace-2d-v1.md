# Optional visible cable tracing

`cable_trace_2d.py` adds a private, source-bound **2D** companion to an evidence
program. It reads an explicitly selected saved cable track and its actual SAM
binary masks. It neither changes that program nor runs a model.

```sh
.venv-sam/bin/python scripts/build_cable_trace_2d.py \
  --program data/<run>/<procedure>/interaction.json \
  --track-id <exact_cable_track_id> \
  --max-frames 90 --max-long-edge 960 \
  --output-dir output/semantic-review/<run>
```

The output pins the source video, exact program, source frame, binary mask,
source frame index and original presentation timestamp. It keeps all original
masks unchanged. At most 90 temporally distributed samples are selected; exact
`--evidence-id` selections are also supported.

Each eligible mask is skeletonized. Only short terminal spurs are pruned; the
longest tree-geodesic path is exact for a tree, and a declared bounded geodesic
approximation is used when the projected mask contains cycles. Disconnected
components are never joined. The pruning threshold is in working-mask pixels,
not physical length. If a mask must be reduced to 960 pixels, each
retained edge is checked against **every original foreground pixel** on its
source-space segment. A reduction may omit a thin visible span and is flagged.

`points_px` are ordered `[x,y]` original mask pixel coordinates with a top-left
origin. Frame and mask dimensions must match. The endpoints are ordered
lexicographically for deterministic output; they are **not identified cable
plugs**, and no physical endpoint identity persists between samples. Branches,
loops, disconnected spans, border truncation, insufficient masks and withheld
SAM observations remain explicit. No missing spans or intermediate time samples
are filled.

This is useful evidence for a later cable spline/rod implementation. It does
not produce depth, 3D cable geometry, metric length, material parameters,
electrical compatibility or physics. A static SAM 3D cable appearance asset is
separate from this visible curve evidence. The semantic companion's deferred
3D `cable_centerline` remains null; this output can be linked by an exact
reference in a separate private review packet.

Dependencies are the already installed NumPy, SciPy, scikit-image and Pillow.
No packages are installed by the module or CLI. Complexity and image bounds
fail closed; oversized paths are withheld rather than simplified across
unobserved regions. Content-addressed writes are immutable, and `load()`
rebuilds the deterministic result from exact original inputs to reject altered
paths, timestamps, masks or claims.
