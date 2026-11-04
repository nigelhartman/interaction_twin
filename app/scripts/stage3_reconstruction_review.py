#!/usr/bin/env python3
"""Offline source/asset review for actual model output, without proxy geometry.

Accepts SAM3D adapter metadata or a small explicit manifest with ``source.image_path``,
``source.mask_path`` and ``outputs.glb_path``. Paths are relative to the manifest.
An optional source-camera view requires ``layout.camera_from_glb`` (4x4, CV camera
axes), ``layout.intrinsics`` (pixel K), ``layout.image_size`` ([width,height]), and
``layout.convention_verified: true``. Native model layout is never guessed.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import trimesh

ROOT = Path(__file__).resolve().parents[1]


def image_uri(image: Image.Image) -> str:
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def factor_rgba(value, default=(1, 1, 1, 1)) -> list:
    if value is None:
        return list(default)
    value = np.asarray(value, dtype=float).reshape(-1)
    if len(value) == 3:
        value = np.r_[value, 255 if value.max(initial=0) > 1 else 1]
    if len(value) != 4 or not np.isfinite(value).all():
        raise ValueError("Material colour must have three or four finite channels")
    if value.max() > 1:
        value = value / 255
    return np.clip(value, 0, 1).tolist()


def mesh_primitives(path: Path) -> tuple[list, dict]:
    """Preserve GLB scene nodes, separate materials, seam vertices and UVs."""
    scene = trimesh.load(path, force="scene", process=False)
    primitives, modes = [], set()
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        mesh = scene.geometry[name]
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            continue
        vertices = trimesh.transform_points(mesh.vertices, transform)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("Only triangulated model meshes are supported")
        if not np.isfinite(vertices).all():
            raise ValueError("Nonfinite model vertices")
        visual = mesh.visual
        material = getattr(visual, "material", None)
        materials = getattr(material, "materials", None)
        materials = list(materials) if materials is not None else [material]
        face_materials = getattr(visual, "face_materials", None)
        face_materials = np.zeros(len(faces), dtype=int) if face_materials is None else np.asarray(face_materials)
        if face_materials.shape != (len(faces),):
            raise ValueError("Invalid per-face material indices")
        uv_all = getattr(visual, "uv", None)
        if uv_all is not None:
            uv_all = np.asarray(uv_all, dtype=float)
            if uv_all.shape != (len(vertices), 2) or not np.isfinite(uv_all).all():
                raise ValueError("Invalid vertex UV coordinates")
        for material_index in np.unique(face_materials):
            if material_index < 0 or material_index >= len(materials):
                raise ValueError("Out-of-range GLB material index")
            selected = faces[face_materials == material_index]
            used, inverse = np.unique(selected.reshape(-1), return_inverse=True)
            mat = materials[material_index]
            texture = getattr(mat, "baseColorTexture", None)
            if texture is None:
                texture = getattr(mat, "image", None)
            factor = factor_rgba(getattr(mat, "baseColorFactor", getattr(mat, "diffuse", None)))
            colors = None
            if visual.kind == "vertex":
                colors = np.asarray(visual.vertex_colors, dtype=float)[used] / 255
                if colors.shape[1] == 3:
                    colors = np.column_stack([colors, np.ones(len(colors))])
            elif visual.kind == "face":
                # Face colours require independent corners; do not average them.
                face_colors = np.asarray(visual.face_colors, dtype=float)[face_materials == material_index] / 255
                expanded = selected.reshape(-1)
                used = expanded
                inverse = np.arange(len(expanded))
                colors = np.repeat(face_colors, 3, axis=0)
            mode = "uv_texture" if texture is not None and uv_all is not None else "vertex_rgb" if colors is not None else "material_color"
            modes.add(mode)
            primitive = {
                "node": str(node), "geometry": str(name), "material_index": int(material_index),
                "vertices": vertices[used].tolist(), "faces": inverse.reshape(-1, 3).tolist(),
                "base_color_factor": factor, "appearance_mode": mode,
                "colors": colors.tolist() if colors is not None else None,
                "uv": uv_all[used].tolist() if uv_all is not None else None,
                "texture": image_uri(texture.convert("RGBA")) if mode == "uv_texture" else None,
            }
            primitives.append(primitive)
    if not primitives:
        raise ValueError("The output file contains no supported mesh geometry")
    all_vertices = np.concatenate([np.asarray(p["vertices"]) for p in primitives])
    return primitives, {
        "bounds": [all_vertices.min(0).tolist(), all_vertices.max(0).tolist()],
        "primitive_count": len(primitives), "appearance_modes": sorted(modes),
        "rendered_vertices": sum(len(p["vertices"]) for p in primitives),
        "faces": sum(len(p["faces"]) for p in primitives),
        "load_policy": "trimesh process=False; all geometry nodes/materials preserved; no vertex merge",
        "rendering_policy": "unlit base-colour/vertex RGB review; PBR lighting, normal maps and metallic/roughness maps are not simulated",
    }


def resolve_file(base: Path, value) -> Path | None:
    if isinstance(value, dict):
        value = value.get("path", value.get("file"))
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def manifest_paths(path: Path, manifest: dict) -> tuple[Path, Path | None, Path]:
    base = path.parent
    inputs = manifest.get("inputs", {})
    source = manifest.get("source", {})
    outputs = manifest.get("outputs", manifest.get("output", {}))
    image = resolve_file(base, source.get("image_path", manifest.get("source_image")))
    mask = resolve_file(base, source.get("mask_path", manifest.get("source_mask")))
    if image is None and (base / "inputs/rgb.png").is_file():
        image = base / "inputs/rgb.png"
    if mask is None and (base / "inputs/mask.png").is_file():
        mask = base / "inputs/mask.png"
    files = inputs.get("files", {})
    image = image or resolve_file(base, files.get("image"))
    mask = mask or resolve_file(base, files.get("mask"))
    asset = resolve_file(base, outputs.get("glb_path", manifest.get("asset_path")))
    if asset is None:
        for record in outputs.get("files", []):
            candidate = resolve_file(base, record)
            if candidate is not None and candidate.suffix.lower() == ".glb":
                asset = candidate
                break
    if asset is None and (base / "object.glb").is_file():
        asset = base / "object.glb"
    if image is None or not image.is_file():
        raise ValueError("Manifest has no readable source image")
    if asset is None or not asset.is_file():
        raise ValueError("No actual reconstructed GLB exists; review generation is disabled")
    if mask is not None and not mask.is_file():
        raise ValueError("Referenced segmentation mask is missing")
    return image, mask, asset


def verified_layout(manifest: dict, image_size: tuple[int, int]) -> dict | None:
    layout = manifest.get("layout")
    if not isinstance(layout, dict) or layout.get("convention_verified") is not True:
        return None
    matrix = np.asarray(layout.get("camera_from_glb"), dtype=float)
    intrinsics = np.asarray(layout.get("intrinsics"), dtype=float)
    if matrix.shape != (4, 4) or intrinsics.shape != (3, 3) or not np.isfinite(matrix).all() or not np.isfinite(intrinsics).all():
        raise ValueError("Verified layout needs finite camera_from_glb 4x4 and intrinsics 3x3")
    if not np.allclose(matrix[3], [0, 0, 0, 1]) or min(intrinsics[0, 0], intrinsics[1, 1]) <= 0:
        raise ValueError("Invalid camera layout")
    if list(layout.get("image_size", [])) != list(image_size):
        raise ValueError("Layout image size differs from the source image")
    if layout.get("camera_axes") != "+X right, +Y down, +Z forward":
        raise ValueError("Explicit CV camera_axes are required")
    return {"camera_from_glb": matrix.tolist(), "intrinsics": intrinsics.tolist(), "image_size": list(image_size), "camera_axes": layout["camera_axes"]}


def display_convention(manifest: dict, backend: str) -> dict:
    """Select orbit-camera axes without altering asset geometry or source pose."""
    requested = manifest.get("display", {}).get("up_axis")
    if requested is None:
        # Official TripoSR get_spherical_cameras: x back, y right, z up.
        # SAM3D's GLB exporter explicitly converts canonical z-up to y-up.
        axis = "+Z" if backend.strip().lower() == "triposr" else "+Y"
        policy = "documented TripoSR native coordinates" if axis == "+Z" else "GLB y-up default"
    else:
        text = str(requested).strip().upper()
        axis = text if text.startswith(("+", "-")) else "+" + text
        policy = "explicit display.up_axis metadata"
    if axis not in {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}:
        raise ValueError("display.up_axis must be X, Y or Z, optionally prefixed with + or -")
    up = np.zeros(3)
    up["XYZ".index(axis[-1])] = -1 if axis.startswith("-") else 1
    forward = np.array([1., 0., 0.]) if abs(up[2]) > .9 else np.array([0., 0., 1.])
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    forward = np.cross(right, up)
    return {"up_axis": axis, "up": up.tolist(), "forward": forward.tolist(),
            "right": right.tolist(), "policy": policy,
            "geometry_modified": False, "source_pose_implied": False}


def cpu_asset_preview(primitives: list, bounds: list, size=(440, 440), display=None) -> Image.Image:
    """Unlit z-buffer preview using the actual UV/vertex data, for file QA."""
    center = np.mean(bounds, axis=0)
    extent = max(np.ptp(np.asarray(bounds), axis=0)) or 1
    yaw, pitch = .55, .18
    display = display or display_convention({}, "")
    display_up = np.asarray(display["up"])
    direction = (np.asarray(display["right"])*np.sin(yaw)*np.cos(pitch)
                 + display_up*np.sin(pitch)
                 + np.asarray(display["forward"])*np.cos(yaw)*np.cos(pitch))
    eye = center + direction * extent * 2.05
    right = np.cross(display_up, direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    basis = np.stack([right, up, direction], axis=1)
    width, height = size
    focal = height / (2*np.tan(np.deg2rad(42)/2))
    pixels = np.full((height, width, 3), [14, 22, 33], dtype=np.uint8)
    zbuffer = np.full((height, width), np.inf)
    for primitive in primitives:
        points = (np.asarray(primitive["vertices"]) - eye) @ basis
        depth = -points[:, 2]
        projected = np.column_stack([focal*points[:, 0]/depth+width/2, -focal*points[:, 1]/depth+height/2])
        colors = np.asarray(primitive["colors"]) if primitive["colors"] else None
        uv = np.asarray(primitive["uv"]) if primitive["uv"] else None
        texture = None
        if primitive["texture"]:
            texture = np.asarray(Image.open(io.BytesIO(base64.b64decode(primitive["texture"].split(",", 1)[1]))).convert("RGB")) / 255
        factor = np.asarray(primitive["base_color_factor"][:3])
        for face in np.asarray(primitive["faces"], dtype=int):
            if (depth[face] <= 0).any():
                continue
            triangle = projected[face]
            x0, y0 = np.maximum(np.floor(triangle.min(0)).astype(int), [0, 0])
            x1, y1 = np.minimum(np.ceil(triangle.max(0)).astype(int), [width-1, height-1])
            if x1 < x0 or y1 < y0:
                continue
            a, b, c = triangle
            denominator = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
            if abs(denominator) < 1e-10:
                continue
            xx, yy = np.meshgrid(np.arange(x0, x1+1)+.5, np.arange(y0, y1+1)+.5)
            wa = ((b[1]-c[1])*(xx-c[0]) + (c[0]-b[0])*(yy-c[1])) / denominator
            wb = ((c[1]-a[1])*(xx-c[0]) + (a[0]-c[0])*(yy-c[1])) / denominator
            wc = 1-wa-wb
            bary = np.stack([wa, wb, wc], axis=-1)
            weights = bary / depth[face]
            inv_depth = weights.sum(-1)
            z = np.divide(1, inv_depth, out=np.full_like(inv_depth, np.inf), where=inv_depth > 0)
            weights = np.divide(weights, inv_depth[..., None], out=np.zeros_like(weights), where=inv_depth[..., None] > 0)
            region_z = zbuffer[y0:y1+1, x0:x1+1]
            visible = (bary >= -1e-6).all(-1) & (z < region_z)
            if not visible.any():
                continue
            if texture is not None and uv is not None:
                tex_uv = weights @ uv[face]
                tx = np.clip(np.rint(tex_uv[..., 0]*(texture.shape[1]-1)).astype(int), 0, texture.shape[1]-1)
                ty = np.clip(np.rint((1-tex_uv[..., 1])*(texture.shape[0]-1)).astype(int), 0, texture.shape[0]-1)
                rgb = texture[ty, tx] * factor
            elif colors is not None:
                rgb = (weights @ colors[face, :3]) * factor
            else:
                rgb = np.broadcast_to(factor, (*visible.shape, 3))
            pixels[y0:y1+1, x0:x1+1][visible] = np.clip(rgb[visible]*255, 0, 255).astype(np.uint8)
            region_z[visible] = z[visible]
    return Image.fromarray(pixels)


def comparison_preview(source: Image.Image, primitives: list, stats: dict, title: str, output: Path, display=None):
    image = Image.new("RGB", (920, 525), "#0b111b")
    draw = ImageDraw.Draw(image)
    draw.text((20, 15), title, fill="#edf3fa")
    draw.text((20, 43), "Source input", fill="#9eb0c6")
    draw.text((480, 43), "Actual generated asset / free viewpoint", fill="#9eb0c6")
    thumbnail = ImageOps.contain(source, (440, 440))
    image.paste(thumbnail, (20+(440-thumbnail.width)//2, 70+(440-thumbnail.height)//2))
    image.paste(cpu_asset_preview(primitives, stats["bounds"], display=display), (480, 70))
    image.save(output)


DOCUMENT = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Model reconstruction review</title><style>
:root{color-scheme:dark;--bg:#0b111b;--panel:#121d2c;--line:#293a50;--muted:#9eb0c6;--text:#edf3fa}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:28px}h1{font-size:27px;margin:0 0 4px}h2{font-size:17px;margin:0}p{margin:8px 0;color:var(--muted)}.badge{display:inline-block;border:1px solid #3d5470;padding:5px 10px;border-radius:7px;font-size:12px;margin:12px 6px 14px 0}.views{display:grid;grid-template-columns:1fr 1fr;gap:16px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}.head{padding:14px 17px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:12px}.viewport{height:480px;position:relative;background:#101925;overflow:hidden}.source-wrap{height:100%;display:flex;align-items:center;justify-content:center}.source-stack{position:relative;max-width:100%;max-height:100%;line-height:0}.source-stack img{display:block;max-width:100%;max-height:480px}.source-stack canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}#scene{width:100%;height:100%;display:block}.controls{display:flex;align-items:center;flex-wrap:wrap;gap:12px;padding:14px 16px;min-height:78px}button{color:var(--text);background:#203249;border:1px solid #3f5875;border-radius:7px;padding:7px 12px;cursor:pointer}button.active{background:#285d80}button:disabled{opacity:.45;cursor:default}label{white-space:nowrap;font-size:13px}input{accent-color:#56c8ed;vertical-align:middle}small{color:var(--muted);font-size:12px}.hint{position:absolute;bottom:12px;left:14px;color:#afc0d2;font-size:12px;pointer-events:none}.details{border:1px solid var(--line);border-radius:10px;padding:17px;margin-top:18px}.warning{color:#efc585}#error{position:absolute;top:20px;left:20px;right:20px;color:#ffbcaf;display:none;white-space:pre-wrap}#mode-label{color:var(--muted);font-size:12px}a{color:#82d4f4}@media(max-width:900px){.views{grid-template-columns:1fr}.viewport{height:420px}.source-stack img{max-height:420px}main{padding:16px}}
</style></head><body><main><h1 id="title"></h1><p>Original pixels next to the actual model output. Drag the model to inspect its hidden surfaces.</p><div id="badges"></div><div class="views"><section class="panel"><div class="head"><h2>Source frame</h2><small id="source-label"></small></div><div class="viewport source-wrap"><div class="source-stack"><img id="source" alt="Original input frame"><canvas id="mask-overlay"></canvas></div></div><div class="controls"><label><input id="mask-toggle" type="checkbox" checked>Visible-object mask</label><small>Segmentation selects pixels; it does not provide hidden geometry.</small></div></section><section class="panel"><div class="head"><h2>Reconstructed asset</h2><span id="mode-label">Orbit · asset coordinates</span></div><div class="viewport"><canvas id="scene" aria-label="Orbitable reconstructed model"></canvas><div class="hint">Drag to orbit · scroll to zoom · Shift + drag to pan</div><div id="error" role="alert"></div></div><div class="controls"><button id="orbit" class="active">Orbit</button><button id="match">Source camera</button><button id="reset">Reset</button><label><input id="texture-toggle" type="checkbox" checked>Model appearance</label><label><input id="wire-toggle" type="checkbox">Wireframe</label></div></section></div><section class="details"><strong id="alignment-label"></strong><p id="appearance-label"></p><p id="provenance-label"></p><p class="warning" id="status-label"></p><small>The raw GLB is preserved. Orbit framing changes only the display camera. A plausible single-image completion does not verify hidden geometry, physical size or contact.</small></section></main><script id="reconstruction-data" type="application/json">DATA_JSON</script><script>RUNTIME_JS</script></body></html>'''


def build_review(manifest_path: Path, output: Path) -> dict:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    image_path, mask_path, asset_path = manifest_paths(manifest_path, manifest)
    source = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L") if mask_path else None
    if mask and mask.size != source.size:
        raise ValueError("Source mask and image must have the same dimensions")
    primitives, stats = mesh_primitives(asset_path)
    layout = verified_layout(manifest, source.size)
    inputs_manifest = manifest.get("inputs", {}).get("manifest", {})
    source_metadata = manifest.get("source", inputs_manifest.get("source", {}))
    mask_provenance = manifest.get("mask_provenance", inputs_manifest.get("mask_provenance", source_metadata.get("mask_provenance", {})))
    appearance = manifest.get("appearance_provenance", "model_inferred")
    model = manifest.get("model", {})
    backend = manifest.get("backend") or (model.get("name") if isinstance(model, dict) else model) or "recorded model output"
    display = display_convention(manifest, backend)
    payload = {
        "object_id": manifest.get("object_id", inputs_manifest.get("object_id", "reconstructed_object")),
        "backend": backend,
        "status": manifest.get("status", "candidate_pending_review"),
        "validation_status": manifest.get("validation_status"),
        "source": image_uri(source), "mask": image_uri(mask) if mask else None,
        "image_size": list(source.size), "source_metadata": source_metadata,
        "mask_provenance": mask_provenance, "appearance_provenance": appearance,
        "primitives": primitives, "stats": stats, "layout": layout,
        "display": display,
        "source_sha256": file_hash(image_path), "asset_sha256": file_hash(asset_path),
    }
    runtime = (ROOT / "reconstruction_observer.js").read_text()
    document = DOCUMENT.replace("DATA_JSON", json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/"))
    document = document.replace("RUNTIME_JS", runtime)
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.html").write_text(document)
    comparison_preview(source, primitives, stats, f"{payload['object_id']} | {payload['backend']} | {payload['status']}", output / "preview.png", display=display)
    report = {"status": "review_generated", "html": str(output / "index.html"), "backend": payload["backend"],
              "model_status": payload["status"], "asset": str(asset_path), "asset_sha256": payload["asset_sha256"],
              "validation_status": payload["validation_status"],
              "display": display,
              "source_sha256": payload["source_sha256"], "stats": stats, "source_camera_available": layout is not None,
              "appearance_provenance": appearance, "external_network_dependencies": False,
              "geometry_substitution": False, "raw_model_file_modified": False,
              "cpu_preview": str(output / "preview.png")}
    (output / "review-metadata.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build_review(args.manifest, args.out), indent=2))


if __name__ == "__main__":
    main()
