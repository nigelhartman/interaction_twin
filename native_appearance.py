"""Bounded native GLB base-colour loading and CPU projection, without UV baking.

The original GLB is never rewritten.  UVs, material associations and scene-node
transforms remain native.  The diagnostic pass is unlit: it does not claim to
reproduce a PBR-lit photograph, reconstruct hidden geometry, or verify physics.
Only opaque static triangles with embedded PNG/JPEG base-colour images and
supported explicit sampler semantics are accepted. Unsupported semantics fail.

For compatible vertex-colour-only GLBs, the historical Trimesh loader and CPU
renderer are deliberately retained byte/pixel-for-pixel. Other inputs use glTF
linear base colour: sRGB texture decode, linear tint/factor/filtering, sRGB output.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import struct
import warnings

import numpy as np
from PIL import Image

from alignment_render import _inputs, _clip_near, render_mesh, raster_triangle_candidates
from portable_twin import MAX_ASSET, MAX_IMAGE, _json, decode_glb_bytes

CONTRACT_VERSION = 'native-base-color-v1'
MAX_IMAGE_PIXELS = 16_777_216
MAX_TOTAL_IMAGE_PIXELS = 16_777_216
MAX_RENDER_PIXELS = 16_777_216


@dataclass(frozen=True)
class NativePrimitive:
    vertices: np.ndarray
    faces: np.ndarray
    uv: np.ndarray | None
    tint: np.ndarray
    factor: np.ndarray
    texture_linear: np.ndarray | None
    sampler: dict
    face_offset: int
    double_sided: bool
    winding_sign: int


@dataclass(frozen=True)
class NativeAppearance:
    primitives: tuple[NativePrimitive, ...]
    legacy_vertex_compatibility: bool


@dataclass(frozen=True)
class NativeMesh:
    vertices: np.ndarray
    faces: np.ndarray
    vertex_colors: np.ndarray | None
    appearance: NativeAppearance
    receipt: dict


def _reject_extensions(value):
    if isinstance(value, dict):
        extensions = value.get('extensions', {})
        if not isinstance(extensions, dict):
            raise ValueError('Invalid GLB extensions declaration')
        if set(extensions) - {'KHR_materials_unlit'}:
            raise ValueError('Unsupported GLB extension semantics')
        if 'KHR_materials_unlit' in extensions and extensions['KHR_materials_unlit'] != {}:
            raise ValueError('Invalid KHR_materials_unlit declaration')
        for child in value.values():
            _reject_extensions(child)
    elif isinstance(value, list):
        for child in value:
            _reject_extensions(child)


def _document(raw):
    # The portable decoder is the structural, allocation and embedded-data guard.
    decoded = decode_glb_bytes(raw)
    offset, document = 12, None
    while offset < len(raw):
        length, kind = struct.unpack_from('<II', raw, offset)
        offset += 8
        if kind == 0x4E4F534A:
            document = _json(raw[offset:offset + length].rstrip(b' \x00'), 4 * 1024 * 1024)
        offset += length
    if document is None:
        raise ValueError('Missing GLB JSON')
    for key in ('extensionsUsed', 'extensionsRequired'):
        if not isinstance(document.get(key, []), list) or set(document.get(key, [])) - {'KHR_materials_unlit'}:
            raise ValueError('Unsupported GLB extension semantics')
    _reject_extensions(document)
    if len(document.get('images', [])) > 64 or len(document.get('textures', [])) > 64 or len(document.get('samplers', [])) > 64:
        raise ValueError('GLB appearance budget exceeded')
    # Reject semantics even on unused materials: an apparently harmless default
    # scene must not obscure unsupported/external appearance elsewhere in a file.
    for material in document.get('materials', []):
        if material.get('alphaMode', 'OPAQUE') != 'OPAQUE':
            raise ValueError('Only opaque GLB materials are supported')
        if 'alphaCutoff' in material:
            raise ValueError('Alpha-cutoff semantics are unsupported')
        if any(key in material for key in ('normalTexture', 'occlusionTexture', 'emissiveTexture')):
            raise ValueError('Only base-colour material textures are supported')
        if material.get('emissiveFactor', [0, 0, 0]) != [0, 0, 0]:
            raise ValueError('Emissive material semantics are unsupported')
        pbr = material.get('pbrMetallicRoughness', {})
        if 'metallicRoughnessTexture' in pbr:
            raise ValueError('Metallic/roughness texture semantics are unsupported')
        factor = pbr.get('baseColorFactor', [1, 1, 1, 1])
        if len(factor) != 4 or factor[3] != 1:
            raise ValueError('Translucent base-colour factors are unsupported')
        if 'doubleSided' in material and type(material['doubleSided']) is not bool:
            raise ValueError('Invalid doubleSided material flag')
        for key in ('metallicFactor', 'roughnessFactor'):
            value = pbr.get(key, 1)
            if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid PBR material factor')
    for sampler in document.get('samplers', []):
        _sampler(sampler)
    # Decode validates arbitrary accessor encodings structurally; this pass
    # additionally limits attributes to glTF base colour and position semantics.
    for mesh in document.get('meshes', []):
        if mesh.get('weights'):
            raise ValueError('Morph weight semantics are unsupported')
        for primitive in mesh.get('primitives', []):
            attrs = primitive.get('attributes', {})
            if set(attrs) - {'POSITION', 'NORMAL', 'TEXCOORD_0', 'COLOR_0'}:
                raise ValueError('Unsupported GLB vertex attribute semantics')
            if 'indices' in primitive:
                index_accessor = document['accessors'][primitive['indices']]
                if index_accessor.get('type') != 'SCALAR' or index_accessor.get('componentType') not in (5121, 5123, 5125) or index_accessor.get('normalized', False) is not False:
                    raise ValueError('Triangle indices must be unnormalized unsigned scalar accessors')
            for semantic in ('POSITION', 'NORMAL', 'TEXCOORD_0', 'COLOR_0'):
                if semantic not in attrs:
                    continue
                a = document['accessors'][attrs[semantic]]
                component, normalized = a['componentType'], a.get('normalized', False)
                expected = {'POSITION': ('VEC3',), 'NORMAL': ('VEC3',), 'TEXCOORD_0': ('VEC2',), 'COLOR_0': ('VEC3', 'VEC4')}[semantic]
                if a.get('type') not in expected or type(normalized) is not bool:
                    raise ValueError('Invalid glTF vertex attribute shape/normalization')
                allowed = component == 5126 if semantic in ('POSITION', 'NORMAL') else (component == 5126 or component in (5121, 5123) and normalized is True)
                if not allowed:
                    raise ValueError('Invalid glTF base-colour attribute encoding')
    # Recover native per-instance material/culling semantics in exactly the
    # selected-scene traversal order used by the structural decoder.
    instances = []
    nodes = document.get('nodes', [])
    def visit(index, parent_sign):
        node = nodes[index]
        determinant = (np.linalg.det(np.asarray(node['matrix'], float).reshape(4, 4).T[:3, :3])
                       if 'matrix' in node else np.prod(node.get('scale', [1, 1, 1])))
        if determinant == 0:
            raise ValueError('Singular scene-node transforms are unsupported')
        sign = parent_sign * (1 if determinant > 0 else -1)
        if 'mesh' in node:
            for primitive in document['meshes'][node['mesh']]['primitives']:
                material = document.get('materials', [])[primitive['material']] if 'material' in primitive else {}
                instances.append((material.get('doubleSided', False), sign))
        for child in node.get('children', []):
            visit(child, sign)
    scenes = document.get('scenes', [])
    roots = (scenes[document.get('scene', 0)].get('nodes', []) if scenes else
             [i for i in range(len(nodes)) if not any(i in node.get('children', []) for node in nodes)])
    for root in roots:
        visit(root, 1)
    if len(instances) != len(decoded['primitives']):
        raise ValueError('Native primitive traversal differs from structural decoder')
    return document, decoded, instances


def _sampler(value):
    mag = value.get('magFilter', 9729)
    minification = value.get('minFilter', mag)
    if mag not in (9728, 9729) or minification not in (9728, 9729) or mag != minification:
        raise ValueError('Only equal nearest/linear min/mag filters without mipmaps are supported')
    if value.get('wrapS', 10497) not in (10497, 33071, 33648) or value.get('wrapT', 10497) not in (10497, 33071, 33648):
        raise ValueError('Unsupported GLB texture wrap mode')
    return {'filter': 'nearest' if mag == 9728 else 'linear',
            'wrap_s': value.get('wrapS', 10497), 'wrap_t': value.get('wrapT', 10497),
            'mipmaps': False, 'unspecified_filter_policy': 'mag_linear; min_matches_mag; no_mipmap'}


def _srgb_to_linear(value):
    value = np.asarray(value, dtype=np.float64)
    return np.where(value <= .04045, value / 12.92, ((value + .055) / 1.055) ** 2.4)


def _linear_to_srgb(value):
    value = np.clip(np.asarray(value, dtype=np.float64), 0, 1)
    return np.where(value <= .0031308, value * 12.92, 1.055 * value ** (1 / 2.4) - .055)


def _image(uri):
    # URI came from the bounded embedded-data decoder, never a resolver.
    encoded = uri.split(',', 1)[1]
    raw = base64.b64decode(encoded, validate=True)
    if not 0 < len(raw) <= MAX_IMAGE:
        raise ValueError('Embedded image byte budget exceeded')
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        try:
            with Image.open(io.BytesIO(raw)) as image:
                if image.format not in ('PNG', 'JPEG') or image.width * image.height > MAX_IMAGE_PIXELS or min(image.size) < 1:
                    raise ValueError('Embedded image pixel budget exceeded')
                if getattr(image, 'n_frames', 1) != 1:
                    raise ValueError('Animated embedded images are unsupported')
                # glTF colour images are sRGB. Embedded colour profiles/EXIF do
                # not silently change or rotate native texture coordinates.
                rgba = np.asarray(image.convert('RGBA'))
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
            raise ValueError('Embedded image pixel budget exceeded') from error
    if np.any(rgba[:, :, 3] != 255):
        raise ValueError('Only fully opaque embedded images are supported')
    return _srgb_to_linear(rgba[:, :, :3].astype(float) / 255).astype(np.float32)


def load_native_glb(source: bytes | Path) -> NativeMesh:
    """Load native embedded appearance, never replacing UVs with vertex colours.

    Path reads are explicit caller-selected local files, bounded to16MiB. All
    resource references inside the GLB remain embedded and resolver-free.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.stat().st_size > MAX_ASSET:
            raise ValueError('GLB exceeds asset byte budget')
        raw = path.read_bytes()
    elif isinstance(source, bytes):
        raw = source
    else:
        raise ValueError('Expected native GLB bytes or a local Path')
    document, decoded, instances = _document(raw)
    parts, vertices, faces, textures = [], [], [], {}
    v_offset, f_offset, total_pixels = 0, 0, 0
    for p, (double_sided, winding_sign) in zip(decoded['primitives'], instances):
        v, f = np.asarray(p['vertices'], float), np.asarray(p['faces'], np.int64)
        color = np.ones((len(v), 4)) if p['colors'] is None else np.asarray(p['colors'], float)
        if not np.isfinite(color).all() or np.any(color < 0) or np.any(color > 1) or np.any(color[:, 3] != 1):
            raise ValueError('Only finite opaque glTF vertex tints in0..1 are supported')
        uv = None if p['uv'] is None else np.asarray(p['uv'], float)
        texture, sampler = None, _sampler({})
        if p['texture']:
            key = hashlib.sha256(p['texture'].encode()).hexdigest()
            if key not in textures:
                textures[key] = _image(p['texture'])
                total_pixels += int(np.prod(textures[key].shape[:2]))
                if total_pixels > MAX_TOTAL_IMAGE_PIXELS:
                    raise ValueError('Total decoded texture pixel budget exceeded')
            texture = textures[key]
            material = document['materials'][p['material_index']]
            texture_index = material['pbrMetallicRoughness']['baseColorTexture']['index']
            texture_def = document['textures'][texture_index]
            if 'sampler' in texture_def:
                si = texture_def['sampler']
                samplers = document.get('samplers', [])
                if type(si) is not int or not 0 <= si < len(samplers):
                    raise ValueError('Invalid texture sampler index')
                sampler = _sampler(samplers[si])
        parts.append(NativePrimitive(v, f, uv, color[:, :3], np.asarray(p['base_color_factor'][:3], float), texture, sampler, f_offset, double_sided, winding_sign))
        vertices.append(v); faces.append(f + v_offset)
        v_offset += len(v); f_offset += len(f)
    vertices, faces = np.concatenate(vertices), np.concatenate(faces)
    modes = sorted({p['appearance_mode'] for p in decoded['primitives']})
    legacy = modes == ['vertex_rgb'] and all(np.array_equal(p.factor, np.ones(3)) for p in parts)
    vertex_colors = None
    if legacy:
        # Exact historical path for existing learned vertex-colour models. The
        # strict native parser above prevents TextureVisuals conversion fallback.
        import trimesh
        old = trimesh.load(io.BytesIO(raw), file_type='glb', force='mesh', process=False)
        if getattr(old.visual, 'kind', None) != 'vertex':
            raise ValueError('Legacy vertex-colour loader is incompatible')
        vertices, faces = np.asarray(old.vertices), np.asarray(old.faces)
        vertex_colors = np.asarray(old.visual.vertex_colors)[:, :3]
        if vertex_colors.shape != (len(vertices), 3):
            raise ValueError('Legacy loader has no usable original vertex colours')
    receipt = {
        'contract_version': CONTRACT_VERSION, 'source_glb_sha256': hashlib.sha256(raw).hexdigest(),
        'source_glb_bytes': len(raw), 'original_glb_unchanged': True,
        'renderer': 'legacy_alignment_render' if legacy else 'native_unlit_base_color_cpu',
        'legacy_vertex_compatibility': legacy, 'legacy_compatible': legacy, 'appearance_modes': modes,
        'primitive_count': len(parts), 'material_count': len(document.get('materials', [])),
        'rendered_vertices': len(vertices), 'faces': len(faces), 'texture_count': len(textures),
        'texture_pixels': total_pixels, 'uv_preserved': any(p.uv is not None for p in parts),
        'uv_to_vertex_color_conversion': False, 'scene_node_transforms_applied': True,
        'uv_origin': 'gltf_top_left; image_rows_preserved',
        'color_policy': ('historical_uint8_vertex_rgb_interpolation_unchanged' if legacy else
                         'srgb_texture_to_linear; linear_vertex_tint_times_base_color_factor; linear_filter; srgb_output'),
        'samplers': [p.sampler for p in parts if p.texture_linear is not None],
        'face_policy': ('historical_two_sided_diagnostic_projection_unchanged' if legacy else 'gltf_doubleSided_or_front_facing; negative_node_determinant_winding_corrected'),
        'primitive_culling': [{'double_sided': p.double_sided, 'node_winding_sign': p.winding_sign} for p in parts],
        'vertices_sha256': hashlib.sha256(np.asarray(vertices, dtype='<f8').tobytes()).hexdigest(),
        'faces_sha256': hashlib.sha256(np.asarray(faces, dtype='<i8').tobytes()).hexdigest(),
        'material_policy': 'opaque_unlit_base_color_only; no_lighting_or_PBR_shading',
        'metric_scale_verified': False, 'physical_accuracy_verified': False,
    }
    return NativeMesh(vertices, faces, vertex_colors, NativeAppearance(tuple(parts), legacy), receipt)


def alignment_contract(raw: bytes, declared_mode: str) -> dict:
    """Return a fail-closed JSON eligibility receipt, without changing raw bytes.

    Declared provider modes are ``vertex-color`` and ``uv-baked``; native names
    are accepted for local diagnostic calls. A provider declaration must agree
    with actual native appearance. No authentication/genuine-inference claim.
    """
    try:
        mesh = load_native_glb(raw)
        modes = mesh.receipt['appearance_modes']
        if declared_mode in ('vertex-color', 'vertex_rgb') and (modes != ['vertex_rgb']):
            raise ValueError('Declared vertex colour differs from native appearance')
        if declared_mode in ('uv-baked', 'uv_texture') and 'uv_texture' not in modes:
            raise ValueError('Declared UV appearance has no native texture')
        if declared_mode == 'material_color' and modes != ['material_color']:
            raise ValueError('Declared material colour differs from native appearance')
        if declared_mode not in ('vertex-color', 'vertex_rgb', 'uv-baked', 'uv_texture', 'material_color', 'mixed', 'native_glb'):
            raise ValueError('Unsupported declared appearance mode')
        return {'eligible': True, 'reason': None, 'receipt': mesh.receipt}
    except (ValueError, TypeError, KeyError, IndexError, OSError) as error:
        return {'eligible': False, 'reason': str(error), 'receipt': {
            'contract_version': CONTRACT_VERSION,
            'source_glb_sha256': hashlib.sha256(raw).hexdigest() if isinstance(raw, bytes) else None,
            'original_glb_unchanged': True, 'physical_accuracy_verified': False}}


def _wrap_index(index, size, mode):
    if mode == 10497:
        return index % size
    if mode == 33071:
        return np.clip(index, 0, size - 1)
    reflected = index % (2 * size)
    return np.where(reflected < size, reflected, 2 * size - 1 - reflected)


def _sample_texture(image, uv, sampler):
    height, width = image.shape[:2]
    # Repeat folding limits huge but finite UV coordinates before integer casts.
    coordinates = np.array(uv, dtype=float, copy=True)
    for dim, mode in enumerate((sampler['wrap_s'], sampler['wrap_t'])):
        if mode == 10497:
            coordinates[:, dim] %= 1
        elif mode == 33648:
            coordinates[:, dim] %= 2
        else:
            coordinates[:, dim] = np.clip(coordinates[:, dim], 0, 1)
    x, y = coordinates[:, 0] * width, coordinates[:, 1] * height
    if sampler['filter'] == 'nearest':
        xi = _wrap_index(np.floor(x).astype(np.int64), width, sampler['wrap_s'])
        yi = _wrap_index(np.floor(y).astype(np.int64), height, sampler['wrap_t'])
        return image[yi, xi]
    x, y = x - .5, y - .5
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    xa, xb = _wrap_index(x0, width, sampler['wrap_s']), _wrap_index(x0 + 1, width, sampler['wrap_s'])
    ya, yb = _wrap_index(y0, height, sampler['wrap_t']), _wrap_index(y0 + 1, height, sampler['wrap_t'])
    return (image[ya, xa] * (1 - fx) + image[ya, xb] * fx) * (1 - fy) + (image[yb, xa] * (1 - fx) + image[yb, xb] * fx) * fy


def _render_native_mesh(mesh: NativeMesh, K, R, t, width, height, scale=1.0, *, include_rgb=True):
    """Project a NativeMesh with a common z-buffer and perspective-correct UVs.

    Return the historical mask/depth/rgb/face_index contract. Sample centres,
    camera axes, near plane, face index ordering and clipping match the old pass.
    Native primitives retain their face/material/UV-seam associations throughout.
    """
    if not isinstance(mesh, NativeMesh):
        raise ValueError('Expected a NativeMesh')
    if type(width) not in (int, np.int64, np.int32) or type(height) not in (int, np.int64, np.int32) or width <= 0 or height <= 0 or width * height > MAX_RENDER_PIXELS:
        raise ValueError('Invalid or oversized render dimensions')
    if mesh.appearance.legacy_vertex_compatibility:
        return render_mesh(mesh.vertices, mesh.faces, K, R, t, width, height, scale=scale, vertex_colors=mesh.vertex_colors)
    _, _, K, R, t, scale = _inputs(mesh.vertices, mesh.faces, K, R, t, scale)
    depth = np.full((height, width), np.inf)
    rgb = np.zeros((height, width, 3), np.uint8)
    face_index = np.full((height, width), -1, np.int32)
    for part in mesh.appearance.primitives:
        points = scale * part.vertices @ R.T + t
        attrs = np.c_[part.tint, part.uv] if part.uv is not None else part.tint
        for local_index in raster_triangle_candidates(points,part.faces,K,width,height):
            face=part.faces[local_index]
            triangle = points[face]
            if not part.double_sided and part.winding_sign * np.dot(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]), triangle[0]) >= 0:
                continue
            p, attribute = _clip_near(points[face], attrs[face], 1e-8)
            for j in range(1, len(p) - 1):
                tri, attribute_tri = p[[0, j, j + 1]], attribute[[0, j, j + 1]]
                homogeneous = tri @ K.T
                projected = homogeneous[:, :2] / homogeneous[:, 2:]
                lo = np.maximum(np.ceil(projected.min(0)), [0, 0])
                hi = np.minimum(np.floor(projected.max(0)), [width - 1, height - 1])
                if np.any(lo > hi):
                    continue
                x0, y0 = lo.astype(int); x1, y1 = hi.astype(int)
                a, b, c = projected
                denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
                if abs(denominator) < 1e-12:
                    continue
                yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
                w0 = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / denominator
                w1 = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / denominator
                bary = np.stack([w0, w1, 1 - w0 - w1], -1)
                reciprocal = np.sum(bary / tri[:, 2], -1)
                z = np.divide(1, reciprocal, out=np.full_like(reciprocal, np.inf), where=reciprocal > 0)
                target = depth[y0:y1 + 1, x0:x1 + 1]
                closer = (bary.min(-1) >= -1e-10) & (z < target)
                if not closer.any():
                    continue
                weights = bary[closer] / tri[:, 2]
                weights /= reciprocal[closer, None]
                if include_rgb:
                    interpolated = weights @ attribute_tri
                    linear = interpolated[:, :3] * part.factor
                    if part.texture_linear is not None:
                        linear *= _sample_texture(part.texture_linear, interpolated[:, 3:5], part.sampler)
                    rgb[y0:y1 + 1, x0:x1 + 1][closer] = np.rint(_linear_to_srgb(linear) * 255).astype(np.uint8)
                target[closer] = z[closer]
                face_index[y0:y1 + 1, x0:x1 + 1][closer] = part.face_offset + local_index
    return {'mask': face_index >= 0, 'depth': depth.astype(np.float32), 'rgb': rgb, 'face_index': face_index}


def ray_native_mesh_points(uv, mesh: NativeMesh, K, R, t, scale=1.0):
    """Exact native canonical ray hits, using the same face visibility as render.

    Returns (Nx3 canonical points, N bool valid). Culled faces cannot create
    correspondences. This does not establish an independently accurate pose.
    """
    from alignment_render import ray_mesh_points
    if mesh.appearance.legacy_vertex_compatibility:
        return ray_mesh_points(uv, mesh.vertices, mesh.faces, K, R, t, scale=scale)
    vertices, faces, K, R, t, scale = _inputs(mesh.vertices, mesh.faces, K, R, t, scale)
    camera = scale * vertices @ R.T + t
    eligible = []
    for part in mesh.appearance.primitives:
        indices = np.arange(part.face_offset, part.face_offset + len(part.faces))
        triangle = camera[faces[indices]]
        if not part.double_sided:
            normal = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
            indices = indices[part.winding_sign * np.einsum('ij,ij->i', normal, triangle[:, 0]) < 0]
        eligible.extend(indices.tolist())
    if not eligible:
        pixels = np.asarray(uv, float)
        if pixels.ndim != 2 or pixels.shape[1] != 2:
            raise ValueError('uv must be Nx2')
        return np.full((len(pixels), 3), np.nan), np.zeros(len(pixels), bool)
    return ray_mesh_points(uv, vertices, faces[np.asarray(eligible)], K, R, t, scale=scale)


def render_native_mesh(mesh: NativeMesh, K, R, t, width, height, scale=1.0):
    """Render native mask/depth/RGB/global face IDs; see module colour policy."""
    return _render_native_mesh(mesh, K, R, t, width, height, scale=scale)


def render_native_mask(mesh: NativeMesh, K, R, t, width, height, scale=1.0):
    """Identical native geometry/depth/face IDs, with RGB intentionally omitted.

    Returns the usual dict, with rgb=None. Avoids texture sampling during bounded
    silhouette search; no decimation, clipping change or visibility approximation.
    """
    result = _render_native_mesh(mesh, K, R, t, width, height, scale=scale, include_rgb=False)
    result['rgb'] = None
    return result
