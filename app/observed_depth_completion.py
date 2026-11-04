"""Optional model-prior source-visible depth, separate from observed geometry.

MoGe is a learned monocular prior. Fitting its depth to observed triangles
does not make missing pixels observed 3D, or recover hidden backs/cavities.
This module never downloads a model and never uses hosted inference.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import spearmanr

import observed_surface_completion as C


PROTOCOL = {
    'version': 'moge-source-visible-depth-prior-v1',
    'model': 'Ruicheng/moge-2-vits-normal', 'implementation': 'moge.model.v2.MoGeModel',
    'device': 'mps', 'cpu_operator_fallback_enabled': False, 'downloads_allowed': False,
    'maximum_input_long_side': 512, 'num_tokens': 1200, 'use_fp16': False,
    'single_image_inference_count': 1,
    'alignment': 'robust affine predicted inverse depth to saved observed inverse depth',
    'fit_check_split': '4px blocks: ((x//4)*73856093+(y//4)*19349663)%5==0 is CHECK; all others FIT',
    'alignment_lattice_stride_px': 2, 'minimum_fit_pixels': 250, 'minimum_check_pixels': 100,
    'minimum_predictor_inverse_depth_iqr_fraction': 0.002,
    'maximum_normalized_design_condition': 100,
    'minimum_fit_spearman_correlation': 0.2,
    'minimum_positive_aligned_fraction': 0.99,
    'robust_fit_relative_soft_l1_scale': 0.003, 'maximum_fit_evaluations': 100,
    'maximum_check_median_relative_depth_error': 0.03,
    'maximum_check_p90_relative_depth_error': 0.08,
    'minimum_spatial_half_fit_pixels': 100,
    'maximum_spatial_fit_check_median_disagreement': 0.05,
    'maximum_spatial_fit_check_p90_disagreement': 0.10,
    'target_mask_core_px': 4, 'other_component_guard_px': 4,
    'observed_depth_quantiles': [2, 98], 'observed_range_padding_fraction': 1.0,
    'minimum_relative_depth_bound_padding': 0.05,
    'inferred_mesh_stride_px': 2, 'maximum_mesh_relative_depth_edge': 0.02,
    'maximum_mesh_triangles': 100000,
    'saved_camera_intrinsics_used': True, 'model_intrinsics_replace_saved_camera': False,
    'physical_confidence': 'unknown', 'hidden_surface_completion': False,
    'watertight': False, 'camera_calibrated': False, 'physical_scale_known': False,
    'new_api_calls': 0, 'heldout_video_frames_used': False,
}


def check_split(x, y):
    return ((np.asarray(x, np.int64)//4)*73856093 + (np.asarray(y, np.int64)//4)*19349663) % 5 == 0


def fit_affine_inverse(predictor, target):
    predictor, target = np.asarray(predictor, float), np.asarray(target, float)
    center = float(np.median(predictor))
    scale = float(np.percentile(predictor, 75)-np.percentile(predictor, 25))
    if not np.isfinite(scale) or scale <= abs(center)*PROTOCOL['minimum_predictor_inverse_depth_iqr_fraction']:
        raise ValueError('insufficient_predictor_depth_variation')
    design = np.c_[(predictor-center)/scale, np.ones(len(predictor))]
    condition = float(np.linalg.cond(design))
    if not np.isfinite(condition) or condition > PROTOCOL['maximum_normalized_design_condition']:
        raise ValueError('ill_conditioned_inverse_depth_alignment')
    initial = np.linalg.lstsq(design, target, rcond=None)[0]
    fit = least_squares(lambda parameters: design@parameters-target, initial,
                        loss='soft_l1', f_scale=PROTOCOL['robust_fit_relative_soft_l1_scale']*float(np.median(target)),
                        max_nfev=PROTOCOL['maximum_fit_evaluations'])
    a, b = float(fit.x[0]/scale), float(fit.x[1]-fit.x[0]*center/scale)
    return {'a': a, 'b': b, 'converged': bool(fit.success), 'evaluations': int(fit.nfev),
            'normalized_design_condition': condition, 'predictor_center': center, 'predictor_iqr': scale}


def align_depth(model_depth, observed_depth, domain):
    model, observed = np.asarray(model_depth, float), np.asarray(observed_depth, float)
    if model.shape != observed.shape or model.shape != np.asarray(domain).shape:
        raise ValueError('All alignment images must use the same source coordinates')
    yy, xx = np.indices(model.shape)
    finite = (np.asarray(domain, bool) & np.isfinite(observed) & (observed > 0)
              & (xx % PROTOCOL['alignment_lattice_stride_px'] == 0)
              & (yy % PROTOCOL['alignment_lattice_stride_px'] == 0))
    ys, xs = np.where(finite)
    check = check_split(xs, ys)
    valid_model = np.isfinite(model[ys,xs]) & (model[ys,xs] > 0)
    fit = ~check & valid_model
    report = {'fit_pixels': int(fit.sum()), 'check_pixels': int(check.sum()),
              'source_pixel_check_is_not_independent_shape_validation': True,
              'heldout_video_frames_read': False, 'passes': False, 'reasons': []}
    if fit.sum() < PROTOCOL['minimum_fit_pixels'] or check.sum() < PROTOCOL['minimum_check_pixels']:
        report['reasons'].append('insufficient_disjoint_fit_check_source_pixels')
        return None, report
    predictor = np.full(len(xs),np.nan)
    predictor[valid_model] = 1/model[ys[valid_model],xs[valid_model]]
    target = 1/observed[ys, xs]
    try:
        parameters = fit_affine_inverse(predictor[fit], target[fit])
    except ValueError as error:
        report['reasons'].append(str(error))
        return None, report
    report['fit_parameters'] = parameters
    if not parameters['converged']:
        report['reasons'].append('robust_alignment_did_not_converge')
    if parameters['a'] <= 0:
        report['reasons'].append('nonpositive_inverse_depth_slope')
    correlation = float(spearmanr(predictor[fit], target[fit]).statistic)
    report['fit_spearman_correlation'] = correlation if np.isfinite(correlation) else None
    if not np.isfinite(correlation) or correlation < PROTOCOL['minimum_fit_spearman_correlation']:
        report['reasons'].append('insufficient_positive_depth_rank_agreement')
    aligned_inverse = parameters['a']*predictor+parameters['b']
    positive = np.isfinite(aligned_inverse) & (aligned_inverse > 0)
    report['positive_aligned_fraction'] = float(positive.mean())
    if positive.mean() < PROTOCOL['minimum_positive_aligned_fraction']:
        report['reasons'].append('aligned_inverse_depth_not_positive')
    # Every CHECK observation is scored; invalid samples are failed evidence.
    error = np.full(int(check.sum()), np.inf)
    valid_check = positive[check]
    error[valid_check] = np.abs(1/aligned_inverse[check][valid_check]-observed[ys[check], xs[check]][valid_check])/observed[ys[check], xs[check]][valid_check]
    # Order statistics retain infinities as failed observations without
    # interpolating inf-inf into NaN at a quantile boundary.
    median = float(np.quantile(error,.5,method='higher'))
    p90 = float(np.quantile(error,.9,method='higher'))
    report['check_median_relative_depth_error'] = median if np.isfinite(median) else None
    report['check_p90_relative_depth_error'] = p90 if np.isfinite(p90) else None
    report['invalid_check_predictions'] = int((~valid_check).sum())
    if median > PROTOCOL['maximum_check_median_relative_depth_error']:
        report['reasons'].append('source_check_median_relative_depth_error')
    if p90 > PROTOCOL['maximum_check_p90_relative_depth_error']:
        report['reasons'].append('source_check_p90_relative_depth_error')
    # Alternative spatial FIT partitions are predefined, never selected by CHECK.
    spatial = []
    cut = float(np.median(xs[fit]))
    for name, partition in [('left', xs < cut), ('right', xs >= cut)]:
        chosen = fit & partition
        row = {'name': name, 'fit_pixels': int(chosen.sum())}
        if chosen.sum() < PROTOCOL['minimum_spatial_half_fit_pixels']:
            row['passes'] = False
            row['reason'] = 'insufficient_spatial_half_support'
        else:
            try:
                other = fit_affine_inverse(predictor[chosen], target[chosen])
                alternative = other['a']*predictor[check]+other['b']
                if not other['converged'] or other['a'] <= 0 or not np.isfinite(alternative).all() or (alternative <= 0).any() or not valid_check.all():
                    raise ValueError('unstable_or_nonpositive_spatial_fit')
                disagreement = np.abs(1/alternative-1/aligned_inverse[check])/(1/aligned_inverse[check])
                dm, dp = float(np.median(disagreement)), float(np.percentile(disagreement, 90))
                row.update(parameters=other, median_relative_disagreement=dm, p90_relative_disagreement=dp,
                           passes=dm <= PROTOCOL['maximum_spatial_fit_check_median_disagreement'] and dp <= PROTOCOL['maximum_spatial_fit_check_p90_disagreement'])
            except ValueError as error:
                row.update(passes=False, reason=str(error))
        spatial.append(row)
    report['spatial_fit_consistency'] = spatial
    if not all(row['passes'] for row in spatial):
        report['reasons'].append('spatial_fit_instability')
    report['passes'] = not report['reasons']
    valid = np.isfinite(model) & (model > 0)
    aligned = np.full(model.shape, np.nan)
    inverse = parameters['a']/model[valid]+parameters['b']
    positive = np.isfinite(inverse) & (inverse > 0)
    values = np.full(len(inverse), np.nan)
    values[positive] = 1/inverse[positive]
    aligned[valid] = values
    return aligned, report


def visible_prior_mesh(depth, inferred, domain, labels, image, pose, K, frame):
    stride = PROTOCOL['inferred_mesh_stride_px']
    h, w = domain.shape
    triangles, pixels = [], set()
    for y in range(0, h-stride, stride):
        for x in range(0, w-stride, stride):
            for corners in (((x,y),(x+stride,y),(x+stride,y+stride)), ((x,y),(x+stride,y+stride),(x,y+stride))):
                uv = np.array(corners)
                if not inferred[uv[:,1],uv[:,0]].any():
                    continue
                # Test every intervening source pixel, including a different-part guard.
                if not domain[y:y+stride+1,x:x+stride+1].all():
                    continue
                z = depth[uv[:,1],uv[:,0]]
                ids = labels[uv[:,1],uv[:,0]]
                if not np.isfinite(z).all() or (z <= 0).any() or len(np.unique(ids)) != 1 or ids[0] <= 0 or np.ptp(z)/np.median(z) > PROTOCOL['maximum_mesh_relative_depth_edge']:
                    continue
                triangles.append(corners)
                pixels.update(corners)
    if not triangles:
        return None
    if len(triangles) > PROTOCOL['maximum_mesh_triangles']:
        raise ValueError('Model-prior surface exceeds fixed triangle budget')
    pixels = sorted(pixels)
    lookup = {p:i for i,p in enumerate(pixels)}
    uv = np.array(pixels)
    xyz = C.unproject(uv, depth[uv[:,1],uv[:,0]], pose, K)
    center = -np.asarray(pose[0]).T@np.asarray(pose[1])
    indices = []
    for triangle in triangles:
        face = [lookup[p] for p in triangle]
        a,b,c = xyz[face]
        if np.dot(np.cross(b-a,c-a),center-a) < 0:
            face[1],face[2] = face[2],face[1]
        indices.append(face)
    return {'kind':'model_prior_source_visible_patch', 'component_id':'camera_body',
            'vertices':xyz.tolist(), 'triangles':indices, 'colors':image[uv[:,1],uv[:,0],::-1].tolist(),
            'uv':np.c_[(uv[:,0]+.5)/w,1-(uv[:,1]+.5)/h].tolist(),
            'vertex_source_pixels':uv.tolist(),
            'vertex_provenance':['model_prior_aligned_inverse_depth' if inferred[y,x] else 'observed_triangle_interpolation_boundary' for x,y in pixels],
            'face_provenance':['model_prior_aligned_inverse_depth']*len(indices),
            'face_confidence':['unknown']*len(indices),
            'observed_face_count':0, 'inferred_face_count':len(indices),
            'texture_url':frame['image_url'], 'source_frame_id':frame['id'],
            'depth_observed':False, 'hidden_surface_completion':False,
            'hole_filling':True, 'watertight':False, 'thickness_known':False,
            'interpretation':'Monocular learned-depth hypotheses only in visible missing regions; image texture is observed appearance, not a geometry validation.'}


def run(root, source_directory, output_directory, excluded_mask_urls=()):
    root = Path(root).resolve()
    source, output = (root/source_directory).resolve(), (root/output_directory).resolve()
    if not source.is_relative_to(root/'data') or not output.is_relative_to(root/'data'):
        raise ValueError('Source and output must remain inside project data')
    output.mkdir(parents=True, exist_ok=False)
    viewer_path = source/'viewer.json'
    view = json.loads(viewer_path.read_text())
    if view.get('kind') != 'observed_surface' or view.get('surface',{}).get('kind') != 'observed_partial_patch':
        raise ValueError('Only a saved observed partial surface is eligible')
    frame = next(f for f in view['frames'] if f['id']==view['surface']['source_frame_id'])
    camera = next(c for c in view['cameras'] if c['frame_id']==frame['id'])
    image_path, mask_path = C._local(root,frame['image_url']), C._local(root,frame['mask_url'])
    exclude_paths = [C._local(root,url) for url in excluded_mask_urls]
    checkpoint = root/'models/moge-2-vits-normal/model.pt'
    provenance_path = checkpoint.with_name('provenance.json')
    provenance = json.loads(provenance_path.read_text())
    if not checkpoint.is_file() or C.sha256(checkpoint)!=provenance['checkpoint_sha256']:
        raise ValueError('Existing checkpoint identity mismatch; downloads disabled')
    implementation = Path(__file__).resolve()
    dependency = Path(C.__file__).resolve()
    from importlib.util import find_spec
    model_code = Path(find_spec('moge.model.v2').origin)
    paths = [viewer_path,source/'protocol.json',image_path,mask_path,checkpoint,provenance_path,*exclude_paths]
    hashes = {str(p.relative_to(root)):C.sha256(p) for p in paths}
    frozen = {'protocol':copy.deepcopy(PROTOCOL), 'source_directory':str(source.relative_to(root)),
              'source_frame':frame, 'component_id':'camera_body', 'input_sha256':hashes,
              'model_provenance':provenance, 'source_timestamp_s':frame['timestamp_s'],
              'source_time_domain':'original source video presentation seconds, exact saved sampled frame; no interpolation',
              'K':camera['K'], 'R':camera['R'], 't':camera['t'],
              'excluded_component_mask_urls':list(excluded_mask_urls),
              'component_boundaries_verified':False,
              'implementation_sha256':C.sha256(implementation), 'geometry_dependency_sha256':C.sha256(dependency),
              'model_module_sha256':C.sha256(model_code),
              'frozen_before_model_inference':True}
    (output/'protocol.json').write_text(json.dumps(frozen,indent=2)+'\n')
    (output/'implementation.py').write_bytes(implementation.read_bytes())
    (output/'geometry-dependency.py').write_bytes(dependency.read_bytes())
    (output/'moge-v2-implementation.py').write_bytes(model_code.read_bytes())
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':
        raise RuntimeError('CPU operator fallback must be disabled')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    # Prevent accidental network access by libraries during this isolated process.
    import socket
    def no_network(*args,**kwargs):
        raise RuntimeError('Network is disabled for the local model-prior experiment')
    socket.socket.connect = no_network
    socket.create_connection = no_network
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no CPU fallback or installation')
    from moge.model.v2 import MoGeModel
    import moge.model.v2 as model_module
    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)
    excluded = [cv2.imread(str(p),cv2.IMREAD_GRAYSCALE) for p in exclude_paths]
    if image is None or mask is None or image.shape[:2]!=mask.shape or any(p is None for p in excluded):
        raise ValueError('Exact source image/mask unavailable')
    h,w = mask.shape
    factor = min(1.,PROTOCOL['maximum_input_long_side']/max(h,w))
    size = (max(1,round(w*factor)),max(1,round(h*factor)))
    resized = cv2.resize(image,size,interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized[:,:,::-1].copy()).permute(2,0,1).float().to('mps')/255
    started = time.monotonic()
    model = MoGeModel.from_pretrained(str(checkpoint)).to('mps').eval()
    with torch.inference_mode():
        prediction = model.infer(tensor,num_tokens=PROTOCOL['num_tokens'],use_fp16=False,apply_mask=True)
        torch.mps.synchronize()
        raw = prediction['depth'].cpu().numpy().astype(np.float32)
        valid = prediction['mask'].cpu().numpy().astype(bool)
        model_K = prediction['intrinsics'].cpu().numpy().astype(float)
    elapsed = time.monotonic()-started
    # Resize inverse depth with explicit invalid support rather than infinities.
    valid &= np.isfinite(raw)&(raw>0)
    weights = cv2.resize(valid.astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)
    inverse = np.zeros(raw.shape,np.float32)
    inverse[valid] = 1/raw[valid]
    up = cv2.resize(inverse,(w,h),interpolation=cv2.INTER_LINEAR)
    predicted = np.full((h,w),np.nan)
    good = (weights>=.999)&np.isfinite(up)&(up>0)
    predicted[good] = weights[good]/up[good]
    K = np.array(camera['K']);pose=(np.array(camera['R']),np.array(camera['t']))
    observed = C.rasterize_depth(view['surface'],pose,K,(h,w))
    domain, labels = C.component_domain(mask,excluded)
    aligned, alignment = align_depth(predicted,observed,domain)
    inferred = np.zeros((h,w),bool)
    mesh = None
    observed_good = domain&np.isfinite(observed)&(observed>0)
    observed_values = observed[observed_good]
    bounds = None
    if alignment['passes'] and aligned is not None:
        low,high = np.percentile(observed_values,PROTOCOL['observed_depth_quantiles'])
        padding = max((high-low)*PROTOCOL['observed_range_padding_fraction'],float(np.median(observed_values))*PROTOCOL['minimum_relative_depth_bound_padding'])
        bounds = {'lower':float(max(1e-8,low-padding)),'upper':float(high+padding), 'based_only_on_observed_source_triangles':True}
        inferred = domain&~observed_good&np.isfinite(aligned)&(aligned>=bounds['lower'])&(aligned<=bounds['upper'])
        combined = np.where(observed_good,observed,aligned)
        combined[~(observed_good|inferred)] = np.nan
        mesh = visible_prior_mesh(combined,inferred,domain,labels,image,pose,K,frame)
        if mesh:
            mesh['texture_sha256']=C.sha256(image_path)
    np.savez_compressed(output/'prediction.npz',raw_model_depth=raw,raw_model_valid=valid,
                        model_intrinsics_normalized=model_K,predicted_depth_source_pixels=predicted,
                        aligned_depth_source_pixels=aligned if aligned is not None else np.full((h,w),np.nan),
                        inferred_source_mask=inferred)
    counts = {'observed_triangles_unchanged':len(view['surface']['triangles']),
              'observed_source_coverage_pixels':int(observed_good.sum()),
              'model_inferred_source_pixels':int(inferred.sum()),
              'inferred_vertices':len(mesh['vertices']) if mesh else 0,
              'inferred_triangles':len(mesh['triangles']) if mesh else 0,
              'observed_3d_points_added':0,'model_inference_calls':1,'new_api_calls':0}
    limits = ['All added depths and faces are learned model-prior hypotheses, not observed multi-view 3D.',
              'Disjoint source-pixel checks test alignment conditional on one source frame; they are not held-out-view shape accuracy.',
              'Source texture does not validate depth. Hidden backs, cavities, thickness, metric scale and physical accuracy remain unverified.',
              'Saved ambiguous intrinsics are used for unprojection. MoGe intrinsics are recorded but never substituted.',
              'Target mask and other component boundaries require semantic review before enabling this optional layer.',
              'Original observed geometry is byte-preserved; this candidate is separate and does not enter production assets.']
    report = {'status':'model_prior_candidate_requires_component_review' if mesh else 'model_alignment_rejected' if not alignment['passes'] else 'no_eligible_model_prior_triangles',
              'alignment':alignment,'counts':counts,'model_inference_seconds':elapsed,'model_input_size_wh':list(size),
              'model_intrinsics_normalized_record_only':model_K.tolist(),'model_intrinsics_used_for_unprojection':False,
              'model_module_sha256':C.sha256(model_module.__file__),'torch_version':torch.__version__,
              'device':'mps','cpu_operator_fallback_enabled':False,'depth_bounds':bounds,
              'physical_confidence':'unknown','limitations':limits}
    (output/'completion.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    derived = {'format':'model-prior-depth-completion-review','version':1,'id':output.name,
               'title':'Optional learned visible-depth completion','kind':'model_prior_depth_completion',
               'status':report['status'],'reason':'A single-image learned prior fitted to observed source depth; additions are inferred, not observed 3D.',
               'metrics':counts,'frames':view['frames'],'cameras':view['cameras'],'points':[],
               'source_frame_id':frame['id'],'component_id':'camera_body','inferred_surface':mesh,
               'observed_surface_url':'/files/'+str(viewer_path.relative_to(root)),
               'depth_completion_report':report,'limitations':limits,'new_api_calls':0,'production_geometry_changed':False}
    (output/'viewer.json').write_text(json.dumps(derived,separators=(',',':'),allow_nan=False)+'\n')
    overlay = image.copy()
    overlay[observed_good]=(.65*overlay[observed_good]+.35*np.array([100,185,105])).astype(np.uint8)
    overlay[inferred]=(.35*overlay[inferred]+.65*np.array([45,165,245])).astype(np.uint8)
    cv2.imwrite(str(output/'source-model-inference.png'),overlay)
    for path,digest in hashes.items():
        if C.sha256(root/path)!=digest:
            raise ValueError('An original model or observed input changed')
    (output/'input-integrity.json').write_text(json.dumps({'unchanged':True,'input_sha256':hashes},indent=2)+'\n')
    return report
