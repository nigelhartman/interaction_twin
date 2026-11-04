"""CPU checkerboard calibration with frozen view splits and explicit pixel units.

A planar checkerboard is the intended target. Repeated/frontoparallel *views*
are rejected. No phone metadata, object mesh, depth prior or earlier pose is read.
"""
from __future__ import annotations
import hashlib, json, math, shutil, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
POLICY={
    'maximum_frames':50,'maximum_runtime_s':120,'maximum_source_duration_s':120,
    'maximum_input_bytes':536870912,'maximum_detection_side':1600,
    'decoder_format_whitelist':['mov','matroska','webm','avi'],
    'minimum_detected_views':12,'minimum_training_views':8,'minimum_heldout_views':3,
    'heldout_every_nth_sample':4,'minimum_corner_hull_fraction':.02,
    'minimum_training_centroid_cells':3,'minimum_homography_fifth_singular_ratio':1e-4,
    'minimum_training_normal_spread_deg':10.,'minimum_training_corner_coverage':.30,
    'maximum_training_rms_px':1.5,'maximum_heldout_median_px':1.5,
    'maximum_heldout_p90_px':3.,'minimum_heldout_pass_fraction':.75,
    'maximum_relative_focal_std':.10,'maximum_calibration_iterations':60,
    'distortion_model':'opencv_brown_k1_k2_p1_p2_fixed_k3_zero',
    'outlier_policy':'at_most_one_TRAIN_view_refit; max(1px,median+3_scaled_MAD)',
    'intrinsic_fit_uses_only_training_views':True,'heldout_selects_model_or_outliers':False,
    'heldout_pose_anchors':'even_checkerboard_index_parity','heldout_score_corners':'odd_checkerboard_index_parity',
    'physical_accuracy_verified':False,'object_metric_scale_established':False,
}

def require(condition,message):
    if not condition:raise ValueError(message)

def sha(data):return hashlib.sha256(data).hexdigest()
def read_json(path):return json.loads(Path(path).read_text())
def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def board_spec(value):
    require(isinstance(value,dict) and set(value)<={'inner_rows','inner_cols','square_size_mm'},'Unknown board fields')
    rows,cols=value.get('inner_rows'),value.get('inner_cols');size=value.get('square_size_mm')
    require(type(rows) is int and type(cols) is int and 4<=rows<=20 and 4<=cols<=20,'Board needs4–20 inner corners on each axis')
    require(rows*cols<=240,'Board exceeds bounded corner count')
    require(size is None or type(size) in {int,float} and math.isfinite(size) and 0<size<=1000,'Square size must be null or a positive measured millimetre value')
    return {'inner_rows':rows,'inner_cols':cols,'square_size_mm':None if size is None else float(size),
            'object_point_units':'grid_units','board_pose_output_units':'grid_units' if size is None else 'mm',
            'size_source':'not_provided' if size is None else 'user_supplied_measurement_not_independently_verified',
            'metric_scale_verified':False}

def object_points(board):
    xy=np.mgrid[0:board['inner_cols'],0:board['inner_rows']].T.reshape(-1,2)
    return np.column_stack([xy,np.zeros(len(xy))]).astype(np.float32)

def pose_dict(rvec,tvec,board):
    scale=board.get('square_size_mm') or 1.
    return {'rotation_vector':np.asarray(rvec).reshape(3).tolist(),
            'translation':(np.asarray(tvec).reshape(3)*scale).tolist(),
            'translation_units':board['board_pose_output_units'],
            'metric_scale_verified':False}

def hull_fraction(corners,size):
    return float(cv2.contourArea(cv2.convexHull(np.asarray(corners,np.float32))))/(size[0]*size[1])

def homography_diversity(views,board,size):
    """Zhang constraints in normalized pixels; TRAIN observations only."""
    obj=object_points(board)[:,:2].astype(float);obj/=max(board['inner_rows'],board['inner_cols'])
    scale=max(size);constraints=[]
    def vij(h,i,j):
        return np.array([h[0,i]*h[0,j],h[0,i]*h[1,j]+h[1,i]*h[0,j],h[1,i]*h[1,j],
                         h[2,i]*h[0,j]+h[0,i]*h[2,j],h[2,i]*h[1,j]+h[1,i]*h[2,j],h[2,i]*h[2,j]])
    for view in views:
        image=(np.asarray(view['corners'],float)-np.array(size)/2)/scale
        h,_=cv2.findHomography(obj,image,0)
        if h is None:continue
        for row in [vij(h,0,1),vij(h,0,0)-vij(h,1,1)]:
            norm=np.linalg.norm(row)
            if norm>0:constraints.append(row/norm)
    singular=np.linalg.svd(np.array(constraints),compute_uv=False) if len(constraints)>=6 else np.zeros(6)
    ratio=float(singular[-2]/singular[0]) if singular[0]>0 else 0.
    return {'normalized_constraint_singular_values':singular.tolist(),'fifth_to_first_ratio':ratio,
            'meaning':'Conditioning proxy for varied planar target views; not physical accuracy.'}

def fit_camera(views,image_size,board,*,deadline=None):
    """Pure numeric stage. Callers must independently bind detections to source.

    `views` already has the prospective TRAIN/heldout assignments. Synthetic
    analytic fixtures may call this function, but are not real calibrations.
    """
    board=board_spec({k:board.get(k) for k in ['inner_rows','inner_cols','square_size_mm']})
    width,height=image_size;obj=object_points(board);count=len(obj)
    require(width>0 and height>0,'Positive native image dimensions required')
    train=[v for v in views if v['split']=='train' and v.get('eligible')]
    tests=[v for v in views if v['split']=='heldout']
    result={'status':'insufficient_evidence','candidate':None,'training_views':[],'heldout_views':[],
            'diagnostics':[],'metrics':{'sampled_views':len(views),'eligible_train_views':len(train),
            'sampled_heldout_views':len(tests),'detected_views':sum(bool(v.get('detected')) for v in views)}}
    if len(train)<8 or sum(v.get('eligible',False) for v in tests)<3 or sum(bool(v.get('eligible')) for v in views)<12:
        result['diagnostics'].append({'code':'need_more_detected_views','action':'Capture at least12 clear board views, including3 reserved views, with the whole pattern visible.'});return result
    for v in train:
        a=np.asarray(v['corners'],float);require(a.shape==(count,2) and np.isfinite(a).all(),'Invalid automatic corner observations')
    centers=np.array([np.mean(v['corners'],axis=0) for v in train]);cells={(min(2,int(x/width*3)),min(2,int(y/height*3))) for x,y in centers}
    all_points=np.concatenate([np.asarray(v['corners'],np.float32) for v in train]);coverage=hull_fraction(all_points,image_size)
    diversity=homography_diversity(train,board,image_size)
    result['metrics'].update(training_centroid_cells=len(cells),training_corner_coverage=coverage,homography_diversity=diversity)
    if len(cells)<3 or coverage<.30 or diversity['fifth_to_first_ratio']<1e-4:
        result['status']='rejected_degenerate_views';result['diagnostics'].append({'code':'insufficient_view_diversity',
            'action':'Move the full board toward different image edges and tilt it in both axes. Repeated front-facing views cannot constrain reliable intrinsics.'});return result
    def calibrate(rows):
        if deadline is not None and time.monotonic()>deadline:raise TimeoutError('calibration_deadline')
        return cv2.calibrateCameraExtended([obj.copy() for _ in rows],[np.asarray(v['corners'],np.float32).reshape(-1,1,2) for v in rows],
            (width,height),None,None,flags=cv2.CALIB_FIX_K3,
            criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,60,1e-10))
    def finite_solution(solution):
        return all(np.isfinite(np.asarray(x)).all() for x in solution)
    first=calibrate(train)
    if not finite_solution(first):
        result.update(status='rejected_degenerate_views',diagnostics=[{'code':'nonfinite_training_solution','action':'Recapture clearer, diverse full-board views; retain this failed fit.'}]);return result
    rms,K,dist,rvecs,tvecs,std_i,_,errors=first
    per_view=np.asarray(errors).reshape(-1);median=float(np.median(per_view));mad=float(np.median(np.abs(per_view-median)))
    threshold=max(1.,median+3*1.4826*mad);reject={v['id'] for v,e in zip(train,per_view) if e>threshold}
    retained=[v for v in train if v['id'] not in reject]
    refit=bool(reject and len(retained)>=8)
    result['training_views']=[{'id':v['id'],'used':v['id'] not in reject,'initial_rms_px':float(e),
        'reason':'TRAIN_view_robust_outlier' if v['id'] in reject else 'retained'} for v,e in zip(train,per_view)]
    if reject:
        retained_centers=np.array([np.mean(v['corners'],axis=0) for v in retained])
        retained_cells={(min(2,int(x/width*3)),min(2,int(y/height*3))) for x,y in retained_centers}
        retained_coverage=hull_fraction(np.concatenate([np.asarray(v['corners'],np.float32) for v in retained]),image_size) if retained else 0.
        retained_diversity=homography_diversity(retained,board,image_size)
        result['metrics'].update(retained_training_views=len(retained),retained_training_centroid_cells=len(retained_cells),
            retained_training_corner_coverage=retained_coverage,retained_homography_diversity=retained_diversity,
            training_outlier_view_ids=sorted(reject),training_outlier_threshold_px=threshold,training_refit_performed=False)
        if len(retained)<8 or len(retained_cells)<3 or retained_coverage<.30 or retained_diversity['fifth_to_first_ratio']<1e-4:
            result.update(status='rejected_degenerate_views',diagnostics=[{'code':'insufficient_diverse_clean_training_views',
                'action':'The prospective TRAIN-only outlier exclusion leaves insufficient diverse evidence. Recapture; do not promote a fitted camera.'}]);return result
    if refit:
        refined=calibrate(retained)
        if not finite_solution(refined):
            result.update(status='rejected_degenerate_views',diagnostics=[{'code':'nonfinite_training_refit','action':'Recapture clearer, diverse full-board views; retain this failed fit.'}]);return result
        rms,K,dist,rvecs,tvecs,std_i,_,errors=refined
    else:retained=train
    normals=np.array([cv2.Rodrigues(r)[0][:,2] for r in rvecs]);dots=np.clip(np.abs(normals@normals.T),0,1)
    normal_spread=float(np.degrees(np.arccos(dots)).max());std=np.asarray(std_i).reshape(-1)
    dist=np.asarray(dist).reshape(-1);focal_std=[float(std[0]/K[0,0]),float(std[1]/K[1,1])]
    result['metrics'].update(training_rms_px=float(rms),retained_training_views=len(retained),
        training_refit_performed=refit,training_outlier_view_ids=sorted(reject),training_outlier_threshold_px=threshold,
        training_normal_spread_deg=normal_spread,relative_focal_std=focal_std,
        uncertainty_note='OpenCV local parameter standard deviations; conditional on model/corner noise, not ground-truth calibration accuracy.')
    candidate={'K':np.asarray(K).tolist(),'distortion_coefficients':dist.tolist(),'image_size':[width,height],
        'distortion_model':POLICY['distortion_model'],'training_board_poses':[{'id':v['id'],**pose_dict(r,t,board)} for v,r,t in zip(retained,rvecs,tvecs)],
        'opencv_intrinsic_parameter_std':std.tolist(),'physical_accuracy_verified':False}
    result['candidate']=candidate
    numeric=np.isfinite(K).all() and np.isfinite(dist).all() and np.isfinite(std).all()
    plausible=numeric and .2*max(image_size)<=K[0,0]<=5*max(image_size) and .2*max(image_size)<=K[1,1]<=5*max(image_size)
    plausible=plausible and -.1*width<=K[0,2]<=1.1*width and -.1*height<=K[1,2]<=1.1*height and np.max(np.abs(dist))<2
    if not plausible or rms>1.5 or max(focal_std)>.10 or normal_spread<10:
        result['status']='rejected_degenerate_views';result['diagnostics'].append({'code':'unstable_or_inaccurate_training_fit',
            'action':'Use sharper, more varied board orientations and positions. This candidate is not an accepted camera record.'});return result
    anchor=np.array([(r+c)%2==0 for r in range(board['inner_rows']) for c in range(board['inner_cols'])]);score=~anchor
    passed=0
    for v in tests:
        if deadline is not None and time.monotonic()>deadline:raise TimeoutError('calibration_deadline')
        row={'id':v['id'],'status':'not_eligible','reasons':list(v.get('reasons',[])),
             'anchor_corner_ids':np.where(anchor)[0].tolist(),'scored_corner_ids':np.where(score)[0].tolist()}
        if v.get('eligible'):
            try:
                corners=np.asarray(v['corners'],np.float32)
                require(corners.shape==(count,2) and np.isfinite(corners).all(),'invalid_heldout_corners')
                found,rvec,tvec=cv2.solvePnP(obj[anchor],corners[anchor],K,dist,flags=cv2.SOLVEPNP_ITERATIVE)
                require(found,'heldout_anchor_pose_failed');rotation=cv2.Rodrigues(rvec)[0]
                require(np.all((obj@rotation.T+np.asarray(tvec).reshape(3))[:,2]>0),'heldout_cheirality_failed')
                projected=cv2.projectPoints(obj,rvec,tvec,K,dist)[0].reshape(-1,2);residual=np.linalg.norm(projected-corners,axis=1)
                median_score=float(np.median(residual[score]));p90=float(np.percentile(residual[score],90));anchor_rms=float(np.sqrt(np.mean(residual[anchor]**2)))
                ok=median_score<=1.5 and p90<=3 and anchor_rms<=1.5
                row.update(status='passed_conditional_corner_check' if ok else 'failed_corner_check',
                    anchor_rms_px=anchor_rms,scored_median_px=median_score,scored_p90_px=p90,
                    scored_residuals_px=residual[score].tolist(),board_pose=pose_dict(rvec,tvec,board))
                passed+=int(ok)
            except (ValueError,cv2.error):row.update(status='pose_failed',reasons=['heldout_anchor_pose_failed'])
        result['heldout_views'].append(row)
    fraction=passed/len(tests) if tests else 0.
    result['metrics'].update(heldout_passed_views=passed,heldout_pass_fraction=fraction)
    result['status']='candidate_passed_checks' if passed>=3 and fraction>=.75 else 'rejected_validation'
    if result['status']!='candidate_passed_checks':result['diagnostics'].append({'code':'withheld_views_do_not_validate',
        'action':'Keep every reserved failure. Recapture consistent camera settings and clearer, varied views; do not tune calibration against these test views.'})
    return result

def detect(gray,board):
    """Resize for detection, then refine native-pixel corners with exact mapping."""
    h,w=gray.shape;factor=min(1.,1600/max(w,h));nw,nh=max(1,round(w*factor)),max(1,round(h*factor))
    small=cv2.resize(gray,(nw,nh),interpolation=cv2.INTER_AREA) if (nw,nh)!=(w,h) else gray
    sx,sy=nw/w,nh/h
    found,corners=cv2.findChessboardCornersSB(small,(board['inner_cols'],board['inner_rows']),flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    mapping={'native_size':[w,h],'detection_size':[nw,nh],'scale_x':sx,'scale_y':sy,
             'inverse_pixel_center_mapping':'native=(detected+0.5)/actual_axis_scale-0.5; native_cornerSubPix_refinement'}
    if not found:return None,mapping
    # OpenCV 4/5 bindings may return N×1×2 or N×2. Normalize the
    # representation before applying the same pixel-centre transform.
    corners=np.asarray(corners,np.float32).reshape(-1,1,2);corners[:,:,0]=(corners[:,:,0]+.5)/sx-.5;corners[:,:,1]=(corners[:,:,1]+.5)/sy-.5
    require(np.all(corners[:,:,0]>=4) and np.all(corners[:,:,0]<w-4) and np.all(corners[:,:,1]>=4) and np.all(corners[:,:,1]<h-4),'corners_too_close_to_image_edge')
    corners=cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,.001))
    return corners.reshape(-1,2),mapping

def _confined(root,path,*,file=False):
    p=Path(path);p=p if p.is_absolute() else root/p
    require(p==p.resolve() and p.is_relative_to(root/'data') and not any(x.startswith('.') for x in p.relative_to(root).parts),'Only non-hidden nonsymlinked data inputs/outputs')
    if file:require(p.is_file(),'Input file missing')
    return p

def _command(argv,deadline):
    remaining=deadline-time.monotonic()
    if remaining<=0:raise TimeoutError('calibration_deadline')
    process=subprocess.run(argv,capture_output=True,timeout=remaining,check=False)
    require(process.returncode==0,'calibration_media_decode_failed')
    require(len(process.stdout)<=16_000_000,'calibration_media_probe_excessive')
    return process.stdout

def _sources(source,out,root,max_frames,deadline):
    rows=[]
    if source.is_dir():
        files=sorted(p for p in source.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'})
        require(0<len(files)<=2000,'Choose1–2000 explicit calibration images')
        selected=np.linspace(0,len(files)-1,min(max_frames,len(files))).astype(int)
        descriptors=[]
        for i in selected:
            p=_confined(root,files[i],file=True);require(p.stat().st_size<=POLICY['maximum_input_bytes'],'Image input too large')
            descriptors.append({'path':str(p.relative_to(root)),'sha256':sha(p.read_bytes()),'source_index':int(i),'timestamp_s':None})
        source_record={'kind':'explicit_image_directory','path':str(source.relative_to(root)),
            'selected_input_sha256':sha(json.dumps(descriptors,sort_keys=True,separators=(',',':')).encode()),
            'ordering':'lexical_filename_order; no invented image timestamps','total_image_files':len(files)}
        for seq,d in enumerate(descriptors):rows.append({**d,'id':f'view_{seq:04d}','input_path':root/d['path']})
    else:
        require(source.is_file() and source.suffix.lower() in {'.mp4','.mov','.mkv','.webm','.avi','.m4v'},'Unsupported explicit calibration source')
        require(source.stat().st_size<=POLICY['maximum_input_bytes'],'Video input too large')
        probe=shutil.which('ffprobe');ffmpeg=shutil.which('ffmpeg');require(probe and ffmpeg,'Prepared ffmpeg/ffprobe required')
        raw=_command([probe,'-v','error','-protocol_whitelist','file,pipe','-format_whitelist','mov,matroska,webm,avi','-select_streams','v:0','-show_entries','stream=width,height:stream_side_data=rotation:format=format_name,duration:frame=best_effort_timestamp_time','-show_frames','-of','json',str(source)],deadline)
        (out/'source-probe.json').write_bytes(raw);meta=json.loads(raw);duration=float(meta.get('format',{}).get('duration',0))
        formats=set(meta.get('format',{}).get('format_name','').split(','))
        require(bool(formats & {'mov','mp4','matroska','webm','avi'}) and formats <= {'mov','mp4','m4a','3gp','3g2','mj2','matroska','webm','avi'},'Unsupported probed movie container')
        require(0<duration<=120,'Use a calibration recording no longer than120 seconds')
        frames=meta.get('frames',[]);require(0<len(frames)<=30000,'Unsupported source frame count')
        pts=[float(f['best_effort_timestamp_time']) for f in frames]
        require(all(math.isfinite(x) for x in pts) and all(b>a for a,b in zip(pts,pts[1:])),'Native frame PTS must be finite and strictly ordered')
        selected=np.linspace(0,len(frames)-1,min(max_frames,len(frames))).astype(int).tolist()
        native=out/'native';native.mkdir();expression='+'.join(f'eq(n\\,{i})' for i in selected)
        _command([ffmpeg,'-v','error','-protocol_whitelist','file,pipe','-format_whitelist','mov,matroska,webm,avi','-noautorotate','-i',str(source),'-vf','select='+expression,'-fps_mode','vfr',str(native/'frame-%04d.png')],deadline)
        decoded=sorted(native.glob('frame-*.png'));require(len(decoded)==len(selected),'Exact selected native frame count mismatch')
        source_record={'kind':'video','path':str(source.relative_to(root)),'sha256':sha(source.read_bytes()),
            'duration_s':duration,'native_frame_count':len(frames),'probe_sha256':sha(raw),'probed_format_names':sorted(formats),'decoder_protocol_whitelist':['file','pipe'],
            'pixel_orientation':'ffmpeg_noautorotate_encoded_pixels','rotation_metadata':meta.get('streams',[{}])[0].get('side_data_list',[])}
        for seq,(index,image) in enumerate(zip(selected,decoded)):
            rows.append({'id':f'view_{seq:04d}','path':str(image.relative_to(root)),'input_path':image,
                'source_index':index,'timestamp_s':pts[index],'native_index_domain':'decoded_video_frame_index_noautorotate'})
    return rows,source_record

def run(source,output,*,root=ROOT,board,max_frames=50,max_seconds=120,on_progress=None):
    root=Path(root).resolve();source=_confined(root,source);out=_confined(root,output)
    require(source.exists() and not out.exists(),'Existing explicit source and fresh output required')
    require(type(max_frames) is int and 1<=max_frames<=50 and type(max_seconds) in {int,float} and 0<max_seconds<=120,'Bounded50frame/120second policy required')
    board=board_spec(board);start=time.monotonic();deadline=start+max_seconds;out.mkdir(parents=True)
    code=Path(__file__).read_bytes();(out/'implementation.py').write_bytes(code)
    protocol={'format':'checkerboard-calibration-protocol','version':1,'board':board,'policy':POLICY,
        'implementation_sha256':sha(code),'opencv_version':cv2.__version__,'numpy_version':np.__version__,
        'requested_limits':{'max_frames':max_frames,'max_seconds':max_seconds},
        'runtime_bound_note':'Core checks between bounded native-library operations; supervising job process enforces the hard wall-clock limit.',
        'decoder_protocol_whitelist':['file','pipe'],
        'camera_session_binding':{'source_is_calibration_session_only':True,'later_capture_association_required':True,
            'same_resolution_alone_does_not_prove_same_camera':True},'new_api_calls':0}
    write_json(out/'protocol.json',protocol)
    report={'format':'checkerboard-camera-calibration','version':1,'status':'insufficient_evidence',
        'board':board,'policy':POLICY,'source':None,'image_size':None,'camera_record':None,
        'metrics':{},'diagnostics':[],'splits':{'train':[],'heldout':[]},'camera_session_binding':protocol['camera_session_binding'],
        'physical_accuracy_verified':False,'object_metric_scale_established':False,'new_api_calls':0,'started_utc':datetime.now(timezone.utc).isoformat()}
    detections=[];admitted_source_sha=None
    def save_detections():
        path=out/'detections.json'
        write_json(path,{'format':'automatic-checkerboard-detections','version':1,'views':detections,
            'corner_index_order':'row-major; OpenCV detected inner corners','split_rule':'sample_sequence_index%4==0 heldout, assigned before detection','manual_corners':False})
        return {'path':str(path.relative_to(root)),'sha256':sha(path.read_bytes())}
    def progress(phase,completed,total):
        value={'phase':phase,'completed':completed,'total':total,'elapsed_s':time.monotonic()-start}
        write_json(out/'progress.json',value)
        if on_progress:on_progress(value)
    try:
        # Bind even a failed media decode to the exact admitted input. No source
        # metadata is taken as a measured camera calibration.
        if source.is_file():
            require(source.stat().st_size<=POLICY['maximum_input_bytes'],'Video input too large')
            admitted_source_sha=sha(source.read_bytes())
            report['source']={'kind':'video','path':str(source.relative_to(root)),'sha256':admitted_source_sha}
        progress('extract',0,max_frames);rows,source_record=_sources(source,out,root,max_frames,deadline)
        if admitted_source_sha is not None:require(source_record['sha256']==admitted_source_sha,'source_changed_during_calibration')
        report['source']=source_record
        image_size=None;train_hashes=set();heldout_hashes=set();overlays=out/'overlays';overlays.mkdir()
        for seq,row in enumerate(rows):
            if time.monotonic()>deadline:raise TimeoutError('calibration_deadline')
            progress('detect',seq,len(rows));split='heldout' if seq%4==0 else 'train'
            p=row.pop('input_path');encoded=p.read_bytes()
            if row.get('sha256'):require(sha(encoded)==row['sha256'],'source_changed_during_calibration')
            image=cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            require(image is not None and image.shape[0]*image.shape[1]<=20_000_000,'Unsupported native image')
            h,w=image.shape[:2];size=(w,h);native_hash=sha(image.tobytes())
            d={**row,'split':split,'image_sha256':sha(encoded),'native_bgr_sha256':native_hash,
                'native_size':[w,h],'pixel_orientation':'encoded_pixels_no_exif_or_metadata_rotation',
                'detected':False,'eligible':False,'reasons':[],'corners':None}
            if image_size is None:image_size=size;report['image_size']=[w,h]
            if size!=image_size:d['reasons'].append('mixed_native_resolutions');detections.append(d);continue
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);d['blur_laplacian_variance_proxy']=float(cv2.Laplacian(gray,cv2.CV_64F).var())
            try:corners,mapping=detect(gray,board);d['corner_mapping']=mapping
            except ValueError:corners=None;d['reasons'].append('corner_too_close_to_edge')
            if corners is None:d['reasons'].append('board_not_detected')
            else:
                d.update(detected=True,corners=corners.tolist(),corner_hull_fraction=hull_fraction(corners,size))
                if d['corner_hull_fraction']<.02:d['reasons'].append('board_too_small')
                if native_hash in (train_hashes if split=='train' else heldout_hashes):d['reasons'].append('duplicate_native_pixels_in_split')
                if split=='heldout' and native_hash in train_hashes:d['reasons'].append('same_native_pixels_as_training')
                d['eligible']=not d['reasons'];overlay=image.copy();cv2.drawChessboardCorners(overlay,(board['inner_cols'],board['inner_rows']),corners.reshape(-1,1,2),True)
                overlay_path=overlays/(d['id']+'.jpg');cv2.imwrite(str(overlay_path),overlay,[cv2.IMWRITE_JPEG_QUALITY,92]);d['overlay_path']=str(overlay_path.relative_to(root));d['overlay_sha256']=sha(overlay_path.read_bytes())
            (train_hashes if split=='train' else heldout_hashes).add(native_hash)
            detections.append(d)
        # Heldout duplicates are checked after all TRAIN hashes exist; this never
        # changes TRAIN eligibility or model fitting.
        for d in detections:
            if d['split']=='heldout' and d['native_bgr_sha256'] in train_hashes:
                d['eligible']=False
                if 'same_native_pixels_as_training' not in d['reasons']:d['reasons'].append('same_native_pixels_as_training')
        report['splits']={name:[d['id'] for d in detections if d['split']==name] for name in ['train','heldout']}
        detection_record=save_detections()
        progress('fit',len(detections),len(detections));fit=fit_camera(detections,image_size or (1,1),board,deadline=deadline);report.update(fit)
        if time.monotonic()>deadline:raise TimeoutError('calibration_deadline')
        if admitted_source_sha is not None:require(sha(source.read_bytes())==admitted_source_sha,'source_changed_during_calibration')
        provenance={'detections':detection_record,
            'protocol':{'path':str((out/'protocol.json').relative_to(root)),'sha256':sha((out/'protocol.json').read_bytes())},
            'implementation':{'path':str((out/'implementation.py').relative_to(root)),'sha256':sha(code)}}
        report['provenance']=provenance
        if report['status']=='candidate_passed_checks':
            camera={'format':'explicit-checkerboard-camera-record','version':1,'status':'estimated_from_detected_board',
                **fit['candidate'],'board':board,'source':report['source'],'splits':report['splits'],
                'provenance':provenance,
                'heldout_checks_passed':True,'metric_scale_verified':False,'object_metric_scale_established':False,
                'camera_session_binding':protocol['camera_session_binding'],
                'application_conditions':['Explicitly associate a later capture from the same camera, lens, resolution, pixel orientation, zoom, focus and processing mode.',
                    'Do not silently apply this distorted camera to a pipeline assuming zero distortion. Undistort or use the saved distortion model explicitly.',
                    'Electronic stabilization, rolling shutter, autofocus and non-flat boards may violate this pinhole model.',
                    'A measured square changes board-pose units only; independent object models still have unknown scale.']}
            path=out/'camera-calibration.json';write_json(path,camera);report['camera_record']={'path':str(path.relative_to(root)),'sha256':sha(path.read_bytes())}
    except (TimeoutError,subprocess.TimeoutExpired):report.update(status='timeout',diagnostics=[{'code':'runtime_bound_reached','action':'Use a shorter/lower-resolution board recording; retain this bounded failure.'}])
    except (ValueError,cv2.error) as error:
        code='source_changed_during_calibration' if str(error)=='source_changed_during_calibration' else 'invalid_calibration_input' if isinstance(error,ValueError) else 'opencv_calibration_failed'
        report.update(status='insufficient_evidence',diagnostics=[{'code':code,'action':'Check the explicit board settings and provide an unchanged, clear calibration recording with consistent native image size.'}])
    save_detections()
    report.update(elapsed_s=time.monotonic()-start,completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(out/'report.json',report);write_json(out/'diagnostics.json',{'status':report['status'],'diagnostics':report['diagnostics'],'metrics':report['metrics']})
    progress('complete',len(detections),len(detections))
    files=[p for p in out.rglob('*') if p.is_file() and p.name!='manifest.json']
    manifest={'format':'checkerboard-calibration-artifacts','version':1,'status':report['status'],
        'files':{str(p.relative_to(out)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in sorted(files)},'new_api_calls':0,'published':False}
    write_json(out/'manifest.json',manifest)
    return report

def board_svg(board):
    board=board_spec(board);cols,rows=board['inner_cols']+1,board['inner_rows']+1
    width,height=cols+2,rows+3;size=board['square_size_mm']
    dimensions=f'width="{width*size}mm" height="{height*size}mm"' if size else f'width="{width*80}" height="{height*80}"'
    cells=''.join(f'<rect x="{c+1}" y="{r+1}" width="1" height="1" fill="#000"/>' for r in range(rows) for c in range(cols) if (r+c)%2==0)
    label=f"{board['inner_cols']} × {board['inner_rows']} inner corners; {cols} × {rows} squares"
    scale='Square size unspecified — grid units only' if size is None else f'Each square {size:g} mm — print at100% and measure'
    return f'<svg xmlns="http://www.w3.org/2000/svg" {dimensions} viewBox="0 0 {width} {height}"><title>{label}</title><desc>{scale}</desc><rect width="100%" height="100%" fill="#fff"/>{cells}<text x="1" y="{rows+2.05}" font-family="sans-serif" font-size=".24" fill="#000">{label}</text><text x="1" y="{rows+2.45}" font-family="sans-serif" font-size=".20" fill="#000">{scale}</text></svg>\n'
