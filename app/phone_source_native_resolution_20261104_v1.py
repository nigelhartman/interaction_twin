"""Extract actual high-resolution phone source evidence and test feature support.

No generation, upload, GLB edit, pose publication, or manual geometric input.
The high-resolution SAM boundary is explicitly a nearest-neighbour transfer of
the saved low-resolution model mask, not fresh SAM3 inference.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image

import phone_source_regeneration_20261104_v1 as P

ROOT=P.ROOT
OUTPUT=P.OUTPUT/'native-resolution-v2'


def command(argv):
    return subprocess.run(argv,check=True,capture_output=True,timeout=120).stdout


def prepare():
    OUTPUT.mkdir(exist_ok=False)
    plan=json.loads((P.OUTPUT/'regeneration-plan.json').read_text())
    program=json.loads((P.CASE/'flow-procedure-1/interaction.json').read_text())
    normalization=json.loads((P.CASE/'normalization.json').read_text())
    source=P.CASE/'source/original.mp4'
    assert P.ref(source)['sha256']==normalization['original_sha256']
    evidence={r['source_index']:r for r in program['evidence']}
    decoded=json.loads(command(['ffprobe','-v','error','-select_streams','v:0','-show_frames',
        '-show_entries','frame=best_effort_timestamp_time,width,height','-of','json',str(source)]))['frames']
    rows=[]
    for anchor in plan['anchors']:
        row=evidence[anchor['source_index']]
        index,native=min(enumerate(decoded),key=lambda r:abs(float(r[1]['best_effort_timestamp_time'])-row['timestamp_s']))
        assert abs(float(native['best_effort_timestamp_time'])-row['timestamp_s'])<1e-6
        folder=OUTPUT/f'source-{row["source_index"]:06d}'
        folder.mkdir()
        # Same documented Hable colour pipeline as the normalized source, but
        # no spatial downscale, synthesis, inpainting, or superresolution.
        filterchain=(f'select=eq(n\\,{index}),zscale=t=linear:npl=100,format=gbrpf32le,'
            'tonemap=tonemap=hable:desat=0,zscale=p=bt709:t=bt709:m=bt709:r=tv,format=rgb24')
        ffmpeg=normalization['ffmpeg_command'][0]
        assert Path(ffmpeg).is_file()
        argv=[ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-n','-i',str(source),'-map','0:v:0',
            '-an','-vf',filterchain,'-frames:v','1',str(folder/'image.png')]
        command(argv)
        original=np.asarray(Image.open(folder/'image.png').convert('RGB'))
        assert original.shape==(2048,1536,3)
        obs=next(o for o in row['observations']if o['track_id']=='phone_blue')
        maskpath=ROOT/obs['mask_path']
        mask=Image.open(maskpath).convert('L').resize((1536,2048),Image.Resampling.NEAREST)
        mask.save(folder/'mask-transferred.png')
        binary=np.asarray(mask)>0;ys,xs=np.nonzero(binary)
        padding=round(max(xs.max()-xs.min(),ys.max()-ys.min())*.12)
        crop=[max(0,int(xs.min())-padding),max(0,int(ys.min())-padding),
              min(1536,int(xs.max())+1+padding),min(2048,int(ys.max())+1+padding)]
        Image.fromarray(original).crop(crop).save(folder/'phone-crop.png')
        mask.crop(crop).save(folder/'mask-crop.png')
        reference=np.asarray(Image.open(ROOT/row['frame_path']).convert('RGB'))
        shrunk=cv2.resize(original,(720,960),interpolation=cv2.INTER_AREA)
        lowmask=np.asarray(Image.open(maskpath))>0
        mae=float(np.abs(shrunk.astype(float)-reference.astype(float))[lowmask].mean())
        rows.append({'source_index':row['source_index'],'timestamp_s':row['timestamp_s'],
            'native_frame_index_zero_based':index,'source_original':P.ref(source),
            'image':P.ref(folder/'image.png'),'transferred_mask':P.ref(folder/'mask-transferred.png'),
            'crop':P.ref(folder/'phone-crop.png'),'crop_mask':P.ref(folder/'mask-crop.png'),
            'crop_xyxy':crop,'native_phone_bbox_wh':[int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)],
            'lower_resolution_SAM_mask':P.ref(maskpath),'mask_origin':'saved_SAM3.1_nearest_neighbour_transfer_not_new_inference',
            'downscaled_original_vs_normalized_phone_RGB_mean_absolute_difference':mae,
            'argv':argv,'bridge_contract_status':'evidence_only_requires_separately_reviewed_full_resolution_input_adapter'})
    # Feature sufficiency is tested only inside exact saved masks. A successful
    # homography is a 2D diagnostic and cannot establish thickness or contact.
    a,b=rows
    ia=cv2.imread(str(ROOT/a['image']['path']),cv2.IMREAD_GRAYSCALE)
    ib=cv2.imread(str(ROOT/b['image']['path']),cv2.IMREAD_GRAYSCALE)
    ma=cv2.imread(str(ROOT/a['transferred_mask']['path']),cv2.IMREAD_GRAYSCALE)
    mb=cv2.imread(str(ROOT/b['transferred_mask']['path']),cv2.IMREAD_GRAYSCALE)
    ma=cv2.erode(ma,np.ones((7,7),np.uint8));mb=cv2.erode(mb,np.ones((7,7),np.uint8))
    sift=cv2.SIFT_create(nfeatures=3000)
    ka,da=sift.detectAndCompute(ia,ma);kb,db=sift.detectAndCompute(ib,mb)
    good=[]
    if da is not None and db is not None:
        good=[m for m,n in cv2.BFMatcher().knnMatch(da,db,k=2)if m.distance<.7*n.distance]
    homography_inliers=None;homography_fraction=None
    if len(good)>=4:
        _,inliers=cv2.findHomography(np.float32([ka[m.queryIdx].pt for m in good]),
            np.float32([kb[m.trainIdx].pt for m in good]),cv2.USAC_MAGSAC,3)
        homography_inliers=int(inliers.sum())if inliers is not None else 0
        homography_fraction=homography_inliers/len(good)
    result={'format':'interaction-twin-phone-native-resolution-preparation','version':1,
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'implementation':P.ref(Path(__file__)),
        'original':P.ref(source),'normalization':P.ref(P.CASE/'normalization.json'),'generation_plan':P.ref(P.OUTPUT/'regeneration-plan.json'),
        'anchors':rows,'source_geometry_synthesized':False,'source_pixels_superresolved':False,
        'SIFT_diagnostic':{'masked_keypoints':[len(ka),len(kb)],'ratio_threshold':.7,
            'ratio_passing_matches':len(good),'homography_inliers':homography_inliers,
            'homography_inlier_fraction':homography_fraction,'pose_or_depth_recovered':False,
            'interpretation':'Back-face correspondences do not resolve phone thickness. Broad faces are approximately planar; an unconstrained feature reconstruction would need calibrated or jointly estimated camera motion and additional noncoplanar tracks, then independent thickness validation.'},
        'geometry_repaired':False,'generation_or_API_commands_executed':False,'published':False,
        'physical_accuracy_verified':False}
    P.save(OUTPUT/'receipt.json',result)
    return {'receipt':P.ref(OUTPUT/'receipt.json'),'SIFT_diagnostic':result['SIFT_diagnostic']}


if __name__=='__main__':
    print(json.dumps(prepare(),indent=2))
