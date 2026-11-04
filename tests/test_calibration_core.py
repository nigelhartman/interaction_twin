"""Calibration mathematics and local provenance checks using labelled fixtures.

Analytic corners and rendered boards here are software fixtures, never evidence
that any user's physical camera has been calibrated.
"""
from __future__ import annotations
import copy, hashlib, json, shutil, subprocess, tempfile, time, unittest
from pathlib import Path
from unittest import mock
import cv2
import numpy as np
import calibration_core as C


def analytic_views():
    board=C.board_spec({'inner_rows':6,'inner_cols':9})
    obj=C.object_points(board)
    K=np.array([[940.,0,640.],[0,915.,480.],[0,0,1.]])
    dist=np.array([-.06,.012,.0005,-.0003,0.])
    rng=np.random.default_rng(40);views=[]
    for i in range(36):
        r=np.array([(i%6-2.5)*.16,(i//6-2.5)*.14,(i%3-1)*.08])
        rotation=cv2.Rodrigues(r)[0]
        cx=[330,640,950][i%3];cy=[220,470,730][(i//3)%3];z=17.+(i%4)*1.2
        center=np.array([(cx-640)/940*z,(cy-480)/915*z,z])
        t=center-rotation@np.array([4.,2.5,0.])
        corners=cv2.projectPoints(obj,r,t,K,dist)[0].reshape(-1,2)+rng.normal(0,.08,(len(obj),2))
        assert np.all(corners>0) and np.all(corners<np.array([1280,960]))
        views.append({'id':f'fixture_{i:02d}','fixture':'synthetic_analytic_projection',
            'split':'heldout' if i%4==0 else 'train','eligible':True,'detected':True,
            'corners':corners.tolist(),'reasons':[]})
    return board,K,dist,views


class AnalyticCalibrationTests(unittest.TestCase):
    def test_known_camera_recovered_without_physical_accuracy_claim(self):
        board,K,dist,views=analytic_views();result=C.fit_camera(views,(1280,960),board)
        self.assertEqual(result['status'],'candidate_passed_checks')
        recovered=np.array(result['candidate']['K'])
        np.testing.assert_allclose(recovered[:2,:],K[:2,:],atol=.4,rtol=0)
        np.testing.assert_allclose(result['candidate']['distortion_coefficients'],dist,atol=.004,rtol=0)
        self.assertEqual(result['metrics']['heldout_passed_views'],9)
        self.assertEqual(result['metrics']['eligible_train_views'],27)
        self.assertGreaterEqual(result['metrics']['training_centroid_cells'],3)
        self.assertFalse(result['candidate']['physical_accuracy_verified'])
        for row in result['heldout_views']:
            self.assertTrue(set(row['anchor_corner_ids']).isdisjoint(row['scored_corner_ids']))
            self.assertEqual(set(row['anchor_corner_ids'])|set(row['scored_corner_ids']),set(range(54)))

    def test_metric_size_changes_only_board_translation_units(self):
        board,_,_,views=analytic_views();unit=C.fit_camera(views,(1280,960),board)
        measured=C.fit_camera(views,(1280,960),{**board,'square_size_mm':25.})
        np.testing.assert_allclose(unit['candidate']['K'],measured['candidate']['K'],atol=1e-5,rtol=0)
        a=unit['candidate']['training_board_poses'][0];b=measured['candidate']['training_board_poses'][0]
        np.testing.assert_allclose(np.array(a['translation'])*25.,b['translation'],atol=1e-5,rtol=0)
        self.assertEqual(a['translation_units'],'grid_units');self.assertEqual(b['translation_units'],'mm')
        self.assertFalse(b['metric_scale_verified'])
        self.assertFalse(C.POLICY['object_metric_scale_established'])

    def test_heldout_scoring_corners_cannot_change_intrinsic_fit(self):
        board,_,_,views=analytic_views()
        corrupted=copy.deepcopy(views)
        scored=np.array([(r+c)%2==1 for r in range(6) for c in range(9)])
        for view in corrupted:
            if view['split']=='heldout':
                points=np.array(view['corners']);points[scored]+=np.array([30.,-20.]);view['corners']=points.tolist()
        with mock.patch.object(cv2,'calibrateCameraExtended',wraps=cv2.calibrateCameraExtended) as fit:
            original=C.fit_camera(views,(1280,960),board)
            bad=C.fit_camera(corrupted,(1280,960),board)
        self.assertEqual(fit.call_count,2)
        for before,after in zip(fit.call_args_list[0].args[1],fit.call_args_list[1].args[1]):
            np.testing.assert_array_equal(before,after)
        np.testing.assert_allclose(original['candidate']['K'],bad['candidate']['K'],atol=1e-5,rtol=0)
        np.testing.assert_allclose([v['translation'] for v in original['candidate']['training_board_poses']],[v['translation'] for v in bad['candidate']['training_board_poses']],atol=1e-5,rtol=0)
        self.assertEqual(bad['status'],'rejected_validation')
        self.assertEqual(bad['metrics']['heldout_passed_views'],0)
        self.assertEqual(len(bad['heldout_views']),9)
        self.assertTrue(all(row['anchor_rms_px']<.3 and row['scored_median_px']>30 for row in bad['heldout_views']))

    def test_frontoparallel_translations_are_rejected_before_camera_fit(self):
        board,K,_,views=analytic_views();obj=C.object_points(board)
        for i,view in enumerate(views):
            z=17.+i%4;cx=[330,640,950][i%3];cy=[220,470,730][(i//3)%3]
            t=np.array([(cx-640)/940*z-4,(cy-480)/915*z-2.5,z])
            view['corners']=cv2.projectPoints(obj,np.zeros(3),t,K,np.zeros(5))[0].reshape(-1,2).tolist()
        with mock.patch.object(cv2,'calibrateCameraExtended',side_effect=AssertionError('degenerate fit attempted')):
            bad=C.fit_camera(views,(1280,960),board)
        self.assertEqual(bad['status'],'rejected_degenerate_views');self.assertIsNone(bad['candidate'])
        self.assertLess(bad['metrics']['homography_diversity']['fifth_to_first_ratio'],1e-4)

    def test_one_prospective_train_refit_preserves_outlier_identity(self):
        board,_,_,views=analytic_views();rng=np.random.default_rng(2)
        noisy=next(v for v in views if v['split']=='train')
        noisy['corners']=(np.array(noisy['corners'])+rng.normal(0,4,(54,2))).tolist()
        result=C.fit_camera(views,(1280,960),board)
        self.assertTrue(result['metrics']['training_refit_performed'])
        self.assertIn(noisy['id'],result['metrics']['training_outlier_view_ids'])
        self.assertEqual(result['status'],'candidate_passed_checks')
        self.assertEqual(next(v for v in result['training_views'] if v['id']==noisy['id'])['used'],False)
        self.assertEqual(len(views),36)

    def test_nonfinite_solver_result_remains_serializable_failure(self):
        board,_,_,views=analytic_views()
        with mock.patch.object(cv2,'calibrateCameraExtended',return_value=(float('nan'),np.eye(3),np.zeros(5),[],[],np.zeros(18),np.zeros(6),np.zeros(27))):
            result=C.fit_camera(views,(1280,960),board)
        self.assertEqual(result['status'],'rejected_degenerate_views');self.assertIsNone(result['candidate'])
        json.dumps(result,allow_nan=False)

    def test_heldout_missing_detection_stays_in_denominator(self):
        board,_,_,views=analytic_views()
        for view in [v for v in views if v['split']=='heldout'][:3]:
            view.update(eligible=False,detected=False,corners=None,reasons=['board_not_detected'])
        result=C.fit_camera(views,(1280,960),board)
        self.assertEqual(result['status'],'rejected_validation')
        self.assertEqual(result['metrics']['heldout_passed_views'],6)
        self.assertAlmostEqual(result['metrics']['heldout_pass_fraction'],6/9)
        self.assertEqual(len(result['heldout_views']),9)

    def test_deadline_stops_before_numeric_solver(self):
        board,_,_,views=analytic_views()
        with self.assertRaises(TimeoutError):C.fit_camera(views,(1280,960),board,deadline=time.monotonic()-1)


class NativeDetectionAndArtifactTests(unittest.TestCase):
    def test_complete_rendered_fixture_writes_bound_camera_record(self):
        # This is a synthetic image pipeline test, not a real board/camera run.
        K=np.array([[940.,0,640.],[0,915.,480.],[0,0,1.]])
        texture=np.full((448,640),255,np.uint8)
        for r in range(7):
            for c in range(10):
                if (r+c)%2==0:texture[r*64:(r+1)*64,c*64:(c+1)*64]=0
        outer=np.array([[-1.,-1,0],[9.,-1,0],[9.,6,0],[-1.,6,0]])
        texel_corners=np.array([[0.,0],[640.,0],[640.,448.],[0.,448.]],np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();source=root/'data'/'synthetic-images';source.mkdir(parents=True)
            for i in range(36):
                r=np.array([(i%6-2.5)*.16,(i//6-2.5)*.14,(i%3-1)*.08]);rotation=cv2.Rodrigues(r)[0]
                cx=[350,640,930][i%3];cy=[240,470,710][(i//3)%3];z=19.+i%4*1.2
                t=np.array([(cx-640)/940*z,(cy-480)/915*z,z])-rotation@np.array([4.,2.5,0])
                target=cv2.projectPoints(outer,r,t,K,np.zeros(5))[0].reshape(-1,2).astype(np.float32)
                image=cv2.warpPerspective(texture,cv2.getPerspectiveTransform(texel_corners,target),(1280,960),borderValue=240)
                cv2.imwrite(str(source/f'fixture-{i:02d}.png'),image)
            out=root/'data'/'synthetic-result';report=C.run(source,out,root=root,board={'inner_rows':6,'inner_cols':9},max_frames=36)
            self.assertEqual(report['status'],'candidate_passed_checks')
            camera_path=root/report['camera_record']['path'];camera=json.loads(camera_path.read_text())
            self.assertEqual(report['camera_record']['sha256'],hashlib.sha256(camera_path.read_bytes()).hexdigest())
            np.testing.assert_allclose(camera['K'],K,atol=3,rtol=0)
            self.assertEqual(camera['image_size'],[1280,960]);self.assertEqual(camera['board']['board_pose_output_units'],'grid_units')
            self.assertFalse(camera['physical_accuracy_verified']);self.assertFalse(camera['object_metric_scale_established'])
            for record in camera['provenance'].values():self.assertEqual(record['sha256'],hashlib.sha256((root/record['path']).read_bytes()).hexdigest())
            for path,record in json.loads((out/'manifest.json').read_text())['files'].items():
                self.assertEqual(record['sha256'],hashlib.sha256((out/path).read_bytes()).hexdigest())

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'Prepared media tools required')
    def test_native_video_pts_and_failed_source_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();frames=root/'data'/'frames';frames.mkdir(parents=True)
            for i in range(12):cv2.imwrite(str(frames/f'{i:02d}.png'),np.full((120,160,3),i*3+80,np.uint8))
            movie=root/'data'/'synthetic-blank.mp4'
            subprocess.run([shutil.which('ffmpeg'),'-v','error','-protocol_whitelist','file,pipe','-framerate','3','-i',str(frames/'%02d.png'),'-c:v','mpeg4',str(movie)],check=True,capture_output=True,timeout=20)
            out=root/'data'/'video-result';report=C.run(movie,out,root=root,board={'inner_rows':6,'inner_cols':9},max_frames=12)
            self.assertEqual(report['status'],'insufficient_evidence');self.assertIsNone(report['camera_record'])
            self.assertEqual(report['source']['sha256'],hashlib.sha256(movie.read_bytes()).hexdigest())
            self.assertEqual(report['source']['decoder_protocol_whitelist'],['file','pipe'])
            views=json.loads((out/'detections.json').read_text())['views']
            pts=[v['timestamp_s'] for v in views];self.assertEqual(len(pts),12)
            self.assertTrue(all(b>a for a,b in zip(pts,pts[1:])))
            np.testing.assert_allclose(pts,np.arange(12)/3,atol=1e-6,rtol=0)
            self.assertEqual([v['source_index'] for v in views],list(range(12)))
            self.assertTrue(all(v['native_size']==[160,120] for v in views))

    def test_resized_detection_refines_in_original_pixel_coordinates(self):
        board=C.board_spec({'inner_rows':6,'inner_cols':9});image=np.full((1803,2401),255,np.uint8)
        x,y,side=500,380,130
        for r in range(7):
            for c in range(10):
                if (r+c)%2==0:image[y+r*side:y+(r+1)*side,x+c*side:x+(c+1)*side]=0
        corners,mapping=C.detect(image,board)
        self.assertEqual(len(corners),54)
        self.assertEqual(mapping['native_size'],[2401,1803]);self.assertEqual(mapping['detection_size'],[1600,1201])
        expected=np.array([[x+c*side-.5,y+r*side-.5] for r in range(1,7) for c in range(1,10)])
        distance=np.linalg.norm(corners[:,None,:]-expected[None,:,:],axis=2)
        self.assertLess(float(distance.min(axis=1).max()),.15)
        self.assertNotEqual(mapping['scale_x'],mapping['scale_y'])

    def test_no_board_directory_preserves_frozen_splits_hashes_and_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();source=root/'data'/'blank';source.mkdir(parents=True)
            for i in range(12):cv2.imwrite(str(source/f'{i:02d}.png'),np.full((120,160,3),i+100,np.uint8))
            out=root/'data'/'calibration-failure'
            result=C.run(source,out,root=root,board={'inner_rows':6,'inner_cols':9},max_frames=12)
            self.assertEqual(result['status'],'insufficient_evidence');self.assertIsNone(result['camera_record'])
            self.assertFalse((out/'camera-calibration.json').exists())
            detections=json.loads((out/'detections.json').read_text())['views']
            self.assertEqual([v['id'] for v in detections if v['split']=='heldout'],['view_0000','view_0004','view_0008'])
            self.assertTrue(all(v['corners'] is None and v['timestamp_s'] is None for v in detections))
            for view in detections:self.assertEqual(view['image_sha256'],hashlib.sha256((root/view['path']).read_bytes()).hexdigest())
            manifest=json.loads((out/'manifest.json').read_text())
            for path,record in manifest['files'].items():self.assertEqual(record['sha256'],hashlib.sha256((out/path).read_bytes()).hexdigest())
            self.assertEqual(json.loads((out/'progress.json').read_text())['phase'],'complete')
            with self.assertRaises(ValueError):C.run(source,out,root=root,board={'inner_rows':6,'inner_cols':9})

    def test_limits_and_symlinks_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();source=root/'data'/'source';source.mkdir(parents=True)
            alias=root/'data'/'alias';alias.symlink_to(source,target_is_directory=True)
            for kwargs in [{'max_frames':51},{'max_seconds':121}]:
                with self.assertRaises(ValueError):C.run(source,root/'data'/'bad',root=root,board={'inner_rows':6,'inner_cols':9},**kwargs)
            with self.assertRaises(ValueError):C.run(alias,root/'data'/'bad',root=root,board={'inner_rows':6,'inner_cols':9})
            self.assertFalse((root/'data'/'bad').exists())

    def test_network_playlist_disguised_as_movie_rejected_and_input_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();source=root/'data'/'remote.mp4';source.parent.mkdir();source.write_text('#EXTM3U\nhttps://example.invalid/private\n')
            probe={'format':{'format_name':'hls','duration':'1.0'},'frames':[{'best_effort_timestamp_time':'0'}]}
            calls=[]
            def command(argv,deadline):calls.append(argv);return json.dumps(probe).encode()
            with mock.patch.object(C,'_command',side_effect=command),mock.patch.object(C.shutil,'which',side_effect=lambda x:'/fixed/'+x):
                result=C.run(source,root/'data'/'result',root=root,board={'inner_rows':6,'inner_cols':9})
            self.assertEqual(result['status'],'insufficient_evidence');self.assertEqual(len(calls),1)
            self.assertEqual(calls[0][calls[0].index('-protocol_whitelist')+1],'file,pipe')
            self.assertEqual(calls[0][calls[0].index('-format_whitelist')+1],'mov,matroska,webm,avi')
            self.assertEqual(result['source']['sha256'],hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertIsNone(result['camera_record'])

    def test_source_mutation_cannot_promote_camera_or_replace_admitted_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();source=root/'data'/'source.mp4';source.parent.mkdir();source.write_bytes(b'original fixture bytes')
            admitted=hashlib.sha256(source.read_bytes()).hexdigest()
            def changed(*args):
                source.write_bytes(b'changed fixture bytes')
                return [],{'kind':'video','path':'data/source.mp4','sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
            with mock.patch.object(C,'_sources',side_effect=changed):
                result=C.run(source,root/'data'/'result',root=root,board={'inner_rows':6,'inner_cols':9})
            self.assertEqual(result['status'],'insufficient_evidence');self.assertIsNone(result['camera_record'])
            self.assertEqual(result['source']['sha256'],admitted)
            self.assertEqual(result['diagnostics'][0]['code'],'source_changed_during_calibration')

    def test_svg_states_inner_corner_and_scale_semantics(self):
        import xml.etree.ElementTree as ET
        grid=C.board_svg({'inner_rows':6,'inner_cols':9});root=ET.fromstring(grid)
        self.assertEqual(root.attrib['viewBox'],'0 0 12 10');self.assertNotIn('mm',root.attrib['width'])
        self.assertIn('9 × 6 inner corners; 10 × 7 squares',grid)
        self.assertIn('grid units only',grid)
        measured=C.board_svg({'inner_rows':6,'inner_cols':9,'square_size_mm':20})
        self.assertEqual(ET.fromstring(measured).attrib['width'],'240.0mm')
        self.assertIn('print at100% and measure',measured)
        for value in [True,0,-1,float('nan')]:
            with self.assertRaises(ValueError):C.board_spec({'inner_rows':6,'inner_cols':9,'square_size_mm':value})

if __name__=='__main__':unittest.main()
