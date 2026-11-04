"""Synthetic no-inference motion registry boundary and exact saved-pose tests."""
import copy,json
from pathlib import Path
import unittest

import motion_catalog as M
from motion_lab import normalize,sha
from tests.test_motion_lab import MotionLabContracts,save


class MotionCatalogContracts(MotionLabContracts):
    # Parent fixture creates no real recording/model and never enters an app index.
    def registered(self):
        root,path,raw=self.fixture();program_path=root/'data/interaction.json';program=json.loads(program_path.read_text())
        program['source']['path']='data/source.mp4';program['assets']=[{'track_id':'part','path':'data/asset.glb','sha256':raw['asset']['sha256']}]
        times=[1.333,2.,3.]
        baseline_path=root/'data/baseline.json';baseline=json.loads(baseline_path.read_text())
        for j,time in enumerate(times):
            program['evidence'][j].update(id=f'source_{j+1:06d}',timestamp_s=time)
            baseline['frames'][j]['timestamp_s']=time;raw['episodes'][0]['frames'][j]['timestamp_s']=time
            raw['comparison']['exact_pts_rows'][j]['timestamp_s']=time
        save(program_path,program);save(baseline_path,baseline)
        raw['interaction_sha256']=sha(program_path);raw['baseline']['sha256']=sha(baseline_path)
        episode=raw['episodes'][0];episode.update(start_s=times[0],end_s=times[-1],reference_timestamp_s=times[0])
        for name in ['plan','audit_binding']:
            p=root/'data'/f'{name}.json';save(p,{'synthetic':True});raw[name]={'path':f'data/{name}.json','sha256':sha(p)}
        save(path,raw);view=normalize(root,path);view_path=root/'data/viewer.json';save(view_path,view)
        index=root/M.INDEX;index.parent.mkdir();record={'experiment_id':'fixture-motion-v1','interaction_path':'data/interaction.json','interaction_sha256':sha(program_path),
            'kind':'short_interval_original_samples_v1','raw':{'path':'data/interval-motion.json','sha256':sha(path)},'normalized':{'path':'data/viewer.json','sha256':sha(view_path)}}
        save(index,{'format':'interaction-motion-catalog','version':1,'experiments':[record]})
        context={'interaction_path':'data/interaction.json','document':program,'program_sha256':sha(program_path)}
        return root,M.MotionCatalog(root),record,context

    def test_registered_descriptor_preserves_exact_raw_pose_and_withheld_absence(self):
        root,catalog,record,ctx=self.registered();result=catalog.describe(record['experiment_id'],twin_id='fixture--v1',**ctx)
        frames=result['motion']['episodes'][0]['frames'];self.assertEqual(frames[1]['pose'],{'R':[[1,0,0],[0,1,0],[0,0,1]],'t':[0,0,3]})
        self.assertIsNone(frames[2]['pose']);self.assertFalse(result['policy']['pose_interpolation']);self.assertFalse(result['policy']['simulation_ready'])
        self.assertEqual(result['inspection_source']['program_evidence_id'],'source_000002')
        data=catalog.artifact(record['experiment_id'],'raw',**ctx)
        self.assertEqual(data,(root/'data/interval-motion.json').read_bytes())
        self.assertEqual(catalog.list(twin_id='fixture--v1',**ctx)['experiments'][0]['experiment_id'],'fixture-motion-v1')

    def test_unknown_id_and_arbitrary_paths_fail_before_artifact_loading(self):
        _,catalog,_,ctx=self.registered()
        for value in ['../../secret','https://host/item','data/interval-motion.json','file:///tmp/x','unknown-id']:
            with self.subTest(value=value),self.assertRaises(ValueError):catalog.load(value,**ctx)
        with self.assertRaises(ValueError):catalog.artifact('fixture-motion-v1','../../source',**ctx)
        with self.assertRaises(ValueError):catalog.load('fixture-motion-v1',**(ctx|{'program_sha256':'0'*64}))

    def test_source_corruption_and_symlink_are_rejected(self):
        root,catalog,_,ctx=self.registered();(root/'data/source.jpg').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):catalog.load('fixture-motion-v1',**ctx)
        root,catalog,record,ctx=self.registered();target=root/'data/viewer.json';target.unlink();target.symlink_to(root/'data/interval-motion.json')
        with self.assertRaises(ValueError):catalog.load('fixture-motion-v1',**ctx)

    def test_repinning_tampered_normalized_pose_cannot_bypass_raw_comparison(self):
        root,catalog,_,ctx=self.registered();path=root/'data/viewer.json';view=json.loads(path.read_text());view['episodes'][0]['frames'][1]['pose']['t'][0]=.4;save(path,view)
        index=root/M.INDEX;registry=json.loads(index.read_text());registry['experiments'][0]['normalized']['sha256']=sha(path);save(index,registry)
        with self.assertRaisesRegex(ValueError,'pose was synthesized'):catalog.load('fixture-motion-v1',**ctx)

    def test_duplicate_registry_and_unsafe_extra_receipt_fail_closed(self):
        root,catalog,_,ctx=self.registered();index=root/M.INDEX;registry=json.loads(index.read_text());registry['experiments']*=2;save(index,registry)
        with self.assertRaises(ValueError):catalog.list(twin_id='fixture--v1',**ctx)
        root,catalog,_,ctx=self.registered();view_path=root/'data/viewer.json';view=json.loads(view_path.read_text());extra=root/'data/unrelated.json';save(extra,{'not_source':True});view['artifact_receipts'].append({'path':'data/unrelated.json','sha256':sha(extra)});save(view_path,view)
        index=root/M.INDEX;registry=json.loads(index.read_text());registry['experiments'][0]['normalized']['sha256']=sha(view_path);save(index,registry)
        with self.assertRaisesRegex(ValueError,'unrelated files'):catalog.load('fixture-motion-v1',**ctx)

    def test_absent_registry_is_empty_without_any_model_or_write(self):
        root,catalog,_,ctx=self.registered();(root/M.INDEX).unlink()
        self.assertEqual(catalog.list(twin_id='fixture--v1',**ctx)['experiments'],[])


if __name__=='__main__':unittest.main()
