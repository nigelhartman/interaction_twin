import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np

import observed_geometry as G
from observed_segmentation_input import build


class SegmentationInputTests(unittest.TestCase):
    def fixture(self,root):
        root=Path(root);case=root/'data/case';seg=case/'segmentation/parts';annotation=root/'annotations/model.json'
        (case/'frames').mkdir(parents=True);(case/'source').mkdir();(seg/'raw_masks/part').mkdir(parents=True);annotation.parent.mkdir()
        (case/'source/video.mp4').write_bytes(b'synthetic fixture, never decoded or inferred')
        sha=G.digest(case/'source/video.mp4')
        image=np.ones((30,40,3),np.uint8)*100;mask=np.zeros((30,40),np.uint8);mask[5:25,5:35]=255
        cv2.imwrite(str(case/'frames/000001.jpg'),image);cv2.imwrite(str(seg/'raw_masks/part/000001.png'),mask)
        annotations={'manual_seed_boxes_used':False,'tracks':[{'id':'part','label':'test part','candidate_entity_ids':['candidate']}]}
        annotation.write_text(json.dumps(annotations))
        manifest={'source':{'file':'source/video.mp4','sha256':sha,'original_sha256':sha,'video':{'duration_s':1.}}}
        (case/'manifest.json').write_text(json.dumps(manifest))
        (case/'frames/index.json').write_text(json.dumps({'frames':[{'index':1,'file':'frames/000001.jpg','timestamp_s':0.,'width':40,'height':30}]}))
        rows={'tracks':[{'track_id':'part','frames':[{'sample_index':1,'timestamp_s':0.,'raw_mask_file':'raw_masks/part/000001.png','raw_mask_sha256':G.digest(seg/'raw_masks/part/000001.png'),
                'source_frame_sha256':G.digest(case/'frames/000001.jpg'),'status':'candidate_visible','association':{'accepted':True},'zero_placeholder_for_missing_or_withheld':False}]}]}
        (seg/'tracks.json').write_text(json.dumps(rows))
        metadata={'status':'complete','source_sha256':sha,'config_sha256':G.digest(annotation),'manual_seed_boxes_used':False,
                  'scope':'saved_model_masks','model':'sam-3.1','provider':'Meta Model API','device':'hosted'}
        (seg/'metadata.json').write_text(json.dumps(metadata))
        audit={'status':'complete_candidate_visual_audit','source_sha256':sha,'source_track_index_sha256':G.digest(seg/'tracks.json'),
               'result':{'observations':[{'track_id':'part','sample_index':1,'verdict':'reject_wrong_or_lost_mask'}]}}
        (seg/'audit.json').write_text(json.dumps(audit))
        return root,case,seg,annotation

    def test_fresh_audit_reject_is_preserved_not_relabeled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,case,seg,annotation=self.fixture(tmp)
            doc,metadata=build(root,case,seg,annotation,root/'data/prepared',seg/'audit.json')
            self.assertEqual(doc['evidence'][0]['observations'][0]['audit_status'],'reject_wrong_or_lost_mask')
            frames,excluded=G.select_views(doc,'part',root)
            self.assertEqual(frames,[])
            self.assertEqual(len(excluded),1)
            self.assertTrue(metadata['fresh_visual_audit_bound'])

    def test_without_audit_masks_stay_unaudited(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,case,seg,annotation=self.fixture(tmp)
            doc,metadata=build(root,case,seg,annotation,root/'data/prepared')
            self.assertEqual(doc['evidence'][0]['observations'][0]['audit_status'],'not_model_audited')
            self.assertFalse(metadata['fresh_visual_audit_bound'])
            self.assertFalse(doc['physical_identity_verified'])

    def test_wrong_source_or_config_identity_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,case,seg,annotation=self.fixture(tmp)
            annotation.write_text(annotation.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'source/config identity mismatch'):
                build(root,case,seg,annotation,root/'data/prepared')

    def test_native_association_withheld_does_not_create_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,case,seg,annotation=self.fixture(tmp)
            path=seg/'tracks.json';rows=json.loads(path.read_text());rows['tracks'][0]['frames'][0]['association']['accepted']=False;path.write_text(json.dumps(rows))
            doc,metadata=build(root,case,seg,annotation,root/'data/prepared')
            self.assertEqual(doc['evidence'],[])
            self.assertEqual(metadata['included_observations'],0)


if __name__=='__main__':unittest.main()
