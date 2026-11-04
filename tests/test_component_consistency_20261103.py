"""Full-frame correction is model-only, source-bound and compatible with selectors."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
C=load('component_consistency','scripts/stage2_component_consistency_20261103.py')
D=load('component_consistency_driver','scripts/run_bottle_component_consistency_20261103_v2.py')


def proposal():
    return {'summary':'One continuous container body; prior separation unsupported.',
        'tracks':[{'id':'container_body','label':'container body','description':'continuous visible body','first_sample':1,'last_sample':3,
            'seeds':[{'sample_index':1,'timestamp_s':0.,'box_xyxy_1000':[100,100,600,800],'reason':'whole visible body'}],
            'uncertainties':['Unseen surfaces unknown']}],
        'actions':[{'description':'container moves','participant_track_ids':['container_body'],'evidence_samples':[1,3],
            'uncertainties':['No proven separation']}],'uncertainties':['Tether/contact unknown']}


class ComponentConsistencyTests(unittest.TestCase):
    def test_selection_is_generic_temporal_and_prior_action_evidence_bounded(self):
        rows=[{'index':n,'timestamp_s':n/1.5}for n in range(1,87)]
        old={'actions':[{'evidence_samples':[2,12,20]},{'evidence_samples':[30,32,33]},
                         {'evidence_samples':[73,74,76,77]},{'evidence_samples':[80,81,82,83]}]}
        ids=C.selected_indices(rows,old)
        self.assertLessEqual(len(ids),12);self.assertIn(1,ids);self.assertIn(86,ids)
        for n in [12,32,76,82]:self.assertIn(n,ids)

    def test_seed_from_unshown_full_frame_is_rejected_before_localization(self):
        rows=[{'index':n,'timestamp_s':float(n-1),'width':96,'height':128}for n in range(1,4)]
        result=proposal();result['tracks'][0]['seeds'][0].update(sample_index=2,timestamp_s=1.)
        with self.assertRaisesRegex(ValueError,'no supplied full-frame evidence'):C.validate_supported_seeds(result,rows,[1,3])

    def test_full_model_revision_localization_and_canonical_selection_contract(self):
        class Fake:
            calls=0
            def json_result(self,prompt,schema,images,*,output,**kwargs):
                self.calls+=1
                raw=proposal()if 'tracks'in schema['properties']else{'localizations':[{'track_id':'container_body','sample_index':1,
                    'box_xyxy_1000':[110,120,610,790],'reason':'visible complete body','visibility':'visible'}]}
                receipt={'status':'complete_accounted','model':'muse-spark-1.3','request_id':f'fake-{self.calls}',
                    'usage':{'input_tokens':1,'output_tokens':1},'estimated_cost_usd':'0.000010'}
                C.B.write(Path(output)/'meta-request.json',receipt);C.B.write(Path(output)/'response.pending.json',raw)
                return raw,receipt
        with tempfile.TemporaryDirectory()as tmp,patch.object(C,'ROOT',Path(tmp).resolve()):
            root=Path(tmp).resolve();old=root/'data/old';new=root/'data/new';rows=[]
            for case in [old,new]:
                (case/'frames').mkdir(parents=True)
                local=[]
                for n in range(1,4):
                    Image.new('RGB',(96,128),(n*30,70,100)).save(case/'frames'/f'{n}.png')
                    local.append({'index':n,'timestamp_s':float(n-1),'width':96,'height':128,'file':f'frames/{n}.png'})
                C.B.write(case/'frames/index.json',{'frames':local});rows=local
                C.B.write(case/'manifest.json',{'case_id':case.name,'source':{'original_sha256':'a'*64,'sha256':'b'*64}})
            prior=old/'reasoning/prior';C.B.write(prior/'discovery.json',{'status':'complete_candidate_proposals','result':proposal()})
            C.B.write(prior/'meta-request.json',{'status':'complete_accounted','model':'muse-spark-1.3','request_id':'old-accounted'})
            fake=Fake();result=C.run(new,old,'prior','revision',client=fake)
            self.assertEqual(fake.calls,2);self.assertEqual(result['role'],'rigid_component_visual_discovery_v1')
            self.assertEqual(result['revision_role'],'full_frame_component_consistency_revision_v1')
            self.assertFalse(result['manual_seed_boxes_used']);self.assertFalse(result['prior_proposals_are_observations'])
            self.assertEqual(result['result']['tracks'][0]['seeds'][0]['box_xyxy_1000'],[110,120,610,790])
            import rigid_selection
            cfg=C.B.read(root/result['tracking_config'])
            _,canonical=rigid_selection._canonical(new,cfg,'b'*64,rows)
            self.assertEqual(canonical['result'],result['result'])

    def test_only_exact_discovery_argv_is_replaced(self):
        r=object.__new__(D.SemanticRunner);r.profile={'reasoning_timeout_s':300,'reasoning_model':'muse-spark-1.3','max_output_tokens':16000}
        argv=[str(ROOT/'.venv-sam/bin/python'),str(ROOT/'scripts/run_overnight_stage.py'),'--task-budget-plan',D.V1.BUDGET,
            'discover_rigid_components.py','discover','--case',D.CASE,'--run','flow-discovery-1','--timeout-s','300',
            '--model','muse-spark-1.3','--max-output-tokens','16000']
        with patch.object(D.P.Runner,'command')as command:
            r.command(argv,300);actual=command.call_args.args[0]
            self.assertEqual(actual[1],str(D.ADAPTER));self.assertIn('--prior-case',actual);self.assertEqual(actual[-1],'--allow-paid-api')
        argv[7]='wrong-case'
        with self.assertRaisesRegex(ValueError,'reviewed fixed'):r.command(argv,300)


if __name__=='__main__':unittest.main()
