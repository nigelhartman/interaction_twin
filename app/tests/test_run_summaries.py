from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from run_summaries import compact_run_summary

ROOT=Path(__file__).resolve().parents[1]


class RunSummaryTests(unittest.TestCase):
    def fixture(self):
        return {'id':'run-example','status':'complete_with_gaps','phase':None,'progress':1,
                'created_at':'date','completed_at':'date','profile':{'id':'meta_sam31_parts','title':'Parts',
                'models':[1]*1000},'request':{'profile':'meta_sam31_parts','case_id':'example','input_path':'private-unused'},
                'source':{'sha256':'a'*64,'probe':{'large':[1]*10000}},'error':None,
                'phases':[{'name':'segment','status':'complete','attempt':1,'elapsed_s':4,
                           'inputs':[1]*10000,'files':[1]*10000,'result':{'huge':[1]*10000}}],
                'artifacts':{'case':'example','interaction':'data/example/result/interaction.json','viewer':'data/example/result/index.html',
                             'component_receipts':[1]*10000,'observed_reconstructions':[{'id':'observed-'+'a'*24,
                             'track_id':'part','status':'complete','workflow_status':'complete_with_gaps','stages':[1]*10000}]},
                'observed_child_status':{'id':'observed-'+'a'*24,'status':'complete','stages':[1]*10000},
                'logs_url':'/files/data/flow-runs/run-example/pipeline.log','physical_accuracy_verified':True}

    def test_compact_ui_contract_drops_receipts_and_does_not_mutate_or_claim_physics(self):
        record=self.fixture();before=deepcopy(record);summary=compact_run_summary(record)
        self.assertEqual(record,before)
        self.assertLess(len(json.dumps(summary)),2000)
        self.assertEqual(summary['fullrecord_url'],'/api/runs/run-example')
        self.assertEqual(summary['artifacts']['interaction'],record['artifacts']['interaction'])
        self.assertEqual(summary['artifacts']['observed_reconstructions'][0]['track_id'],'part')
        self.assertEqual(summary['phases'][0]['name'],'segment')
        self.assertNotIn('result',summary['phases'][0]);self.assertNotIn('files',summary['phases'][0])
        self.assertNotIn('stages',summary['observed_child_status']);self.assertNotIn('models',summary['profile'])
        self.assertNotIn('input_path',summary['request']);self.assertNotIn('physical_accuracy_verified',summary)

    def test_nonfinite_progress_and_invalid_identifiers_fail_safely(self):
        for p in [float('nan'),float('inf'),True,None]:
            record=self.fixture();record['progress']=p;self.assertEqual(compact_run_summary(record)['progress'],0)
        for value in ['../private','abc?key=1','/foo','',None]:
            with self.assertRaises(ValueError):compact_run_summary({'id':value})

    def test_native_gpu_pending_status_remains_visible_without_exporting_job_details(self):
        record = self.fixture()
        record['sam3d_reconstruction_status'] = {
            'provider': 'sam3d_native', 'generated_components': 1,
            'pending_component_ids': ['body', 'lid', '../private'],
            'native_motion_status': 'fresh_native_alignment_required',
            'old_poses_reused': False, 'fallback_backend_used': False,
            'physical_accuracy_verified': False, 'private_runtime_path': '/not-public'}
        status = compact_run_summary(record)['sam3d_reconstruction_status']
        self.assertEqual(status['pending_component_ids'], ['body', 'lid'])
        self.assertEqual(status['generated_components'], 1)
        self.assertNotIn('private_runtime_path', status)
        self.assertFalse(status['physical_accuracy_verified'])

    def test_summary_endpoint_explicit_and_full_api_route_preserved(self):
        spec=importlib.util.spec_from_file_location('summary_workspace',ROOT/'scripts/serve_workspace.py')
        server=importlib.util.module_from_spec(spec);spec.loader.exec_module(server)
        rows=[self.fixture()]
        for route,expected in [('/api/runs',rows),('/api/runs/summary',[compact_run_summary(rows[0])])]:
            handler=server.Handler.__new__(server.Handler);handler.path=route;handler.headers={'Host':'127.0.0.1:8765'}
            from types import SimpleNamespace
            handler.server=SimpleNamespace(admin_auth=SimpleNamespace(authenticated=lambda _:True))
            results=[];handler.json=lambda status,body:results.append((status,body))
            with patch.object(server,'list_runs',return_value=rows):handler._read(False)
            self.assertEqual(results[0][0],200);self.assertEqual(results[0][1]['runs'],expected)
        self.assertIn('files',rows[0]['phases'][0])

    def test_mechanism_status_and_descriptive_export_are_visible_without_private_receipts(self):
        record=self.fixture();record['mechanism_pipeline_status']={'topology':'reused_verified_mechanism_topology',
            'scene':'constrained_scene_with_unknown_connections','object_program':'descriptive_object_program_ready',
            'independent_scene_fallback':False,'physical_accuracy_verified':False,'simulation_ready':False,
            'large_private_receipt':{'values':[0]*10000}}
        record['artifacts']['object_program']='data/object-programs/example/object-program.json'
        summary=compact_run_summary(record)
        self.assertEqual(summary['mechanism_pipeline_status']['scene'],'constrained_scene_with_unknown_connections')
        self.assertFalse(summary['mechanism_pipeline_status']['independent_scene_fallback'])
        self.assertNotIn('large_private_receipt',summary['mechanism_pipeline_status'])
        self.assertEqual(summary['artifacts']['object_program'],record['artifacts']['object_program'])


if __name__=='__main__':unittest.main()
