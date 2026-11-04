"""Once-only nested admission with real flock, inert transport and no credentials."""
import concurrent.futures
from decimal import Decimal
import hashlib
import json
import importlib.util
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock,patch

import mechanism_pass_budget as B
import mechanism_pipeline as M
from increment_budget import IncrementMetaClient,remaining_increment
from meta_api import MetaClient,MetaError
from overnight_budget import OvernightMetaClient,PLAN as OUTER


class MechanismPassBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        for name in [OUTER,B.HISTORICAL_PLAN]:
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((B.ROOT/name).read_bytes())
        self.status={'settled_usd':'7.000000','reserved_usd':'4.038410','cap_usd':'100.000000','per_request_cap_usd':'2.000000','blocked':False}
        self.plan_ref=B.create_once(self.status,root=self.root)
        pin=patch.object(B,'APPROVED_SHA256',self.plan_ref['sha256']);pin.start();self.addCleanup(pin.stop)

    def client(self,*,dispatch_guard=None):
        status=self.status
        class FakeOuter(OvernightMetaClient):
            def __init__(self,*,root,dispatch_guard,**kwargs):
                self.project_root=Path(root);self.dispatch_guard=dispatch_guard;self.ledger=Mock(status=lambda:dict(status))
        return B.scoped_client_class(FakeOuter,root=self.root)(dispatch_guard=dispatch_guard)

    def test_once_only_creation_preserves_original_plan_parent_and_holds(self):
        original={name:(self.root/name).read_bytes()for name in [OUTER,B.HISTORICAL_PLAN,B.PLAN]}
        for status in [{**self.status,'settled_usd':'8.000000'}, {**self.status,'settled_usd':'0.000000','reserved_usd':'0.000000'}]:
            with self.assertRaisesRegex(ValueError,'no automatic renewal'):
                B.create_once(status,root=self.root)
        for name,raw in original.items():self.assertEqual((self.root/name).read_bytes(),raw)
        value=B.load(self.root);self.assertEqual(value['budget_baseline']['reserved_usd'],'4.038410')
        self.assertLess(remaining_increment(json.loads(original[B.HISTORICAL_PLAN]),self.status),Decimal('2'))
        self.assertEqual(remaining_increment(value,self.status),Decimal('5'))

    def test_changed_parent_or_new_plan_is_not_a_new_allowance(self):
        for name in [OUTER,B.PLAN,B.HISTORICAL_PLAN]:
            p=self.root/name;raw=p.read_bytes();d=json.loads(raw);d['changed_after_approval']=True;p.write_text(json.dumps(d))
            with self.assertRaises(ValueError):B.load(self.root)
            p.write_bytes(raw)
        self.assertEqual(B.load(self.root)['additional_estimated_api_ceiling_usd'],'5.000000')

    def test_direct_unwrapped_cli_and_profile_override_fail_before_preparation(self):
        profile={'mechanism_pipeline_mode':M.MODE,'mechanism_task_budget_plan':B.PLAN,'overnight_task_budget_plan':OUTER}
        self.assertEqual(M.configuration(profile,root=self.root)['task_budget_plan'],B.PLAN)
        for key in ['mechanism_task_budget_plan','overnight_task_budget_plan']:
            with self.assertRaises(ValueError):M.configuration({**profile,key:'output/forged-plan.json'},root=self.root)
        with self.assertRaisesRegex(ValueError,'wrapper'):B.scoped_client_class(MetaClient,root=self.root)
        p=self.root/'data/program.json';p.parent.mkdir();p.write_text('{}')
        with patch('mechanism_reasoning.prepare')as prepare:
            with self.assertRaisesRegex(ValueError,'wrapper'):
                M.topology(p,root=self.root,task_budget_plan=B.PLAN,overnight_task_budget_plan=OUTER,allow_paid_api=True)
            prepare.assert_not_called()

    def test_suballocation_and_parent_exhaustion_stop_before_inert_transport(self):
        client=self.client()
        with patch.object(IncrementMetaClient,'request',return_value='inert')as transport:
            self.assertEqual(client.request('fixture'),'inert')
            self.status['settled_usd']='10.010000'
            with self.assertRaisesRegex(MetaError,'new_mechanism_suballocation'):client.request('fixture')
            self.assertEqual(transport.call_count,1)
            self.status['settled_usd']='25.000000'
            with self.assertRaisesRegex(MetaError,'overnight_task_estimate'):client.request('fixture')
            self.assertEqual(transport.call_count,1)

    def test_composed_lease_runs_inside_one_outer_lock_without_deadlock(self):
        guard=Mock();client=self.client(dispatch_guard=guard)
        with patch.object(IncrementMetaClient,'request',return_value='inert')as transport:
            thread=threading.Thread(target=lambda:client.request('fixture'),daemon=True);thread.start();thread.join(2)
            self.assertFalse(thread.is_alive(),'Nested flock deadlock')
            guard.assert_called_once();transport.assert_called_once()
        guard=Mock(side_effect=ValueError('lease lost'));client=self.client(dispatch_guard=guard)
        with patch.object(IncrementMetaClient,'request')as transport,self.assertRaisesRegex(ValueError,'lease lost'):
            client.request('fixture')
        transport.assert_not_called()

    def test_concurrent_admissions_recheck_subbudget_after_each_settlement(self):
        client=self.client();inflight=0;maximum=0
        def transport(*args,**kwargs):
            nonlocal inflight,maximum
            inflight+=1;maximum=max(maximum,inflight);time.sleep(.01)
            self.status['settled_usd']=str(Decimal(self.status['settled_usd'])+1);inflight-=1
            return 'inert'
        def request(_):
            try:return client.request('fixture')
            except MetaError:return 'bounded_stop'
        with patch.object(IncrementMetaClient,'request',side_effect=transport),concurrent.futures.ThreadPoolExecutor(max_workers=6)as pool:
            outcomes=list(pool.map(request,range(6)))
        self.assertEqual(maximum,1);self.assertEqual(outcomes.count('inert'),4);self.assertEqual(outcomes.count('bounded_stop'),2)
        self.assertEqual(self.status['settled_usd'],'11.000000');self.assertEqual(self.status['reserved_usd'],'4.038410')

    def test_actual_admin_stage_uses_new_client_only_inside_outer_wrapper(self):
        import meta_api,increment_budget
        spec=importlib.util.spec_from_file_location('fixture_admin_budget_stage',B.ROOT/'scripts/run_admin_stage.py')
        stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)
        for wrapped in [False,True]:
            base=OvernightMetaClient if wrapped else MetaClient
            selected=[]
            def context(*args,**kwargs):selected.append(kwargs['base_class']);return type('InertContextClient',(),{})
            argv=['run_admin_stage.py','--memory-context-ref','{}','--source-sha256','a'*64,'discover_rigid_components.py']
            with patch.object(stage,'ROOT',self.root),patch.object(stage,'contextual_client',side_effect=context),\
                 patch.object(meta_api,'MetaClient',base),patch.object(increment_budget,'IncrementMetaClient',base),\
                 patch('sys.argv',argv),patch.object(stage.runpy,'run_path')as execute:
                stage.main();execute.assert_called_once()
            if wrapped:self.assertTrue(issubclass(selected[0],OvernightMetaClient));self.assertEqual(selected[0].__name__,'MechanismPassClient')
            else:self.assertEqual(selected,[None],'Outside the new outer pass the historical client must remain selected')

    def test_new_allowance_cannot_resend_an_uncertain_existing_topology_attempt(self):
        import meta_api,mechanism_reasoning
        program=self.root/'data/program.json';program.parent.mkdir();program.write_text('{}')
        folder=self.root/'data/mechanism-pipelines'/M.sha(program)/'topology';folder.mkdir(parents=True)
        prompt='Inert saved evidence contract';contract={'contract':{'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()},'prompt':prompt,'schema':{},'provider_schema':{}}
        (folder/'plan.json').write_text(json.dumps({'contract':contract['contract']}));(folder/'prompt.txt').write_text(prompt)
        (folder/'schema.json').write_text('{}');(folder/'provider-schema.json').write_text('{}')
        (folder/'attempt').mkdir();failure=folder/'attempt/failure.json';failure.write_bytes(b'{"charge_uncertain":true}')
        before=failure.read_bytes()
        with patch.object(meta_api,'MetaClient',OvernightMetaClient),patch.object(mechanism_reasoning,'plan',return_value=contract),\
             patch.object(mechanism_reasoning,'run_prepared')as dispatch:
            result=M.topology(program,root=self.root,task_budget_plan=B.PLAN,overnight_task_budget_plan=OUTER,allow_paid_api=True)
        self.assertEqual(result['status'],'pending_prior_mechanism_attempt_no_automatic_resend');dispatch.assert_not_called()
        self.assertEqual(failure.read_bytes(),before)


if __name__=='__main__':unittest.main()
