"""Read-only accelerated transport fixtures. No SSH, GPU or inference runs."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import sam3d_cluster_bridge as B
import sam3d_cluster_remote as Old
import sam3d_transfer_remote_v2 as R
import sam3d_transfer_v2 as T


class TransferFixtures(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        self.task=self.root/'task';self.identifier='upload-fixture-v2';self.helper_sha=hashlib.sha256(B.REMOTE_HELPER.read_bytes()).hexdigest()
        self.files={}
        for name in R.NATIVE:self.files['native/'+name]=self.task/'outputs'/self.identifier/name
        for name in R.EVIDENCE:
            if name in {'launch.json','supervisor-result.json'}:p=self.task/'upload-bridge-jobs'/self.identifier/name
            elif name=='offline-manifest.json':p=self.task/('offline-job-'+self.identifier+'.json')
            else:p=self.task/'component-jobs'/self.identifier/name
            self.files['evidence/'+name]=p
        for name,p in self.files.items():
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes((b'large exact native data '+bytes(range(128)))*(40000 if name=='native/object.glb'else 2)if not name.endswith('.json')else b'{"fixture_only":true}')
        self.write('native/inputs/manifest.json',{'fixture_only':True,'object_id':'fixture_part'})
        file_ref=lambda name:self.reference(name)
        self.metadata={'status':'candidate_generated_pending_visual_review','learned_inference_executed':True,
            'outputs':{'files':[{'file':name,**{k:v for k,v in file_ref('native/'+name).items()if k!='path'}}for name in ['object.glb','object-splat.ply','layout-and-pointmap.npz']]},
            'inputs':{'manifest_sha256':file_ref('native/inputs/manifest.json')['sha256'],
                'files':{role:{k:v for k,v in file_ref('native/inputs/'+filename).items()if k!='path'}for role,filename in [('image','image-source.png'),('mask','mask-source.png')]}},
            'model_input_sha256':{name:file_ref('native/inputs/'+name)['sha256']for name in ['rgb.png','mask.png']}}
        self.write('native/metadata.json',self.metadata)
        self.write('evidence/launch.json',{'format':'sam3d-upload-bridge-launch','job_id':self.identifier,'helper_sha256':self.helper_sha,
            'maximum_model_commands':1,'automatic_retry':False,'parent_allocation_modified':False})
        self.write('evidence/supervisor-result.json',{'job_id':self.identifier,'returncode':0,'automatic_retry':False})
        self.write('evidence/execution.json',{'job_id':self.identifier,'status':'native_output_ready_for_independent_import_review',
            'model_commands_attempted':1,'model_inference_executed':True,'automatic_retry':False,'parent_allocation_modified':False,
            'output_metadata_sha256':file_ref('native/metadata.json')['sha256']})
        self.rows=[file_ref(name)for name in sorted(self.files)];self.inventory={'status':'native_output_ready','files':copy.deepcopy(self.rows)}
        self.config={'ssh_control_path':'/private/tmp/offline-transfer/cm-%C'}
        self.task_patch=patch.object(R,'TASK',self.task);self.task_patch.start();self.addCleanup(self.task_patch.stop)
    def write(self,name,value):self.files[name].write_text(json.dumps(value))
    def reference(self,name):
        p=self.files[name];return{'path':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
    def request(self,name='native/object.glb',offset=0):
        rows={r['path']:r for r in self.rows}
        return{'action':'chunk_v2','job_id':self.identifier,'path':name,'offset':offset,'file_ref':rows[name],
            'completion_refs':[rows[k]for k in sorted(R.PROOFS)],'generation_helper_sha256':self.helper_sha,'inventory_sha256':T.digest(self.rows)}
    def transport(self):
        t=T.VerifiedTransferTransport(self.config);t._request_transfer=lambda request:R.serve(request)
        return t
    def old_request(self,instance,payload):
        if payload['action']=='inventory':return copy.deepcopy(self.inventory)
        return{'status':'delegated_v1','original_action':payload['action']}


class ReadOnlyRemoteTests(TransferFixtures):
    def test_chunks_keep_exact_bytes_and_completion_proofs_without_remote_writes(self):
        before={name:p.read_bytes()for name,p in self.files.items()};req=self.request();result=R.serve(req)
        raw=base64.b64decode(result['base64']);self.assertEqual(raw,before['native/object.glb'][:R.MAX_CHUNK])
        self.assertEqual(hashlib.sha256(raw).hexdigest(),result['chunk_sha256']);self.assertTrue(result['read_only'])
        self.assertEqual(before,{name:p.read_bytes()for name,p in self.files.items()})
        second=R.serve(self.request(offset=R.MAX_CHUNK));self.assertEqual(base64.b64decode(second['base64']),before['native/object.glb'][R.MAX_CHUNK:])
    def test_incomplete_changed_or_repeated_model_attempt_is_refused(self):
        for name,change in [('evidence/execution.json',{'model_commands_attempted':2}),('evidence/supervisor-result.json',{'returncode':1}),('native/metadata.json',{'learned_inference_executed':False})]:
            original=self.files[name].read_bytes();value=json.loads(original);value.update(change);self.write(name,value)
            with self.assertRaisesRegex(ValueError,'ProofHashChanged|FileLengthDiffers'):R.serve(self.request())
            self.files[name].write_bytes(original)
    def test_other_file_or_runtime_identity_cannot_substitute_the_original_inventory(self):
        req=self.request();req['generation_helper_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'LaunchProofDiffers'):R.serve(req)
        req=self.request();req['file_ref']['sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'ModelOutputHashDiffers'):R.serve(req)
    def test_no_new_job_action_hidden_path_or_credential_file_is_readable(self):
        for mutate in [lambda r:r.update(action='dispatch'),lambda r:r.update(job_id='../../.cache/huggingface/token'),
            lambda r:r.update(path='../../.secrets/meta.env'),lambda r:r.update(file_ref={'path':'native/../../secret','bytes':3,'sha256':'a'*64})]:
            request=self.request();mutate(request)
            with self.assertRaises(ValueError):R.serve(request)
    def test_symlink_and_truncated_chunk_fail_before_serving_bytes(self):
        path=self.files['native/object.glb'];raw=path.read_bytes();outside=self.root/'unrelated';outside.write_bytes(raw)
        path.unlink();path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError,'OrdinaryCompletedFileLengthDiffers'):R.serve(self.request())


class ClientTransportTests(TransferFixtures):
    def test_v1_dispatch_and_runtime_proofs_remain_unchanged(self):
        t=self.transport()
        with patch.object(B.SSHTransport,'request',lambda instance,payload:self.old_request(instance,payload)):
            result=t.request({'action':'dispatch','job_id':self.identifier})
        self.assertEqual(result['original_action'],'dispatch');self.assertEqual(t.transfer_statistics['full_inventory_requests'],0)
    def test_inventory_is_required_once_then_all_chunks_remain_byte_verified(self):
        t=self.transport()
        with self.assertRaisesRegex(ValueError,'full V1 inventory'):t.request({'action':'chunk','job_id':self.identifier,'path':'native/object.glb','offset':0})
        with patch.object(B.SSHTransport,'request',lambda instance,payload:self.old_request(instance,payload)):t.request({'action':'inventory','job_id':self.identifier})
        raw=b''
        for offset in range(0,self.files['native/object.glb'].stat().st_size,R.MAX_CHUNK):
            result=t.request({'action':'chunk','job_id':self.identifier,'path':'native/object.glb','offset':offset});raw+=base64.b64decode(result['base64'])
        self.assertEqual(raw,self.files['native/object.glb'].read_bytes());self.assertEqual(t.transfer_statistics['full_inventory_requests'],1)
        self.assertEqual(t.transfer_statistics['verified_chunk_bytes'],len(raw))
    def test_actual_existing_bridge_fetch_retains_full_file_hash_length_and_inventory_checks(self):
        t=self.transport();destination=self.root/'fetch';destination.mkdir()
        with patch.object(B.SSHTransport,'request',lambda instance,payload:self.old_request(instance,payload)):B.fetch(t,destination,self.identifier)
        for row in self.rows:
            target=destination/'returned'/row['path'];self.assertEqual(target.read_bytes(),self.files[row['path']].read_bytes())
        self.assertTrue(json.loads((destination/'return-inventory.json').read_text())['all_sha256_verified'])
    def test_same_size_mutable_blob_is_detected_by_client_full_hash_not_trusted_stat(self):
        t=self.transport();path=self.files['native/object.glb'];old_size=path.stat().st_size
        raw=bytearray(path.read_bytes());raw[-1]^=1;path.write_bytes(raw);self.assertEqual(path.stat().st_size,old_size)
        destination=self.root/'tampered-fetch';destination.mkdir()
        with patch.object(B.SSHTransport,'request',lambda instance,payload:self.old_request(instance,payload)):
            with self.assertRaisesRegex(ValueError,'Returned native bytes differ'):B.fetch(t,destination,self.identifier)
        self.assertFalse((destination/'returned/native/object.glb').exists());self.assertFalse((destination/'return-inventory.json').exists())
    def test_tampered_wire_chunk_is_rejected_before_existing_fetch_can_accept_it(self):
        t=self.transport()
        with patch.object(B.SSHTransport,'request',lambda instance,payload:self.old_request(instance,payload)):t.request({'action':'inventory','job_id':self.identifier})
        def changed(request):
            response=R.serve(request);response['base64']=base64.b64encode(b'wrong').decode();return response
        t._request_transfer=changed
        with self.assertRaisesRegex(ValueError,'chunk bytes/hash differ'):t.request({'action':'chunk','job_id':self.identifier,'path':'native/object.glb','offset':0})
    def test_changed_completion_inventory_cannot_mix_new_proofs_into_existing_transfer(self):
        t=self.transport()
        with patch.object(B.SSHTransport,'request',lambda instance,payload:self.old_request(instance,payload)):t.request({'action':'inventory','job_id':self.identifier})
        self.inventory['files'][0]['sha256']='f'*64
        with patch.object(B.SSHTransport,'request',lambda instance,payload:self.old_request(instance,payload)):
            with self.assertRaisesRegex(ValueError,'inventory changed'):t.request({'action':'inventory','job_id':self.identifier})
    def test_new_helper_does_not_rehash_large_files_per_chunk(self):
        request=self.request();seen=[];original=R.checked_reference
        def observed(reference,files,**kwargs):
            if kwargs.get('read'):seen.append(reference['bytes'])
            return original(reference,files,**kwargs)
        with patch.object(R,'checked_reference',side_effect=observed):R.serve(request)
        self.assertEqual(len(seen),4);self.assertLess(sum(seen),16*1024)
        self.assertNotIn(self.files['native/object.glb'].stat().st_size,seen)

if __name__=='__main__':unittest.main()
