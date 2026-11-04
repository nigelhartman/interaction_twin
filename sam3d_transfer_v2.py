"""Optional transfer-only transport. Existing GPU generation remains unchanged.

Default disabled: callers must explicitly pass VerifiedTransferTransport as the
existing bridge's transport_factory. V1 dispatch/poll/inventory and proofs remain
unchanged. This module never alters allocation, remote jobs, helper or config.
"""
from __future__ import annotations
import base64
import copy
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

import sam3d_cluster_bridge as B

REMOTE=Path(__file__).resolve().with_name('sam3d_transfer_remote_v2.py')
PROOFS={'evidence/launch.json','evidence/supervisor-result.json','evidence/execution.json','native/metadata.json'}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def inventory_rows(response):
    rows=response.get('files');expected={'native/'+name for name in B.NATIVE}|{'evidence/'+name for name in B.EVIDENCE}
    if(response.get('status')!='native_output_ready'or not isinstance(rows,list)or not all(isinstance(r,dict)for r in rows)or len(rows)!=len(expected)or
       {r.get('path')for r in rows}!=expected):raise ValueError('Exact completed V1 inventory required before accelerated transfer')
    for row in rows:
        if(set(row)!={'path','sha256','bytes'}or type(row['bytes'])is not int or not 0<row['bytes']<=256*2**20 or
           not isinstance(row['sha256'],str)or not B.re.fullmatch('[a-f0-9]{64}',row['sha256'])):
            raise ValueError('Completed source inventory schema/size/hash differs')
    if sum(r['bytes']for r in rows)>512*2**20:raise ValueError('Original native return total budget exceeded')
    return copy.deepcopy(rows)


class VerifiedTransferTransport(B.SSHTransport):
    """One full V1 inventory plus read-only SHA-bound V2 chunks; no stat cache."""
    def __init__(self,config):
        super().__init__(config)
        if not REMOTE.is_file()or REMOTE.is_symlink():raise ValueError('Reviewed transfer helper unavailable')
        self.transfer_helper=REMOTE.read_bytes();self.transfer_helper_sha256=hashlib.sha256(self.transfer_helper).hexdigest()
        self.inventories={};self.transfer_statistics={'full_inventory_requests':0,'chunk_requests':0,'verified_chunk_bytes':0,
            'large_file_hashes_per_chunk':0,'default_enabled':False}

    def request(self,payload):
        action=payload.get('action')
        if action=='inventory':
            result=super().request(payload);rows=inventory_rows(result);key=payload['job_id']
            record={'rows':rows,'sha256':digest(rows)}
            if key in self.inventories and self.inventories[key]!=record:raise ValueError('A completed job inventory changed; no mixed-source transfer')
            self.inventories[key]=record;self.transfer_statistics['full_inventory_requests']+=1
            return result
        if action!='chunk':return super().request(payload)
        if set(payload)!={'action','job_id','path','offset'}:raise ValueError('Exact existing chunk request required')
        inventory=self.inventories.get(payload['job_id'])
        if inventory is None:raise ValueError('Obtain the source-bound full V1 inventory before V2 chunks')
        rows={r['path']:r for r in inventory['rows']};reference=rows.get(payload['path'])
        if not reference or type(payload['offset'])is not int or not 0<=payload['offset']<reference['bytes']:
            raise ValueError('Chunk path/offset differs from the completed source inventory')
        request={**payload,'action':'chunk_v2','file_ref':reference,'completion_refs':[rows[k]for k in sorted(PROOFS)],
            'generation_helper_sha256':self.helper_sha,'inventory_sha256':inventory['sha256']}
        result=self._request_transfer(request)
        if(result.get('protocol')!='sam3d-source-bound-transfer-v2'or result.get('read_only')is not True or
           result.get('path')!=reference['path']or result.get('offset')!=payload['offset']or
           result.get('file_sha256')!=reference['sha256']or result.get('inventory_sha256')!=inventory['sha256']):
            raise ValueError('Transferred chunk belongs to another original source/offset')
        raw=base64.b64decode(result.get('base64',''),validate=True)
        if(len(raw)!=min(4*2**20,reference['bytes']-payload['offset'])or
           hashlib.sha256(raw).hexdigest()!=result.get('chunk_sha256')):
            raise ValueError('Transferred chunk bytes/hash differ; no native import')
        self.transfer_statistics['chunk_requests']+=1;self.transfer_statistics['verified_chunk_bytes']+=len(raw)
        return result

    def _request_transfer(self,payload):
        options=['ssh','-o','BatchMode=yes','-o','NumberOfPasswordPrompts=0','-o','ConnectionAttempts=1',
            '-o','ConnectTimeout=8','-o','ControlMaster=no','-o','ProxyCommand=false','-S',self.socket]
        try:
            check=subprocess.run(options+['-O','check','KISSKI'],stdin=subprocess.DEVNULL,capture_output=True,timeout=12)
        except(subprocess.TimeoutExpired,OSError)as error:
            raise B.WorkerUnavailable('Existing SSH master unavailable for source-bound transfer.',
                B.transport_diagnostic('transfer_v2_master',timeout=isinstance(error,subprocess.TimeoutExpired)))from error
        if check.returncode:
            raise B.WorkerUnavailable('Existing SSH connection unavailable; original GPU attempt remains intact.',
                B.transport_diagnostic('transfer_v2_master',returncode=check.returncode,stderr=check.stderr))
        command=shlex.join([B.REMOTE_PYTHON,'-I','-c',self.transfer_helper.decode()])
        try:
            self.contacted=True
            process=subprocess.run(options+['KISSKI',command],input=json.dumps(payload,allow_nan=False).encode(),capture_output=True,timeout=90)
        except(subprocess.TimeoutExpired,OSError)as error:
            raise B.WorkerUnavailable('Native transfer channel unavailable; same completed job can resume.',
                B.transport_diagnostic('transfer_v2_chunk',timeout=isinstance(error,subprocess.TimeoutExpired)))from error
        try:result=json.loads(next(line for line in reversed(process.stdout.splitlines())if line.startswith(b'{')))
        except(ValueError,StopIteration)as error:
            raise B.WorkerUnavailable('Native transfer returned no usable response.',
                B.transport_diagnostic('transfer_v2_response',returncode=process.returncode,stderr=process.stderr))from error
        if not isinstance(result,dict):raise ValueError('Bounded native transfer response object required')
        if result.get('status')=='transfer_rejected':raise ValueError('Native transfer proof rejected: '+str(result.get('reason','unknown'))[:200])
        if process.returncode:
            raise B.WorkerUnavailable('Native transfer channel failed; original attempt is preserved.',
                B.transport_diagnostic('transfer_v2_response',returncode=process.returncode,stderr=process.stderr))
        return result
