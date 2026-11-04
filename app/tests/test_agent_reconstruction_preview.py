"""Saved reconstruction UI data and no-network case-switch contracts."""
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

ROOT=Path(__file__).resolve().parents[1]


class AgentReconstructionPreviewTests(unittest.TestCase):
    def test_real_saved_package_identity_and_no_inference_claims(self):
        doc=json.loads((ROOT/'workspace/reconstruction-preview.json').read_text())
        self.assertLess((ROOT/'workspace/reconstruction-preview.json').stat().st_size,1500000)
        self.assertFalse(doc['browser_connects_to_mcp']);self.assertEqual(doc['new_model_calls'],0)
        self.assertEqual(doc['new_mcp_calls'],0)
        story=doc['scripted_inspection'];self.assertEqual(story['trace_summary']['call_count'],32)
        for name,key in [('inspection-task.json','task_sha256'),('mcp-trace.json','trace_sha256')]:
            self.assertEqual(hashlib.sha256((ROOT/story['links'][name]).read_bytes()).hexdigest(),story[key])
        self.assertEqual(hashlib.sha256((ROOT/story['zip_path']).read_bytes()).hexdigest(),story['zip_sha256'])
        self.assertFalse(story['task']['simulation_ready']);self.assertFalse(story['task']['robot_execution_allowed'])
        self.assertEqual(len(story['task']['blocking_before_robot_execution']),7)
        self.assertGreaterEqual(sum(len(c['versions'])for c in doc['cases']),8)
        for case in doc['cases']:
            for version in case['versions']:
                self.assertEqual(version['descriptor']['twin_id'],case['id'])
                self.assertFalse(version['descriptor']['gauge']['metric'])
                self.assertFalse(version['descriptor']['physical_accuracy_verified'])
                self.assertFalse(version['descriptor']['simulation_ready'])
                artifact=version['quality_response_artifact']
                raw=(ROOT/artifact['path']).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(),artifact['sha256'])
                full=json.loads(raw)
                if version['quality_preview_omitted_fields']:
                    self.assertEqual(version['quality_preview_omitted_fields'],['report.frames'])
                    del full['report']['frames']
                self.assertEqual(full,version['quality'])

    def test_actual_js_case_switch_and_version_tools_do_not_fetch_or_mix_sources(self):
        code=r'''
const fs=require('fs'),vm=require('vm'),root=process.argv[1];
class Element{constructor(){this.children=[];this.value='';this.dataset={};this.hidden=false;this.disabled=false;this.textContent='';this.classList={toggle(){}};}append(...v){this.children.push(...v);}replaceChildren(...v){this.children=v;if(v.length&&v[0].value)this.value=v[0].value;}setAttribute(){}get firstChild(){return this.children[0];}scrollIntoView(){}click(){}set innerHTML(v){throw Error('Untrusted HTML forbidden');}}
const elements={},events={},buttons=['list_reconstructions','get_reconstruction','get_reconstruction_report'].map(t=>{const e=new Element();e.dataset.reconstructionTool=t;return e;});
const document={getElementById:id=>(elements[id]??=new Element()),createElement:()=>new Element(),querySelectorAll:()=>buttons,addEventListener:(name,f)=>{events[name]=f;}};
const data=JSON.parse(fs.readFileSync(root+'/workspace/reconstruction-preview.json','utf8'));
const first=data.cases.find(c=>c.id.includes('jabra')),camera=data.cases.find(c=>c.id.includes('camera'));
const window={TWIN_RECONSTRUCTION_PREVIEW:data,TwinAgentPreview:{active:{id:first.id,program_sha256:first.list_response.program_sha256}}};
const context={window,document,location:{protocol:'http:',href:'http://localhost/workspace/agents.html'},URL,Blob,setTimeout,matchMedia:()=>({matches:true}),fetch(){throw Error('UI must use saved responses only');}};
vm.createContext(context);vm.runInContext(fs.readFileSync(root+'/workspace/agent-reconstructions.js','utf8'),context);
function check(value,msg){if(!value)throw Error(msg);}
const ui=window.TwinReconstructionPreview;check(ui&&ui.current.id===first.id,'initial binding');
ui.renderVersion('case-lid-triposr-v1');buttons.find(b=>b.dataset.reconstructionTool==='get_reconstruction_report').onclick();
let response=JSON.parse(elements['reconstruction-response'].textContent);check(response.version_id==='case-lid-triposr-v1','version tool binding');
window.TwinAgentPreview.active={id:camera.id,program_sha256:camera.list_response.program_sha256};events['twin:agent-preview-change']();
response=JSON.parse(elements['reconstruction-response'].textContent);check(response.twin_id===camera.id,'case switch stale response');
window.TwinAgentPreview.active={id:camera.id,program_sha256:'0'.repeat(64)};events['twin:agent-preview-change']();check(elements['reconstruction-content'].hidden,'stale program accepted');
window.TwinAgentPreview.active={id:'unknown',program_sha256:'0'.repeat(64)};events['twin:agent-preview-change']();check(elements['reconstruction-content'].hidden,'unknown source accepted');
for(const path of ['../.secrets/meta.env','data/../.secrets/test','data/.hidden/file','https://evil.test/key']){let rejected=false;try{ui.local(path);}catch{rejected=true;}check(rejected,'unsafe artifact URL');}
'''
        result=subprocess.run(['node','-e',code,str(ROOT)],cwd=ROOT,capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
