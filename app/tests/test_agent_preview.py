"""Saved agent preview invariants and browser-local controls without inference."""
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build_agent_preview', ROOT/'scripts/build_agent_preview.py')
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)


class AgentPreviewContracts(unittest.TestCase):
    def test_between_time_avoids_dense_motion_even_when_semantics_are_sparse(self):
        document = {'source': {'duration_s': 4.}, 'evidence': [{'timestamp_s': 0.}, {'timestamp_s': 2.}],
                    'motion': [{'timestamp_s': t} for t in [.2, .4, .6, .8, 1., 1.2, 1.4, 1.6, 1.8]]}
        value = preview.between_sample_time(document, 0.)
        self.assertAlmostEqual(value, .1)
        self.assertNotIn(value, [row['timestamp_s'] for row in document['evidence']+document['motion']])

    def test_actual_saved_snapshot_keeps_unknown_gaps_and_no_physics_claims(self):
        data = json.loads((ROOT/'workspace/agent-preview.json').read_text())
        self.assertFalse(data['browser_executes_models'])
        self.assertFalse(data['browser_executes_mcp'])
        self.assertFalse(data['stdio_transport_tested_by_this_builder'])
        self.assertFalse(data['network_requests_by_builder'])
        self.assertTrue(data['twins'])
        for twin in data['twins']:
            self.assertEqual(twin['outputs']['describe_twin']['response']['program_sha256'], twin['program_sha256'])
            gap = twin['outputs'].get('query_state_gap')
            if gap:
                self.assertTrue(all(row['value'] == 'unknown' for row in gap['response']['states'].values()))
                self.assertEqual(gap['response']['conditional_poses'], [])
            self.assertFalse(twin['outputs']['get_simulation_manifest']['response']['policy']['simulation_ready'])

    def test_local_controls_switch_real_saved_responses_and_download_without_network(self):
        script = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const snapshot=JSON.parse(fs.readFileSync(process.argv[1],'utf8')),downloads=[],elements=new Map();
class Node{constructor(tag,id){this.tag=tag;this.id=id;this.children=[];this.dataset={};this.attrs={};this.value='';this.hidden=true;this.textContent='';this.classList={toggle(){}}}
append(...nodes){this.children.push(...nodes)}replaceChildren(...nodes){this.children=[...nodes]}setAttribute(k,v){this.attrs[k]=v}click(){if(this.download)downloads.push(this.download);this.onclick?.()}focus(){}}
const get=id=>{if(!elements.has(id))elements.set(id,new Node('div',id));return elements.get(id)};
const tabs=['describe_twin','get_procedure','query_state','get_simulation_manifest'].map((name,index)=>{const n=new Node('button','tab-'+index);n.dataset.tool=name;return n});
const context={window:{TWIN_AGENT_PREVIEW:snapshot},document:{getElementById:get,querySelectorAll:()=>tabs,createElement:tag=>new Node(tag)},location:{protocol:'file:',href:'file:///tmp/interaction-twin/workspace/agents.html',search:''},URL,Blob,URLSearchParams,history:{replaceState(){}},setTimeout:fn=>fn(),console,fetch(){throw Error('Preview must not contact any server or model')}};
vm.createContext(context);vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
const app=context.window.TwinAgentPreview;assert(app);
for(const twin of snapshot.twins){app.selectTwin(twin.id);assert.equal(get('twin-title').textContent,twin.title);assert.equal(JSON.parse(get('response-json').textContent).twin_id,twin.id);tabs[1].click();assert.equal(JSON.parse(get('request-json').textContent).name,'get_procedure');tabs[2].click();if(twin.outputs.query_state_gap){get('state-gap').click();const state=JSON.parse(get('response-json').textContent);assert(Object.values(state.states).every(row=>row.value==='unknown'));assert.equal(state.conditional_poses.length,0)}tabs[3].click();assert.equal(JSON.parse(get('response-json').textContent).policy.simulation_ready,false);get('download-handoff').click();tabs[0].click()}
get('download-config').click();assert.equal(downloads.length,snapshot.twins.length+1);
assert.throws(()=>app.localUrl('data/../../.secrets/meta.env'));assert.throws(()=>app.localUrl('https://example.org/file.glb'));
console.log(JSON.stringify({twins:snapshot.twins.length,downloads:downloads.length,network_calls:0}));
'''
        result = subprocess.run(['node', '-e', script, str(ROOT/'workspace/agent-preview.json'), str(ROOT/'workspace/agents.js')],
                                check=True, capture_output=True, text=True, timeout=20)
        self.assertEqual(json.loads(result.stdout)['network_calls'], 0)


if __name__ == '__main__':
    unittest.main()
