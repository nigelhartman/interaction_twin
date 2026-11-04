"""Independent contract checks, using DOM doubles rather than browser pixels."""
from pathlib import Path
import json
import subprocess
import tempfile
import unittest

from component_seed_review import write_derivative,sha

ROOT=Path(__file__).resolve().parents[1]


class IncrementUIAuditTests(unittest.TestCase):
    def test_terminal_job_cannot_enable_ineligible_component(self):
        code=r'''
const fs=require('fs'),vm=require('vm');
const source=fs.readFileSync(process.argv[1],'utf8');
const segment=source.slice(source.indexOf('function showJob(job){'),source.indexOf('async function watchJob('));
function check(eligible,runtime,status,expected){
 const elements={};function el(){return{value:'',hidden:false,disabled:false,textContent:'',options:[],replaceChildren(){},append(){},prepend(option){this.options.unshift(option)}};}
 const $=id=>(elements[id]??=el());$('job-case').value='case';$('job-component').value='part';
 const context={$ ,jobCatalog:[{id:'case',components:[{track_id:'part',eligible}]}],jobRuntimeAvailable:runtime,savedJobRows:[],
                node:el,human:String,finite:Number.isFinite,link:el};
 vm.createContext(context);vm.runInContext(segment,context);
 vm.runInContext('showJob({id:"observed-fixture",status:'+JSON.stringify(status)+'})',context);
 if($('start-observed').disabled!==expected)throw Error('Incorrect eligibility: '+JSON.stringify({eligible,runtime,status,expected}));
}
check(false,true,'complete',true);check(true,true,'complete',false);check(true,true,'running',true);check(true,false,'complete',true);
'''
        result=subprocess.run(['node','-e',code,str(ROOT/'workspace/geometry-lab.js')],cwd=ROOT,
                              capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_seed_writer_rejects_changed_config_receipt_before_false_automatic_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary)/'run';(run/'model-audit-v1').mkdir(parents=True)
            config={'manual_seed_boxes_used':False,'manual_geometry_used':False,'tracks':[]}
            (run/'annotations.json').write_text(json.dumps(config))
            (run/'tracks.json').write_text('{}')
            audit={'status':'complete_candidate_visual_audit','source_sha256':'a'*64,
                   'source_track_index_sha256':sha(run/'tracks.json'),'result':{'observations':[]}}
            (run/'model-audit-v1/audit.json').write_text(json.dumps(audit))
            metadata={'config_sha256':sha(run/'annotations.json'),'source_sha256':'a'*64,
                      'sam_video_propagation_executed':True,'manual_seed_boxes_used':False}
            (run/'metadata.json').write_text(json.dumps(metadata))
            (run/'annotations.json').write_text(json.dumps(dict(config,tracks=[{'id':'new','prompts':[]}])) )
            destination=Path(temporary)/'new.json'
            with self.assertRaisesRegex(ValueError,'Configuration or source identity'):
                write_derivative(run,destination)
            self.assertFalse(destination.exists())


if __name__=='__main__':unittest.main()
