"""Meaningful scale, source-time and scope contracts for conditional assembly."""
import importlib.util
from pathlib import Path

import numpy as np
import unittest
import json
import subprocess

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('jabra_assembly',ROOT/'scripts/stage4_jabra_assembly.py')
assembly=importlib.util.module_from_spec(spec);spec.loader.exec_module(assembly)

def observation(sample, **updates):
    value={'track_id':'container_base','sample_index':sample,'single_component_visible':True,
        'component_boundary_plausible':True,'other_components_inside_mask':False,
        'interior_state':'empty_visible','reconstruction_view_usable':True,'reason':'model observation'}
    value.update(updates)
    return value

class JabraAssemblyTests(unittest.TestCase):
    def test_closed_or_occupied_high_score_does_not_beat_empty_component(self):
        selection={'object_id':'container_base','all_ranked_candidates':[{'sample_index':1,'score':.99},{'sample_index':2,'score':.8},{'sample_index':3,'score':.9}]}
        observations=[observation(1,interior_state='closed_or_hidden'),observation(2),observation(3,interior_state='occupied_visible')]
        assert assembly.select_scoped_view(selection,observations)['sample_index']==2

    def test_missing_duplicate_or_no_usable_scope_never_falls_back(self):
        selection={'object_id':'container_base','all_ranked_candidates':[{'sample_index':1,'score':.99}]}
        for values in [[],[observation(1),observation(1)],[observation(1,other_components_inside_mask=True)]]:
            with self.assertRaises(ValueError):assembly.select_scoped_view(selection,values)

    def test_scope_veto_overrides_plausible_silhouette_when_mask_contains_earbud(self):
        assert assembly.scope_mask_reasons(observation(1,other_components_inside_mask=True))==['component_scope_model_rejected_mask']
        assert assembly.scope_mask_reasons(observation(1,single_component_visible=False))==['component_scope_model_rejected_mask']
        # A correctly separated visible base may still be tracked when occupied;
        # occupancy prevents reconstruction input selection, not all motion.
        assert assembly.scope_mask_reasons(observation(1,interior_state='occupied_visible'))==[]

    def test_depth_scale_preserves_projection_of_entire_rigid_asset(self):
        record={'candidate_rotation':np.eye(3).tolist(),'candidate_translation':[.3,-.5,8.]}
        vertices=np.array([[0.,0.,0.],[1.,0.,.7],[-.3,.4,.2]])
        scale=assembly.scale_from_depth(.35,8.)
        matrix=assembly.scaled_pose(record,scale)
        raw=vertices+np.array(record['candidate_translation'])
        scaled=vertices@matrix[:3,:3].T+matrix[:3,3]
        np.testing.assert_allclose(scaled[:,:2]/scaled[:,2,None],raw[:,:2]/raw[:,2,None])
        np.testing.assert_allclose(scaled,raw*scale)

    def test_invalid_depth_or_nonrigid_pose_rejected(self):
        for a,b in [(0,1),(1,0),(-1,3),(float('nan'),1),(1,float('inf'))]:
            with self.assertRaises(ValueError):assembly.scale_from_depth(a,b)
        with self.assertRaises(ValueError):assembly.scaled_pose({'candidate_rotation':(np.eye(3)*2).tolist(),'candidate_translation':[0,0,3]},1.)

    def test_preceding_selection_hides_unknowns_and_stale_poses(self):
        rows=[{'timestamp_s':1.,'status':'supported_conditional_motion_candidate'},
              {'timestamp_s':1.2,'status':'unsupported_candidate_projection_retained'},
              {'timestamp_s':1.4,'status':'reference_candidate_unverified_geometry_and_pose'}]
        assert assembly.preceding_supported(rows,.9) is None
        assert assembly.preceding_supported(rows,1.1)==rows[0]
        assert assembly.preceding_supported(rows,1.3) is None
        assert assembly.preceding_supported(rows,1.4)==rows[2]
        assert assembly.preceding_supported(rows,2.) is None

    def test_javascript_temporal_gate_withholds_unknown_and_stale_samples(self):
        code=r"""
const assert=require('assert');const api=require(process.argv[1]);
const o={max_hold_s:.21,frames:[{timestamp_s:1,visible:true},{timestamp_s:1.2,visible:false},{timestamp_s:1.4,visible:true}]};
assert.equal(api.selectRecord(o,.9).visible,false);assert.equal(api.selectRecord(o,1.1).visible,true);
assert.equal(api.selectRecord(o,1.3).visible,false);assert.equal(api.selectRecord(o,1.4).visible,true);
assert.equal(api.selectRecord(o,2).visible,false);
const expected=[0,0,270,480];assert.deepEqual(api.viewport(270,480,[540,960]),expected);
"""
        subprocess.run(['node','-e',code,str(ROOT/'jabra_assembly_observer.js')],check=True,capture_output=True,text=True)
