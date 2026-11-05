#!/usr/bin/env python3
"""New output-only industrial-arm/storytelling drafts; preserve V1 deliveries.

Reuses the approved visual edit and untouched compositors, with a narrow fresh
capture adapter for the separately validated V2 surface-grasp and arm-rendering
receipts. No API, browser capture, hosted voice or publication.
"""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
import build_flagship_trailer_20261105_v3 as M
import build_submission_final_20261105_v4 as E
ROOT=M.ROOT;F=M.F;BaseAssets=M.Assets

OUTPUT_ROOTS=(M.OUT,ROOT/'output/trailer-rework-20261105-v2/films')
def safe_output(path):
    path=Path(path).resolve()
    if not any(path.is_relative_to(root)and path!=root for root in OUTPUT_ROOTS):raise ValueError('Fresh draft outputs must stay in their explicit film-output roots.')
    return path

class StoryAssets(BaseAssets):
    def __init__(self,charging=None,case_capture=None):
        if not charging or not case_capture:raise ValueError('Both exact bound industrial-armV2 captures required.')
        super().__init__(None,None)
        self.case_capture_ref=F.ref(Path(case_capture).resolve());self.case_capture=M.read(self.case_capture_ref)
        self.charging_ref=F.ref(Path(charging).resolve());self.charging=M.read(self.charging_ref)
        self.new_capture_checks={}
        freeze=ROOT/'output/trailer-rework-20261105-v2/implementation-freeze.json';self.capture_freeze_ref=F.ref(freeze)
        if self.capture_freeze_ref['sha256']!='b9caf29b319ad493d990f42ac3bbbf12b81cdb3c1f1f518b5a48585182f5013e':raise ValueError('Industrial capture freeze differs from independently cleared source.')
        frozen=json.loads(freeze.read_text())
        for reference in frozen['references']:F.checked(reference)
        for kind,c,ref,validator,duration in [('case',self.case_capture,self.case_capture_ref,'validate_case_motion_capture_20261105_v2.cjs',14),('charging',self.charging,self.charging_ref,'validate_charging_capture_20261105_v2.cjs',13)]:
            if c['version']!=2 or c['canvas']!=[1920,1080]or c['duration_s']!=duration or c.get('video_ref_requires_local_download_binding')is not False or c.get('local_validation',{}).get('passed')is not True:raise ValueError('FreshV2 captures must pass exact-byte local binding first.')
            self.new_capture_checks[kind]=json.loads(subprocess.run(['node',str(ROOT/'scripts'/validator),str(F.checked(ref))],check=True,capture_output=True,text=True).stdout)
            F.checked(c['video_ref']);F.checked(c['exact_browser_receipt_ref'])
        if self.case_capture['industrial_arm_factory']!=self.charging['industrial_visual_adapter']['industrial_arm_factory']:raise ValueError('Both shots must use the same pinned industrial-arm factory.')
        self.capture_implementation_refs=[F.ref(ROOT/name)for name in ['showcase/case-motion-capture-20261105-v2.html','showcase/case-motion-capture-20261105-v2.js','showcase/case-grasp-adapter-20261105-v2.js','showcase/industrial-arm-presentation-20261105-v1.js','showcase/charging-rehearsal-20261105-v2.html','showcase/charging-rehearsal-20261105-v2.js','scripts/validate_case_motion_capture_20261105_v2.cjs','scripts/bind_case_motion_capture_20261105_v2.py','scripts/validate_charging_capture_20261105_v2.cjs','scripts/bind_charging_capture_20261105_v2.py','output/trailer-rework-20261105-v2/case-grasp/candidate-binding.json','output/trailer-rework-20261105-v2/charging-visual/visual-adapter.json']]
        present={r['path']for r in self.capture_implementation_refs}
        self.capture_implementation_refs.extend(r for r in frozen['references']if r['path']not in present)

def audio_checked(folder,duration):
    folder=safe_output(folder);d=json.loads((folder/'audio.json').read_text())
    if d['duration_s']!=duration or d['all_speech_atempo']!=1 or d['hosted_speech']is not False:raise ValueError('Exact local1x narrative audio required.')
    if any(v['atempo']!=1 or v.get('local_voice_rate_setting')!=150 for v in d['voice']):raise ValueError('Approved natural150narrator required.')
    return d,F.checked(d['mix'])

def annotate(folder,source_audio,script_review):
    snapshot=folder/'story-editor-source.py';snapshot.write_bytes(Path(__file__).read_bytes())
    p=folder/'draft-receipt.json';d=json.loads(p.read_text());c=M.read(d['case_capture']);q=M.read(d['charging_capture'])
    d.update(format='interaction-twin-narrative-industrial-motion-draft',version=4,local_script_and_visual_reviews_not_Muse_verdict=d['completed_review_rounds'],story_narration=F.ref(source_audio/'audio.json'),local_script_review=F.ref(script_review),editor_builder=F.ref(Path(__file__)),editor_builder_snapshot=F.ref(snapshot),new_industrial_arm_capture_version=2,industrial_capture_implementation_freeze=F.ref(ROOT/'output/trailer-rework-20261105-v2/implementation-freeze.json'),case_grasp_adapter_is_authored=True,case_original_typed_plan_unchanged=False,case_surface_grasp_binding=c['grasp_candidate_binding'],industrial_arm_factory=c['industrial_arm_factory'],charging_industrial_visual_adapter=q['industrial_visual_adapter'],charging_kinematic_paths_and_fixture_staging_unchanged=True,normal_local_voice_rate_setting=150,fresh_Muse_verdict_available=False,publication=False)
    d['review_scope']='Independent local narrative review plus local visual/capture review. Owner playback remains required; no fresh Muse verdict is claimed.'
    d['previous_charging_implementation_freeze']=d.pop('charging_implementation_freeze');d['capture_implementation_note']='ExactV2 capture files, factory, grasp/visual adapters and bound validators are pinned in capture_implementation_refs; prior charging freeze covers the unchanged accepted kinematics and source fixture.'
    F.write(p,d)
    for name in('captions.vtt','transcript.txt'):(folder/name).write_bytes((source_audio/name).read_bytes())
    return d

def film45(folder,audio_folder,case,charging,script_review,visual_review):
    folder=safe_output(folder);audio_folder=safe_output(audio_folder);_,wave=audio_checked(audio_folder,45);a=StoryAssets(charging,case)
    output_guard=M.output_path;M.output_path=safe_output
    try:M.render_motion_draft(folder,a,wave,True,[script_review,visual_review])
    finally:M.output_path=output_guard;a.close()
    d=annotate(folder,audio_folder,script_review);print(json.dumps({'movie':d['movie'],'receipt':F.ref(folder/'draft-receipt.json'),'captions':F.ref(folder/'captions.vtt')},indent=2),flush=True)

def film180(folder,opening,audio_folder,script_review):
    folder=safe_output(folder);opening=safe_output(opening);audio_folder=safe_output(audio_folder);audio_checked(audio_folder,180)
    # Scoped in-process adapter only. Imported compositor files stay untouched.
    original=M.Assets;output_guard=M.output_path;M.Assets=StoryAssets;M.output_path=safe_output
    try:E.build(folder,opening,audio_folder)
    finally:M.Assets=original;M.output_path=output_guard
    snapshot=folder/'story-editor-source.py';snapshot.write_bytes(Path(__file__).read_bytes())
    p=folder/'provenance.json';d=json.loads(p.read_text());d.update(format='interaction-twin-narrative-industrial-submission-draft',version=5,editor_builder=F.ref(Path(__file__)),editor_builder_snapshot=F.ref(snapshot),local_script_review=F.ref(script_review),normal_local_voice_rate_setting=150,fresh_Muse_verdict_available=False,previous_tail_audio_retained=False,narrative_rewritten_across_entire180s=True,industrial_arm_visual_version=2,industrial_capture_implementation_freeze=F.ref(ROOT/'output/trailer-rework-20261105-v2/implementation-freeze.json'),publication=False)
    F.write(p,d);print(json.dumps({'movie':d['movie'],'provenance':F.ref(p),'captions':F.ref(folder/'captions.vtt')},indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--film45',type=Path);p.add_argument('--film180',type=Path);p.add_argument('--audio',type=Path,required=True);p.add_argument('--case',type=Path);p.add_argument('--charging',type=Path);p.add_argument('--opening',type=Path);p.add_argument('--script-review',type=Path,required=True);p.add_argument('--reference-visual-review',type=Path);a=p.parse_args()
    if a.film45:
        if not all([a.case,a.charging,a.reference_visual_review]):p.error('--film45 requires both boundV2 captures and reference visual review')
        film45(a.film45,a.audio,a.case,a.charging,a.script_review.resolve(),a.reference_visual_review.resolve())
    elif a.film180:
        if not a.opening:p.error('--film180 requires exact new45s opening')
        film180(a.film180,a.opening,a.audio,a.script_review.resolve())
    else:p.error('Choose an output-only film action.')
if __name__=='__main__':main()
