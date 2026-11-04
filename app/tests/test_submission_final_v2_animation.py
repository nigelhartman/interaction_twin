"""Pure storyboard/clock checks. No frames, speech, movie or API are generated."""
import importlib.util
import math
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('submission_final_v2',ROOT/'scripts/build_submission_final_20261104_v2.py')
V=importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)


class SubmissionFinalV2AnimationTests(unittest.TestCase):
    def test_additive_outputs_and_preserved_exact_opening(self):
        self.assertEqual(V.OUT,ROOT/'output/submission-final-20261104-v2')
        self.assertEqual(V.MEDIA,ROOT/'showcase/media/submission-final-20261104-v2')
        self.assertEqual(V.OPENING,ROOT/'showcase/media/flagship-20261104-v1/cut-v2/interaction-twin-flagship-45s.mp4')
        self.assertEqual(V.CAPTURES,ROOT/'output/submission-final-20261104-v1/captures')
        self.assertEqual((V.D,V.TOTAL,V.FPS),(135,180,30))

    def test_tail_chapters_and_changed_cues_keep_exact_slots(self):
        self.assertEqual([(s,e)for s,e,_ in V.BEATS],[(0,18),(18,36),(36,50),(50,63),(63,78),(78,105),(105,125),(125,135)])
        self.assertEqual(V.BEATS[2][1]-V.BEATS[2][0],14)
        self.assertEqual(V.BEATS[6][1]-V.BEATS[6][0],20)
        for i,(start,end,text)in enumerate(V.CUES):
            self.assertGreaterEqual(start,V.BEATS[i][0])
            self.assertLessEqual(end,V.BEATS[i][1])
            self.assertLess(start,end)
            self.assertTrue(text)

    def test_review_roles_and_bounds_are_explicit_illustration(self):
        phases=[]
        for i in range(421):
            state=V.review_state(i/420)
            if not phases or phases[-1]!=state['illustrated_phase']:
                phases.append(state['illustrated_phase'])
            self.assertTrue(state['same_Muse_model_separate_prompts'])
            self.assertEqual(state['maximum_repair_rounds'],2)
            self.assertTrue(state['owner_approval_required'])
            self.assertFalse(state['live_model_events'])
            self.assertLessEqual(state['phase_progress'],1)
            self.assertGreaterEqual(state['phase_progress'],0)
        self.assertEqual(phases,['build','review','repair','build','review','approval_transfer','approve'])
        self.assertEqual(V.review_state(-1),V.review_state(0))
        self.assertEqual(V.review_state(2),V.review_state(1))

    def test_architecture_separates_each_Meta_stage_from_our_program_and_MCP_reuse(self):
        stages=[]
        for i in range(601):
            state=V.architecture_state(i/600)
            if not stages or stages[-1]!=state['illustrated_stage']:
                stages.append(state['illustrated_stage'])
            self.assertTrue(state['Meta_models_are_upstream'])
            self.assertTrue(state['MCP_reuse_is_separate_from_processing'])
            self.assertFalse(state['live_model_events'])
            self.assertGreaterEqual(state['stage_progress'],0)
            self.assertLessEqual(state['stage_progress'],1)
        self.assertEqual(stages,list(V.ARCHITECTURE_STAGES))

    def test_schedule_intervals_have_no_gaps_or_zero_duration(self):
        for schedule,end in [(V.REVIEW_SCHEDULE,14),(V.ARCHITECTURE_SCHEDULE,20)]:
            self.assertEqual(schedule[0][0],0)
            self.assertEqual(schedule[-1][1],end)
            for i,(start,stop,_)in enumerate(schedule):
                self.assertGreater(stop,start)
                if i:
                    self.assertEqual(start,schedule[i-1][1])

    def test_constant_speed_pulses_are_bounded_and_cross_bends_without_jumping(self):
        points=[(960,650),(960,735),(330,735),(330,650)]
        self.assertEqual(V.path_point(points,0),points[0])
        self.assertEqual(V.path_point(points,1),points[-1])
        self.assertEqual(V.path_point(points,-1),points[0])
        length=sum(math.dist(a,b)for a,b in zip(points,points[1:]))
        last=V.path_point(points,0)
        for i in range(1,101):
            p=V.path_point(points,i/100)
            self.assertLessEqual(math.dist(p,last),length/100+1e-9)
            self.assertTrue(330<=p[0]<=960 and 650<=p[1]<=735)
            last=p

    def test_architecture_boxes_fit_video_and_leave_footer_clear(self):
        for left,top,right,bottom in V.ARCHITECTURE_BOXES:
            self.assertTrue(0<=left<right<=V.W)
            self.assertTrue(200<=top<bottom<935)
        for i,a in enumerate(V.ARCHITECTURE_BOXES):
            for b in V.ARCHITECTURE_BOXES[i+1:]:
                self.assertTrue(a[2]<=b[0]or b[2]<=a[0]or a[3]<=b[1]or b[3]<=a[1])

    def test_spoken_cues_fit_requested_word_budgets_without_claiming_generated_audio(self):
        review=V.CUES[2][2];architecture=V.CUES[6][2]
        self.assertLessEqual(len(review.split()),38)
        self.assertLessEqual(len(architecture.split()),55)
        self.assertIn('same Muse model',review)
        self.assertIn('two repair rounds',review)
        self.assertIn('owner approves',review)
        for phrase in ['SAM three point one','SAM three D','Our Python code','source-linked object programs','Read-only M C P','Three J S']:
            self.assertIn(phrase,architecture)


if __name__=='__main__':
    unittest.main()
