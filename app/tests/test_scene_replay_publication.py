"""Workspace refresh only reuses verified constrained descriptive companions."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import object_program
import scene_replay_publication as replay


class ReplayPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def case(self, name='new-source', source='a'*64, backend='Meta SAM 3D Objects'):
        path = self.root/'data'/name/'program/interaction.json'; path.parent.mkdir(parents=True)
        doc = {'source': {'sha256': source}, 'assets': [{'backend': backend}], 'motion': []}
        path.write_text(json.dumps(doc))
        return {'id': name+'--program', 'interaction_path': path.relative_to(self.root).as_posix(),
                'original_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'interaction': doc}

    def actual_companion_fixture(self):
        # Reuse the existing analytic FK/source-reference fixture. Only provider
        # grounding and GLB decoding are replaced there; real export validation,
        # source hashes, angle curves, native scales and artifacts remain active.
        from tests.test_object_program import ObjectProgramTests
        fixture = ObjectProgramTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        self.root = fixture.root
        artifact = fixture.file('data/object-programs/unit/object-program.json', fixture.doc)
        case = {'id':'native--program', 'original_sha256':fixture.doc['evidence_program']['sha256'],
                'interaction_path':fixture.doc['evidence_program']['path'],
                'interaction':{'source':fixture.source,'assets':[{'backend':'Meta SAM 3D Objects'}]}}
        return fixture, case, {'artifact':artifact,'document':fixture.doc}

    def test_missing_companion_is_pending_and_never_builds_or_overwrites_independent_scene(self):
        case = self.case(); index=self.root/'showcase/aligned-scenes.json';index.parent.mkdir()
        saved={'format':'sam3d-aligned-scene-index','version':1,'scenes':[{'source_sha256':'a'*64,'manifest':{'path':'data/private/scene.json','sha256':'b'*64,'bytes':100}}]}
        index.write_text(json.dumps(saved)); before=index.read_bytes()
        with patch('sam3d_common_scene.build') as baseline, patch('sam3d_mechanisms.build') as constrained, patch('mechanism_reasoning.run_prepared') as paid:
            result=replay.publish_estimated_replays([case],root=self.root)
        self.assertEqual(result[0]['status'],'pending_constrained_mechanism_pipeline')
        self.assertEqual(result[0]['new_cpu_fits'],0);self.assertFalse(result[0]['independent_scene_fallback'])
        self.assertEqual(index.read_bytes(),before)
        baseline.assert_not_called();constrained.assert_not_called();paid.assert_not_called()

    def test_actual_validated_constrained_companion_is_indexed_without_rewriting_originals(self):
        fixture,case,_=self.actual_companion_fixture(); original=self.root/case['interaction_path'];before=original.read_bytes()
        result=replay.publish_estimated_replays([case],root=self.root)
        self.assertEqual(result[0]['status'],'estimated_constrained_replay_ready')
        self.assertEqual(result[0]['manifest'],fixture.doc['constrained_scene'])
        indexed=json.loads((self.root/'showcase/aligned-scenes.json').read_text())['scenes'][0]
        self.assertEqual(indexed['manifest'],fixture.doc['constrained_scene'])
        self.assertEqual(indexed['object_program'],result[0]['object_program'])
        self.assertEqual(object_program.checked_ref(indexed['object_program'],self.root).read_bytes(),
                         (self.root/'data/object-programs/unit/object-program.json').read_bytes())
        self.assertEqual(original.read_bytes(),before);self.assertFalse(result[0]['simulation_ready'])

    def test_wrong_program_or_changed_asset_cannot_replace_existing_registry(self):
        fixture,case,companion=self.actual_companion_fixture();index=self.root/'showcase/aligned-scenes.json';index.parent.mkdir();index.write_text('{"format":"sam3d-aligned-scene-index","version":1,"scenes":[]}');before=index.read_bytes()
        different=self.case('other',source=fixture.source['sha256'])
        with self.assertRaisesRegex(ValueError,'different original program'):
            replay.publish_constrained_result(companion,different['interaction_path'],root=self.root)
        (self.root/fixture.doc['component_assets'][0]['native_glb']['path']).write_bytes(b'glTFchanged')
        with self.assertRaises(ValueError):replay.publish_constrained_result(companion,case['interaction_path'],root=self.root)
        self.assertEqual(index.read_bytes(),before)

    def test_newest_program_per_source_only_and_no_legacy_or_modified_program(self):
        old=self.case('old');new=self.case('new');other=self.case('other',source='b'*64)
        legacy=self.case('legacy',backend='TripoSR');changed=self.case('changed');(self.root/changed['interaction_path']).write_text('{}')
        with patch.object(object_program,'discover',return_value=None) as discover:
            result=replay.publish_estimated_replays([old,new,other,legacy,changed],root=self.root)
        self.assertEqual(discover.call_count,2);self.assertEqual({r['case_id'] for r in result},{new['id'],other['id']})
        self.assertTrue(all(r['new_api_calls']==0 for r in result))

    def test_refresh_bounds_and_path_confinement_fail_before_reading(self):
        for name in ['.secrets/meta.env','data/../.secrets/meta.env','data/.private/value']:
            with self.subTest(name=name),patch.object(replay,'_sha') as hashing,self.assertRaises(ValueError):replay._local(self.root,name)
            hashing.assert_not_called()
        for kwargs in [{'maximum_new':4},{'maximum_new':True},{'timeout_s':1000}]:
            with self.assertRaises(ValueError):replay.publish_estimated_replays([],root=self.root,**kwargs)


if __name__=='__main__':unittest.main()
