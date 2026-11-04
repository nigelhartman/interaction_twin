"""Only exact curated source footage can gain an explicit private rights record."""
import concurrent.futures
import json
import unittest
from unittest.mock import patch

from agent_memory import digest
import marketing_sources as M
from tests import test_curated_twin_catalog as curated_fixtures


class MarketingSourcesTests(unittest.TestCase):
    def setUp(self):
        self.fixture=curated_fixtures.CuratedTwinCatalogTests(methodName='runTest');self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root;self.id=self.fixture.id;self.sources=M.MarketingSources(self.root)

    def test_get_exposes_curated_video_but_never_silently_authorizes_or_writes(self):
        before=sorted(self.root.rglob('*'))
        with patch.object(M,'private_directory',side_effect=AssertionError('GET cannot initialize authorization state')):
            listing=self.sources.list()
        self.assertEqual(before,sorted(self.root.rglob('*')));self.assertEqual(listing['sources'],[])
        pack=listing['packs'][0];self.assertTrue(pack['can_authorize']);self.assertFalse(pack['marketing_media_authorized'])
        self.assertEqual(pack['source_authorization_status'],'needs_rights_acknowledgement');self.assertEqual(pack['source_ids'],[])
        self.assertEqual(pack['public_source_video_ref'],self.fixture.ref(self.fixture.doc['source']['path']))

    def test_explicit_authorization_is_hash_bound_idempotent_and_links_pack(self):
        original=self.fixture.doc['source']['path'];raw=(self.root/original).read_bytes()
        source=self.sources.authorize_pack(self.id,rights_acknowledged=True)
        self.assertRegex(source['id'],r'^pack_[a-f0-9]{64}$');self.assertEqual(source['pack_ids'],[self.id]);self.assertEqual(source['file'],self.fixture.ref(original))
        self.assertEqual(source,self.sources.authorize_pack(self.id,rights_acknowledged=True));self.assertEqual(source,self.sources.get(source['id']))
        path=self.root/M.AUTH_FOLDER/(source['id']+'.json');record=json.loads(path.read_text())
        self.assertEqual(source['id'],'pack_'+digest(record));self.assertEqual(record['curated_record_sha256'],digest(self.fixture.record))
        self.assertTrue(record['rights_acknowledged']);self.assertFalse(source['authorization']['claim_free_guaranteed']);self.assertEqual(path.stat().st_mode&0o777,0o600)
        listing=self.sources.list();self.assertEqual([r['id']for r in listing['sources']],[source['id']])
        pack=listing['packs'][0];self.assertTrue(pack['marketing_media_authorized']);self.assertFalse(pack['can_authorize']);self.assertEqual(pack['source_ids'],[source['id']])
        self.assertEqual(pack['source_authorization_status'],'authorized');self.assertEqual((self.root/original).read_bytes(),raw)

    def test_missing_ack_or_private_unknown_id_cannot_create_authorization(self):
        for ack in [False,None,1,'true']:
            with self.subTest(ack=ack),self.assertRaisesRegex(ValueError,'Acknowledge'):self.sources.authorize_pack(self.id,rights_acknowledged=ack)
        for identifier in ['private-pending-pack','../../.secrets/meta.env','https://example.invalid/video.mp4']:
            with self.subTest(identifier=identifier),self.assertRaises(ValueError):self.sources.authorize_pack(identifier,rights_acknowledged=True)
        self.assertFalse((self.root/M.AUTH_FOLDER).exists())

    def test_missing_public_video_has_explanation_and_authorization_fails_closed(self):
        del self.fixture.public_assets[self.fixture.doc['source']['path']];self.fixture.publish()
        pack=self.sources.list()['packs'][0]
        self.assertFalse(pack['can_authorize']);self.assertFalse(pack['marketing_media_authorized']);self.assertIsNone(pack['public_source_video_ref'])
        self.assertEqual(pack['source_authorization_status'],'source_unavailable');self.assertTrue(pack['source_authorization_reason'])
        with self.assertRaisesRegex(ValueError,'No exact public source'):self.sources.authorize_pack(self.id)
        self.assertFalse((self.root/M.AUTH_FOLDER).exists())

    def test_video_hash_program_source_or_curated_record_changes_never_reuse_rights(self):
        source=self.sources.authorize_pack(self.id);video=self.root/source['file']['path'];original=video.read_bytes()
        video.write_bytes(b'tampered source')
        with self.assertRaises(ValueError):self.sources.get(source['id'])
        video.write_bytes(original)
        self.fixture.record['title']='New curated revision label';self.fixture.publish()
        with self.assertRaisesRegex(ValueError,'another curated revision'):self.sources.get(source['id'])
        listing=self.sources.list();self.assertEqual(listing['sources'],[]);self.assertFalse(listing['packs'][0]['marketing_media_authorized'])
        new=self.sources.authorize_pack(self.id);self.assertNotEqual(new['id'],source['id'])
        self.assertEqual(len(list((self.root/M.AUTH_FOLDER).glob('pack_*.json'))),2)

    def test_record_cannot_swap_in_another_public_video_or_arbitrary_path(self):
        original=self.fixture.record['source']['path'];foreign='data/foreign/public.mp4';foreign_ref=self.fixture.file(foreign,b'public-but-not-this-program')
        self.fixture.admit(foreign);self.fixture.record['source'].update(path=foreign,sha256=foreign_ref['sha256']);self.fixture.publish()
        with self.assertRaisesRegex(ValueError,'program source differ'):self.sources.authorize_pack(self.id)
        self.fixture.record['source']['path']='.secrets/meta.env';self.fixture.publish()
        with self.assertRaises(ValueError):self.sources.authorize_pack(self.id)
        self.assertFalse((self.root/M.AUTH_FOLDER).exists());self.assertTrue((self.root/original).is_file())

    def test_immutable_authorization_mutation_or_symlink_is_unavailable(self):
        source=self.sources.authorize_pack(self.id);path=self.root/M.AUTH_FOLDER/(source['id']+'.json');original=path.read_bytes()
        record=json.loads(original);record['rights_acknowledged']=False;path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'Immutable'):self.sources.get(source['id'])
        self.assertEqual(self.sources.list()['sources'],[])
        path.unlink();outside=self.root/'other.json';outside.write_bytes(original);path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError,'unavailable'):self.sources.get(source['id'])
        with self.assertRaisesRegex(ValueError,'changed'):self.sources.authorize_pack(self.id)
        self.assertEqual(outside.read_bytes(),original)

    def test_concurrent_duplicate_ack_publishes_one_complete_immutable_record(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2)as executor:
            rows=list(executor.map(lambda _:self.sources.authorize_pack(self.id),range(2)))
        self.assertEqual(rows[0],rows[1]);folder=self.root/M.AUTH_FOLDER
        self.assertEqual(len(list(folder.glob('pack_*.json'))),1);self.assertFalse(list(folder.glob('*.tmp')))
        self.assertEqual(json.loads(next(folder.glob('pack_*.json')).read_text())['case_id'],self.id)

    def test_actual_current_library_preserves_declared_clips_and_never_writes_rights(self):
        with patch.object(M,'private_directory',side_effect=AssertionError('Read only actual listing')):
            listing=M.MarketingSources().list()
        self.assertEqual([row['id']for row in listing['sources']if row['id']in M.DECLARED],list(M.DECLARED))
        camera=next(row for row in listing['packs']if row['id']==M.CAMERA)
        if camera['marketing_media_authorized']:
            self.assertTrue(any(identifier.startswith('pack_')for identifier in camera['source_ids']))
        else:self.assertEqual(camera['source_authorization_status'],'needs_rights_acknowledgement')


if __name__=='__main__':unittest.main()
