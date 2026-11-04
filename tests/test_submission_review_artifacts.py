import importlib.util, json, tempfile, unittest
from pathlib import Path

SPEC=importlib.util.spec_from_file_location('review_builder',Path(__file__).resolve().parents[1]/'scripts/build_submission_review_artifacts.py')
B=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(B)

class ReviewArtifactsTests(unittest.TestCase):
    def setup_root(self,path):
        root=Path(path)
        for source,_,mime in B.ARTIFACTS:
            p=root/source;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text('{"source":"retained"}\n' if mime=='application/json' else '# Preserved source\n')
        return root
    def test_exact_allowlist_bytes_and_immutable_versions(self):
        with tempfile.TemporaryDirectory() as t:
            root=self.setup_root(t);before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
            result=B.build(root=root);out=root/B.PREFIX/'v1'
            self.assertEqual(result['artifact_count'],8)
            self.assertEqual({p:p.read_bytes() for p in before},before)
            self.assertEqual(B.verify(out,root=root),result)
            with self.assertRaises(ValueError):B.build(root=root)
            self.assertFalse((out/'meta.env').exists())
    def test_corruption_and_source_drift_fail(self):
        with tempfile.TemporaryDirectory() as t:
            root=self.setup_root(t);result=B.build(root=root);out=root/B.PREFIX/'v1'
            target=root/result['artifacts'][0]['served_path'];target.write_text('corrupt')
            with self.assertRaises(ValueError):B.verify(out,root=root)
        with tempfile.TemporaryDirectory() as t:
            root=self.setup_root(t);result=B.build(root=root);out=root/B.PREFIX/'v1'
            (root/result['artifacts'][0]['original_path']).write_text('changed')
            with self.assertRaises(ValueError):B.verify(out,root=root)
    def test_traversal_and_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=self.setup_root(t)
            for version in ['../escape','.hidden','v0','v12345','/tmp/out']:
                with self.assertRaises(ValueError):B.build(root=root,version=version)
            source=root/B.ARTIFACTS[0][0];source.unlink();source.symlink_to(root/B.ARTIFACTS[1][0])
            with self.assertRaises(ValueError):B.build(root=root)
        with tempfile.TemporaryDirectory() as t:
            root=self.setup_root(t);(root/'data').mkdir();(root/'data/submission-review-20261101').symlink_to(root/'docs',target_is_directory=True)
            with self.assertRaises(ValueError):B.build(root=root)
    def test_invalid_metadata_is_not_copied(self):
        with tempfile.TemporaryDirectory() as t:
            root=self.setup_root(t)
            source=next(x[0] for x in B.ARTIFACTS if x[2]=='application/json');(root/source).write_text('{"n":NaN}')
            with self.assertRaises(ValueError):B.build(root=root)
            self.assertFalse((root/B.PREFIX/'v1').exists())
    def test_fixed_walkthrough_versions_preserve_original_copy_set(self):
        with tempfile.TemporaryDirectory() as t:
            root=self.setup_root(t);first=B.build(root=root)
            for source,_,mime in B.artifact_set('v3'):
                p=root/source
                if '/render-v3/' in source:
                    p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{"source":"new frozen render"}\n')
            second=B.build(root=root,version='v2',walkthrough='v3')
            self.assertEqual(B.verify(root/B.PREFIX/'v1',root=root),first)
            self.assertEqual(B.verify(root/B.PREFIX/'v2',root=root),second)
            self.assertTrue(all('/render-v2/' not in r['original_path'] for r in second['artifacts']))
            with self.assertRaises(ValueError):B.build(root=root,version='v3',walkthrough='../arbitrary')

if __name__=='__main__':unittest.main()
