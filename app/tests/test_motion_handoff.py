"""Portable motion handoff fail-closed source/resource and ZIP boundary checks."""
import copy,importlib.util,json,sys,tempfile,unittest,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from rehearse_motion_handoff import verify_motion
from inspect_motion_handoff import archive_bytes,validate
from tests.test_motion_catalog import MotionCatalogContracts


class MotionHandoffContracts(MotionCatalogContracts):
    def test_consumer_never_accepts_changed_source_asset_or_resource(self):
        root,catalog,record,ctx=self.registered();twin='fixture--v1'
        descriptor=catalog.describe(record['experiment_id'],twin_id=twin,**ctx)
        downloads={key:catalog.artifact(record['experiment_id'],key,**ctx) for key in ['raw','normalized','source-image-0','source-mask-0']}
        collected={'document':ctx['document'],'twin_id':twin,'source_program_sha256':ctx['program_sha256'],
            'asset_bytes':{'part':(root/'data/asset.glb').read_bytes()}}
        verify_motion(descriptor,downloads,collected)
        for mutation in ['program','physics','uri','mask']:
            d=copy.deepcopy(descriptor);b=dict(downloads)
            if mutation=='program':d['program_sha256']='0'*64
            if mutation=='physics':d['policy']['simulation_ready']=True
            if mutation=='uri':d['artifacts']['raw']['uri']='file:///tmp/source'
            if mutation=='mask':b['source-mask-0']+=b'corrupt'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):verify_motion(d,b,collected)


class BundleArchiveContracts(unittest.TestCase):
    def temp(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);return Path(t.name)

    def test_traversing_duplicate_and_symlink_members_rejected_without_extraction(self):
        for kind in ['traversal','duplicate','symlink']:
            root=self.temp();p=root/'bundle.zip'
            with zipfile.ZipFile(p,'w') as z:
                if kind=='traversal':z.writestr('../escape.txt',b'no')
                elif kind=='duplicate':z.writestr('item',b'a');z.writestr('item',b'b')
                else:
                    info=zipfile.ZipInfo('item');info.create_system=3;info.external_attr=(0o120777<<16);z.writestr(info,b'/tmp/out')
            with self.subTest(kind=kind),self.assertRaises(ValueError):archive_bytes(p)
            self.assertFalse((root/'escape.txt').exists())

    def test_manifest_corruption_rejected_before_creating_destination(self):
        root=self.temp();p=root/'bundle.zip';target=root/'relocated'
        manifest={'format':'interaction-twin-motion-inspection-bundle','version':1,'files':{'item.json':{'bytes':2,'sha256':'0'*64}}}
        with zipfile.ZipFile(p,'w') as z:z.writestr('bundle-manifest.json',json.dumps(manifest));z.writestr('item.json','{}')
        with self.assertRaisesRegex(ValueError,'hash/length'):validate(p,target)
        self.assertFalse(target.exists())


if __name__=='__main__':unittest.main()
