"""Offline forked repair namespaces preserve failures and never resend requests."""
import concurrent.futures
import copy
import hashlib
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import admin_pipeline as A
import pipeline as P


class RepairAttemptReservationTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(dir=P.ROOT/'data')
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve()
        self.case=self.root/'data/source-case'
        self.case.mkdir(parents=True)
        self.source_sha=hashlib.sha256(b'original recording').hexdigest()
        P.write(self.case/'manifest.json',{'source':{'original_sha256':self.source_sha,'sha256':'b'*64}})
        self.manifest={'path':'data/source-case/manifest.json','sha256':P.sha(self.case/'manifest.json'),
                       'bytes':(self.case/'manifest.json').stat().st_size}
        self.base={'id':'run-original','source':{'sha256':self.source_sha},'phases':[
            {'name':'prepare','status':'complete','attempt':1,'files':[self.manifest],
             'result':{'case':'data/source-case'}},
            {'name':'procedure_proposal','status':'complete','attempt':1}]}
        self.dependency=self.root/'data/dependency.json'
        P.write(self.dependency,{'fixture':'never executed model'})
        (self.root/'admin_pipeline.py').write_bytes(Path(A.__file__).read_bytes())
        for module in [A,P]:
            binding=patch.object(module,'ROOT',self.root)
            binding.start();self.addCleanup(binding.stop)

    def runner(self,identifier):
        runner=A.AdminRunner(identifier,base_state=self.base,boundary='procedure_proposal')
        runner.profile={}
        return runner

    def test_forked_repairs_get_new_attempts_above_failed_outputs_and_reserved_siblings(self):
        failed=self.case/'flow-procedure-proposal-2'
        failed.mkdir();(failed/'metadata.json').write_bytes(b'unchanged partial metadata')
        (failed/'prompt.txt').write_bytes(b'unchanged prompt; no dispatch')
        original=copy.deepcopy(self.base)
        attempts=[]
        def execute(attempt):
            attempts.append(attempt)
            out=self.case/f'flow-procedure-proposal-{attempt}'
            out.mkdir();P.write(out/'proposal.json',{'fixture':attempt})
            return {'proposal':str(out/'proposal.json')},[out]
        first=self.runner('run-repair-fork-a')
        first.phase('procedure_proposal',[self.dependency],execute)
        second=self.runner('run-repair-fork-b')
        second.phase('procedure_proposal',[self.dependency],execute)
        self.assertEqual(attempts,[3,4])
        self.assertEqual(self.base,original)
        self.assertEqual((failed/'metadata.json').read_bytes(),b'unchanged partial metadata')
        self.assertEqual((failed/'prompt.txt').read_bytes(),b'unchanged prompt; no dispatch')
        for runner,attempt in [(first,3),(second,4)]:
            row=next(r for r in runner.state['phases']if r['name']=='procedure_proposal')
            self.assertEqual(row['attempt'],attempt)
            reservation=P.read(self.root/row['attempt_reservation']['path'])
            self.assertEqual(reservation['attempt'],attempt)
            self.assertEqual(reservation['run_id'],runner.id)
            self.assertIn(row['attempt_reservation'],row['files'])
            self.assertTrue(P.verify_receipts(row['files']))

    def test_abandoned_reservations_and_concurrent_forks_cannot_reuse_numbers(self):
        abandoned,reference=A.reserve_repair_attempt(self.base,'run-abandoned','procedure_proposal',2,root=self.root)
        self.assertEqual(abandoned,2)
        self.assertFalse((self.case/'flow-procedure-proposal-2').exists())
        with concurrent.futures.ThreadPoolExecutor(max_workers=4)as executor:
            results=list(executor.map(lambda n:A.reserve_repair_attempt(self.base,f'run-fork-{n}','procedure_proposal',2,root=self.root),range(6)))
        self.assertEqual(sorted(r[0]for r in results),list(range(3,9)))
        self.assertEqual(P.read(self.root/reference['path'])['run_id'],'run-abandoned')

    def test_uncertain_current_paid_attempt_is_preserved_and_not_redispatched(self):
        runner=self.runner('run-failed-paid')
        def failed(attempt):
            (self.case/f'flow-procedure-proposal-{attempt}').mkdir()
            raise RuntimeError('Preserved uncertain request receipt')
        runner.phase('procedure_proposal',[self.dependency],failed,optional=True)
        before=copy.deepcopy(runner.state)
        retry=Mock()
        with self.assertRaisesRegex(ValueError,'no automatic resend'):
            runner.phase('procedure_proposal',[self.dependency],retry,optional=True)
        retry.assert_not_called()
        self.assertEqual(runner.state,before)
        self.assertEqual(len(list((self.root/'.private/admin-attempt-reservations').glob('*/attempt-*.json'))),1)

    def test_source_hash_change_or_symlink_scan_fails_without_reservation(self):
        P.write(self.case/'manifest.json',{'source':{'original_sha256':'c'*64}})
        with self.assertRaisesRegex(ValueError,'manifest changed'):
            A.reserve_repair_attempt(self.base,'run-mismatch','procedure_proposal',2,root=self.root)
        self.assertFalse((self.root/'.private/admin-attempt-reservations').exists())
        P.write(self.case/'manifest.json',{'source':{'original_sha256':self.source_sha,'sha256':'b'*64}})
        outside=self.root/'other-case';outside.mkdir()
        (self.case/'flow-procedure-proposal-2').symlink_to(outside,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'Invalid preserved'):
            A.reserve_repair_attempt(self.base,'run-symlink','procedure_proposal',2,root=self.root)
        self.assertEqual(list(outside.iterdir()),[])


if __name__=='__main__':unittest.main()
