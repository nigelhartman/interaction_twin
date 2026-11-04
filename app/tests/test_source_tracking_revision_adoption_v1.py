"""Reuse real workflow isolation/race tests for the generic tracking adapter."""
import source_tracking_revision_adoption_v1 as C
from tests import test_laptop_cached_revision_adoption_20261103 as F


class SourceTrackingAdoptionTests(F.CachedRevisionAdoptionTests):
    def prepare(self):
        return C.build(root=self.root, contract=self.contract)

    def adopt(self, plan):
        return C.adopt(plan, root=self.root, contract=self.contract)

    def test_source_strategy_is_explicit(self):
        plan = self.prepare()
        self.assertEqual(plan['format'], C.FORMAT)
        self.assertEqual(plan['result']['repair_execution'][0]['strategy'], C.STRATEGY)
        self.assertFalse(plan['automatic_publication'])
