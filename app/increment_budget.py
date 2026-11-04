"""An additional local spending envelope; the shared ledger remains authoritative.

This never loads credentials or changes/reconciles ledger entries. The owner runs
paid jobs serially. Each admission reserves headroom for the shared per-request
ceiling, including all retained uncertain charges.
"""
from decimal import Decimal
from pathlib import Path
import json

from meta_api import MetaClient, MetaError


PLAN = Path(__file__).resolve().parent / 'output/autonomous-improvements-20261031/plan.json'


def remaining_increment(plan, status):
    baseline = plan['budget_baseline']
    initial = Decimal(baseline['settled_usd']) + Decimal(baseline['reserved_usd'])
    current = Decimal(status['settled_usd']) + Decimal(status['reserved_usd'])
    limit = Decimal(plan['additional_estimated_api_ceiling_usd'])
    if not limit.is_finite() or not Decimal('0') <= limit <= Decimal('20') or current < initial:
        raise ValueError('Increment envelope or ledger baseline is invalid')
    return limit - (current - initial)


class IncrementMetaClient(MetaClient):
    def __init__(self, *, increment_plan=PLAN, **kwargs):
        super().__init__(**kwargs)
        self.increment_plan = Path(increment_plan)

    def request(self, *args, **kwargs):
        plan = json.loads(self.increment_plan.read_text())
        status = self.ledger.status()
        if remaining_increment(plan, status) < Decimal(status['per_request_cap_usd']):
            raise MetaError('additional_increment_estimate_limit_reached_before_transport')
        return super().request(*args, **kwargs)
