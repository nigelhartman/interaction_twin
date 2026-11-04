#!/usr/bin/env python3
"""One serial, budgeted robotics Q&A pass over the three real sample families."""
import datetime
from decimal import Decimal
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meta_budget import read_budget_status
from increment_budget import IncrementMetaClient
import meta_api

RUN = ROOT / 'output/overnight-robotics-20261102-v1'
PLAN = RUN / 'plan.json'
QUESTION = ('Which separately manipulable components and demonstrated actions can a robotics '
            'engineer reuse from this recording, and what must still be established before '
            'using this package for robot training? Cite the available evidence and keep '
            'unobserved contact, physical properties and execution unknown.')
IDS = [
    'flow-rigid-parts-jabra-20261031-v1--flow-procedure-1',
    'flow-meta-video-wallet-20261030-v1--flow-procedure-1',
    'flow-meta-video-camera-20261030-v1--flow-procedure-1',
]


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


class OvernightMetaClient(IncrementMetaClient):
    """Add a USD50 envelope; retain the earlier, stricter USD20 increment guard."""
    def request(self, *args, **kwargs):
        plan = json.loads(PLAN.read_text())
        status = self.ledger.status()
        baseline = plan['budget_baseline']
        initial = Decimal(baseline['settled_usd']) + Decimal(baseline['reserved_usd'])
        current = Decimal(status['settled_usd']) + Decimal(status['reserved_usd'])
        limit = Decimal(plan['additional_estimated_api_ceiling_usd'])
        if not Decimal('0') <= limit <= Decimal('50') or current < initial:
            raise meta_api.MetaError('invalid_overnight_budget_envelope')
        if limit - (current - initial) < Decimal(status['per_request_cap_usd']):
            raise meta_api.MetaError('overnight_budget_limit_before_transport')
        return super().request(*args, **kwargs)


def main():
    # Exclusive plan creation prevents silently replacing a spending baseline.
    RUN.mkdir(parents=True, exist_ok=True)
    before = read_budget_status(ROOT / '.secrets/meta-budget.sqlite3')
    plan = {
        'format': 'overnight-robotics-improvement-plan', 'version': 1,
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'additional_estimated_api_ceiling_usd': '50', 'budget_baseline': before,
        'global_estimated_cap_usd': '100', 'earlier_usd20_increment_guard_retained': True,
        'request_count_maximum': 3, 'max_images_per_request': 8,
        'max_total_output_tokens_per_request': 8000, 'automatic_retries': False,
        'case_ids': IDS, 'question': QUESTION,
        'pricing_verified': {'date': '2026-11-02',
                             'source': 'https://dev.meta.ai/docs/pricing-rate-limits',
                             'spark_input_per_million_usd': '1.25',
                             'spark_output_per_million_usd': '4.25'},
        'scope': 'Evidence-grounded descriptive answers; no physical commands or source edits.',
    }
    write_new(PLAN, plan)
    meta_api.MetaClient = OvernightMetaClient
    from task_intelligence import run_analysis
    cases = json.loads((ROOT / 'workspace/cases.json').read_text())['cases']
    summaries = []
    for twin_id in IDS:
        row = next(item for item in cases if item['id'] == twin_id)
        try:
            result = run_analysis(row['interaction_path'], root=ROOT,
                                  mode='ask', question=QUESTION, max_images=8,
                                  max_output_tokens=8000, timeout=300, allow_paid_api=True)
            summaries.append({'twin_id': twin_id, 'status': 'validated_cached_answer',
                              'request_key': result['request_key'],
                              'estimated_cost_usd': result['estimated_cost_usd'],
                              'answer_type': result['result']['answer_type']})
        except Exception as exc:
            # No automatic retry or provider/body/environment logging.
            summaries.append({'twin_id': twin_id, 'status': 'stopped_no_retry',
                              'error_type': type(exc).__name__})
            break
        print(json.dumps(summaries[-1]), flush=True)
    after = read_budget_status(ROOT / '.secrets/meta-budget.sqlite3')
    write_new(RUN / 'results.json', {'answers': summaries, 'budget_before': before,
                                   'budget_after': after})
    print(json.dumps({'finished': True, 'answers': len(summaries), 'budget_after': after}), flush=True)


if __name__ == '__main__':
    main()
