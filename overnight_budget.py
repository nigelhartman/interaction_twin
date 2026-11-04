"""Additional owner-approved overnight envelope; no credential or ledger edits.

A shared flock serializes overnight admissions through transport settlement.
Existing per-request/global ledger limits and historical increment guard remain.
Do not nest ``pass_guard`` around ``OvernightMetaClient``: each owns the lock.
"""
from __future__ import annotations
from contextlib import contextmanager
from decimal import Decimal
import fcntl
import json
import os
from pathlib import Path

from increment_budget import IncrementMetaClient, remaining_increment
from meta_api import MetaError
from meta_budget import read_budget_status

ROOT=Path(__file__).resolve().parent
PLAN='output/overnight-pipeline-20261102/task-budget-plan.json'
LOCK='.state/overnight-pipeline-budget.lock'


def _path(root,relative,*,exists=True):
    root=Path(root).resolve();candidate=root/relative
    if candidate.resolve()!=candidate or any(p.is_symlink()for p in [candidate,*candidate.parents]):
        raise ValueError('Overnight budget path escapes its ordinary project namespace')
    if exists and(not candidate.is_file()or candidate.stat().st_size>128000):
        raise ValueError('Bounded existing overnight budget file required')
    return candidate


def plan(root=ROOT):
    value=json.loads(_path(root,PLAN).read_text())
    limit=Decimal(value['additional_estimated_api_ceiling_usd'])
    if(not limit.is_finite()or not Decimal('0')<limit<=Decimal('20')or
       value.get('uncertain_paid_requests_must_not_be_retried')is not True or
       value.get('existing_uncertainty_holds_preserved')is not True or
       value.get('previous_rejected_localization_request_excluded')is not True or
       value.get('global_estimated_guard_usd')!='100.000000' or
       value.get('maximum_repair_rounds')!=2 or value.get('maximum_total_rounds')!=3):
        raise ValueError('Overnight authorization/budget contract changed')
    return value


def budget_status(*,root=ROOT,ledger_status=None):
    root=Path(root).resolve();value=plan(root)
    status=ledger_status if ledger_status is not None else read_budget_status(root/'.secrets/meta-budget.sqlite3')
    remaining=remaining_increment(value,status)
    return{'scope':'local_estimate_not_provider_balance','task_ceiling_usd':value['additional_estimated_api_ceiling_usd'],
        'task_remaining_estimated_usd':format(remaining,'.6f'),'admission_headroom_usd':status['per_request_cap_usd'],
        'settled_usd':status['settled_usd'],'reserved_usd':status['reserved_usd'],
        'request_count':status.get('request_count'),'existing_uncertainty_holds_preserved':True,
        'can_admit':remaining>=Decimal(status['per_request_cap_usd']) and status.get('blocked')is not True}


@contextmanager
def pass_guard(*,root=ROOT,status_reader=None):
    root=Path(root).resolve();folder=root/'.state'
    if folder.is_symlink():raise ValueError('Overnight lock directory cannot be a symlink')
    folder.mkdir(mode=0o700,exist_ok=True)
    path=_path(root,LOCK,exists=False)
    descriptor=os.open(path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    with os.fdopen(descriptor,'a+')as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
        status=(status_reader() if status_reader else read_budget_status(root/'.secrets/meta-budget.sqlite3'))
        state=budget_status(root=root,ledger_status=status)
        if not state['can_admit']:raise MetaError('overnight_task_estimate_limit_reached_before_transport')
        yield state


class OvernightMetaClient(IncrementMetaClient):
    """New <=$20 task + existing outer increment + unchanged shared $100 ledger."""
    def __init__(self,*,root=ROOT,task_budget_plan=PLAN,overnight_plan=None,dispatch_guard=None,**kwargs):
        self.project_root=Path(root).resolve()
        if overnight_plan is not None:
            if str(task_budget_plan)!=PLAN:raise ValueError('Do not override two overnight budget plans')
            task_budget_plan=overnight_plan
        if dispatch_guard is not None and not callable(dispatch_guard):raise ValueError('Dispatch fence must be callable')
        self.dispatch_guard=dispatch_guard
        candidate=Path(task_budget_plan)
        if candidate.is_absolute():candidate=candidate.relative_to(self.project_root)
        if candidate.as_posix()!=PLAN:raise ValueError('Fixed owner-approved overnight budget required')
        plan(self.project_root)
        super().__init__(**kwargs)

    def request(self,*args,**kwargs):
        with pass_guard(root=self.project_root,status_reader=self.ledger.status):
            if self.dispatch_guard is not None:self.dispatch_guard()
            return super().request(*args,**kwargs)
