"""One immutable $5 suballocation inside the approved overnight $20 pass.

This never resets an older scope, reconciles a charge, renews an allowance or
dispatches inference. The nested allowance is checked inside the existing outer
admission lock via dispatch_guard, never by acquiring that same lock twice.
"""
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime,timezone

from increment_budget import remaining_increment
from meta_api import MetaError
from overnight_budget import OvernightMetaClient,PLAN as PARENT_PLAN,plan as parent_plan

ROOT=Path(__file__).resolve().parent
PLAN='output/overnight-pipeline-20261102/mechanism-subtask-budget-v1.json'
HISTORICAL_PLAN='output/constrained-kinematics-20261102/task-budget-plan.json'
FORMAT='interaction-twin-once-only-mechanism-suballocation'
APPROVED_SHA256='e4bc15e1913faf7859469a05c0ae064d34e99493af6c7abedcd870bff1a9b615'


def reference(root,relative):
    root=Path(root).resolve();path=root/relative
    if path.resolve()!=path or path.is_symlink()or not path.is_file()or path.stat().st_size>128000:
        raise ValueError('Ordinary bounded mechanism budget plan required')
    raw=path.read_bytes()
    return {'path':relative,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def load(root=ROOT):
    root=Path(root).resolve()
    if reference(root,PLAN)['sha256']!=APPROVED_SHA256:
        raise ValueError('Mechanism suballocation differs from its once-only approved digest')
    value=json.loads((root/PLAN).read_text())
    if (value.get('format')!=FORMAT or value.get('version')!=1 or value.get('additional_estimated_api_ceiling_usd')!='5.000000' or
        value.get('parent_plan_ref')!=reference(root,PARENT_PLAN)or value.get('historical_plan_ref')!=reference(root,HISTORICAL_PLAN)or
        value.get('automatic_renewal')is not False or value.get('existing_uncertainty_holds_preserved')is not True or
        value.get('historical_scope_reset')is not False or value.get('uncertain_paid_requests_must_not_be_retried')is not True or
        value.get('all_api_spending_since_baseline_counts')is not True or value.get('requires_validated_overnight_wrapper')is not True):
        raise ValueError('Immutable mechanism suballocation or parent binding changed')
    parent=parent_plan(root);baseline=value.get('budget_baseline',{})
    for field in ['settled_usd','reserved_usd']:
        amount=Decimal(baseline[field])
        if not amount.is_finite()or amount<0:raise ValueError('Invalid exact mechanism budget baseline')
    if sum(Decimal(baseline[k])for k in ['settled_usd','reserved_usd'])<sum(Decimal(parent['budget_baseline'][k])for k in ['settled_usd','reserved_usd']):
        raise ValueError('Mechanism suballocation predates its approved parent')
    return value


def create_once(status,*,root=ROOT):
    """Explicit one-time preparation from a caller's current guarded ledger view."""
    root=Path(root).resolve();parent_plan(root);parent=reference(root,PARENT_PLAN);old=reference(root,HISTORICAL_PLAN)
    if (root/PLAN).exists():
        raise ValueError('Mechanism suballocation already exists; no automatic renewal or baseline reset')
    # Validate the supplied conservative view, including held reservations. The
    # caller captures it under the existing overnight lock; no credentials read.
    baseline={key:status[key]for key in ['settled_usd','reserved_usd']}
    for amount in baseline.values():
        if not isinstance(amount,str)or not Decimal(amount).is_finite()or Decimal(amount)<0:raise ValueError('Exact current budget strings required')
    if status.get('cap_usd')!='100.000000'or status.get('per_request_cap_usd')!='2.000000':raise ValueError('Existing global/per-request guards must remain unchanged')
    if remaining_increment(parent_plan(root),status)<Decimal(status['per_request_cap_usd']):raise ValueError('Approved parent allowance cannot admit a request')
    value={'format':FORMAT,'version':1,'created_utc':datetime.now(timezone.utc).isoformat(),
           'authorization':'One named suballocation inside the existing explicitly approved $20 pass; no additional parent allowance.',
           'parent_plan_ref':parent,'historical_plan_ref':old,'additional_estimated_api_ceiling_usd':'5.000000','budget_baseline':baseline,
           'all_api_spending_since_baseline_counts':True,'automatic_renewal':False,'historical_scope_reset':False,
           'existing_uncertainty_holds_preserved':True,'uncertain_paid_requests_must_not_be_retried':True,
           'requires_validated_overnight_wrapper':True,'global_estimated_guard_usd':'100.000000','per_request_cap_usd':'2.000000'}
    path=root/PLAN
    if path.resolve()!=path or any(p.is_symlink()for p in [path,*path.parents]):raise ValueError('Mechanism budget output cannot use symlinks')
    raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode();fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return reference(root,PLAN)


def validate_profile(profile,*,root=ROOT):
    if profile.get('mechanism_task_budget_plan')!=PLAN or profile.get('overnight_task_budget_plan')!=PARENT_PLAN:
        raise ValueError('New mechanism allowance requires its exact approved outer profile')
    load(root);return PLAN


def scoped_client_class(base_class,*,root=ROOT):
    """Only an already wrapped child can construct this additional allowance."""
    if not isinstance(base_class,type)or not issubclass(base_class,OvernightMetaClient):
        raise ValueError('New mechanism allowance requires the validated overnight client wrapper')
    root=Path(root).resolve();load(root)
    class MechanismPassClient(base_class):
        def __init__(self,*,dispatch_guard=None,**kwargs):
            if dispatch_guard is not None and not callable(dispatch_guard):raise ValueError('Trusted dispatch guard required')
            if any(key in kwargs for key in ['task_budget_plan','overnight_plan','increment_plan','root']):
                raise ValueError('Mechanism suballocation cannot override parent scopes or baselines')
            def guarded():
                envelope=load(root);status=self.ledger.status()
                if remaining_increment(envelope,status)<Decimal(status['per_request_cap_usd']):
                    raise MetaError('new_mechanism_suballocation_estimate_limit_reached_before_transport')
                if dispatch_guard is not None:dispatch_guard()
            super().__init__(root=root,dispatch_guard=guarded,**kwargs)
    return MechanismPassClient
