"""Offline budget invariants against real SQLite transactions; never calls Meta."""
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from meta_budget import (BudgetError, BudgetExceeded, BudgetLedger, CostEstimate, SamPricing,
                         TextPricing, estimate_sam, estimate_text, micro_usd, text_cost, read_budget_status)


def pricing(**overrides):
    return TextPricing(**dict(model="muse-test", input_per_million_usd="1", output_per_million_usd="2",
                              source="https://dev.meta.ai/docs/pricing", observed_at="2026-10-30", **overrides))


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "budget.sqlite3"
        self.ledger = BudgetLedger(self.path)

    def reserve(self, identifier, dollars="1", ledger=None):
        # One million input tokens at the supplied dollars-per-million rate.
        p = TextPricing("muse-test", dollars, "2", "https://dev.meta.ai/docs/pricing", "2026-10-30")
        estimate = estimate_text(p, input_tokens=1_000_000, max_output_tokens=0)
        return (ledger or self.ledger).reserve(identifier, estimate, model=p.model, endpoint="/v1/chat/completions", kind="text")

    def test_exact_decimal_rounding_no_float_nan_negative(self):
        self.assertEqual(micro_usd("0.00000001"), 1)
        self.assertEqual(micro_usd(Decimal("1.234567")), 1234567)
        for invalid in (0.3, True, "NaN", "Infinity", "-1", None):
            with self.subTest(invalid=invalid), self.assertRaises(BudgetError): micro_usd(invalid)

    def test_initial_state_permission_no_silent_budget_change(self):
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.ledger.status()["remaining_estimated_usd"], "100.000000")
        for cap in ("99", "101"):
            with self.assertRaisesRegex(BudgetError, "persisted_budget_config_mismatch"):
                BudgetLedger(self.path, cap_usd=cap)
        self.assertEqual(self.ledger.status()["cap_usd"], "100.000000")

    def test_readonly_status_matches_and_never_creates_missing_file(self):
        self.reserve("read")
        self.assertEqual(read_budget_status(self.path), self.ledger.status())
        absent = Path(self.tmp.name) / "never-create.sqlite3"
        with self.assertRaises(BudgetError): read_budget_status(absent)
        self.assertFalse(absent.exists())

    def test_admission_is_durable_and_idempotent_but_transport_once(self):
        self.assertTrue(self.reserve("req1")["newly_reserved"])
        self.assertFalse(self.reserve("req1")["newly_reserved"])
        reopened = BudgetLedger(self.path)
        self.assertEqual(reopened.status()["reserved_usd"], "1.000000")
        reopened.mark_started("req1")
        with self.assertRaisesRegex(BudgetError, "request_not_startable"): self.ledger.mark_started("req1")
        with self.assertRaisesRegex(BudgetError, "request_id_conflict"): self.reserve("req1", "1.1")
        self.assertEqual(self.ledger.status()["request_count"], 1)

    def test_per_request_and_cumulative_caps(self):
        with self.assertRaisesRegex(BudgetExceeded, "per_request_cap_exceeded"): self.reserve("too-large", "2.000001")
        for index in range(50): self.reserve("r" + str(index), "2")
        self.assertTrue(self.ledger.status()["blocked"])
        with self.assertRaises(BudgetExceeded): self.reserve("overflow", "0.000001")
        self.assertEqual(self.ledger.status()["request_count"], 50)

    def test_independent_connections_cannot_race_through_cap(self):
        path = Path(self.tmp.name) / "concurrent.sqlite3"
        ledger = BudgetLedger(path, cap_usd="10", per_request_cap_usd="2")
        def worker(index):
            connection = BudgetLedger(path, cap_usd="10", per_request_cap_usd="2")
            try: self.reserve("c" + str(index), "2", ledger=connection); return True
            except BudgetExceeded: return False
        with ThreadPoolExecutor(max_workers=12) as pool: admitted = list(pool.map(worker, range(30)))
        self.assertEqual(sum(admitted), 5)
        self.assertEqual(ledger.status()["reserved_usd"], "10.000000")
        self.assertEqual(ledger.status()["request_count"], 5)

    def test_sql_failure_rolls_back_no_phantom_reservation(self):
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TRIGGER reject_insert BEFORE INSERT ON requests BEGIN SELECT RAISE(ABORT, 'offline_test'); END")
        with self.assertRaises(sqlite3.IntegrityError): self.reserve("fail")
        self.assertEqual(self.ledger.status()["reserved_usd"], "0.000000")
        self.assertEqual(self.ledger.status()["request_count"], 0)

    def test_timeout_retains_reservation_blocks_work_and_survives_restart(self):
        self.reserve("timeout", "1.5"); self.ledger.mark_started("timeout")
        self.ledger.mark_uncertain("timeout", "timeout")
        restored = BudgetLedger(self.path)
        self.assertEqual(restored.status()["reserved_usd"], "1.500000")
        self.assertTrue(restored.status()["blocked"])
        with self.assertRaises(BudgetExceeded): self.reserve("next")
        with self.assertRaises(BudgetError): self.ledger.release_no_charge("timeout", reason="cancelled_before_transport")
        self.ledger.settle("timeout", "0.8", usage={"input_tokens": 100, "output_tokens": 20})
        self.assertFalse(self.ledger.status()["blocked"])
        self.assertEqual(self.ledger.status()["settled_usd"], "0.800000")

    def test_missing_usage_fails_closed_and_arbitrary_data_not_saved(self):
        self.reserve("missing"); self.ledger.mark_started("missing")
        for usage in ({}, {"prompt": "sensitive"}, {"input_tokens": "bad"}, {"input_tokens": -1}):
            with self.assertRaises(BudgetError): self.ledger.settle("missing", "0", usage=usage)
        self.assertEqual(self.ledger.status()["reserved_usd"], "1.000000")
        self.ledger.mark_uncertain("missing", "missing_usage")
        self.assertTrue(self.ledger.status()["blocked"])
        self.assertNotIn(b"sensitive", self.path.read_bytes())

    def test_actual_overrun_recorded_in_full_and_blocks_until_review(self):
        self.reserve("over"); self.ledger.mark_started("over")
        result = self.ledger.settle("over", "1.1", usage={"input_tokens": 500, "output_tokens": 0})
        self.assertTrue(result["over_reserved"])
        self.assertEqual(self.ledger.status()["settled_usd"], "1.100000")
        self.assertEqual(self.ledger.status()["reserved_usd"], "0.000000")
        with self.assertRaises(BudgetExceeded): self.reserve("next")
        self.ledger.acknowledge_overage("over", authorization_reference="human-review-1")
        self.assertFalse(self.ledger.status()["blocked"])
        self.assertEqual(self.ledger.status()["remaining_estimated_usd"], "98.900000")

    def test_over_cap_is_not_hidden_and_acknowledgement_cannot_add_credit(self):
        self.reserve("over"); self.ledger.mark_started("over")
        self.ledger.settle("over", "101", usage={"input_tokens": 500, "output_tokens": 0})
        self.assertEqual(self.ledger.status()["over_cap_usd"], "1.000000")
        self.ledger.acknowledge_overage("over", authorization_reference="human-review-1")
        self.assertTrue(self.ledger.status()["blocked"])
        with self.assertRaises(BudgetExceeded): self.reserve("next")

    def test_settlement_idempotency_cannot_reduce_spent_history(self):
        self.reserve("s"); self.ledger.mark_started("s")
        self.ledger.settle("s", "0.3", usage={"input_tokens": 42, "output_tokens": 0})
        self.ledger.settle("s", "0.3", usage={"input_tokens": 42, "output_tokens": 0})
        with self.assertRaisesRegex(BudgetError, "settlement_conflict"):
            self.ledger.settle("s", "0", usage={"input_tokens": 0, "output_tokens": 0})
        with self.assertRaises(BudgetError): self.ledger.release_no_charge("s", reason="provider_confirmed_no_charge")
        self.assertEqual(self.ledger.status()["settled_usd"], "0.300000")

    def test_settlement_requires_complete_usage_and_saved_price_floor(self):
        self.reserve("priced"); self.ledger.mark_started("priced")
        with self.assertRaisesRegex(BudgetError, "missing_or_invalid_usage"):
            self.ledger.settle("priced", "1", usage={"input_tokens": 100})
        with self.assertRaisesRegex(BudgetError, "settlement_below_priced_usage"):
            self.ledger.settle("priced", "0", usage={"input_tokens": 100, "output_tokens": 20})
        self.assertEqual(self.ledger.status()["reserved_usd"], "1.000000")

    def test_sam_settlement_and_identity_bound_to_frame_price(self):
        p = SamPricing("sam3.1", "official-source", "2026-10-30", per_video_frame_usd="0.002")
        estimate = estimate_sam(p, video_frames=10)
        self.ledger.reserve("sam", estimate, model="sam3.1", endpoint="/v1/segment", kind="sam")
        self.ledger.mark_started("sam")
        with self.assertRaises(BudgetError):
            self.ledger.settle("sam", "0.02", usage={"video_frames": 10, "images": 0, "billable_video_frames": 10})
        self.ledger.settle("sam", estimate.cost_usd, usage=estimate.usage)
        self.assertEqual(self.ledger.status()["settled_usd"], "0.100000")

    def test_no_charge_release_requires_definite_before_transport_or_provider_confirmation(self):
        self.reserve("local"); self.ledger.release_no_charge("local", reason="local_validation_before_transport")
        self.reserve("sent"); self.ledger.mark_started("sent")
        with self.assertRaises(BudgetError): self.ledger.release_no_charge("sent", reason="cancelled_before_transport")
        self.ledger.release_no_charge("sent", reason="provider_confirmed_no_charge")
        self.assertEqual(self.ledger.status()["reserved_usd"], "0.000000")
        with self.assertRaises(BudgetError): self.ledger.mark_started("local")

    def test_provider_validation_http_can_release_but_5xx_cannot(self):
        self.reserve("validation"); self.ledger.mark_started("validation")
        for invalid in (None, 200, 302, 500, "400"):
            with self.assertRaises(BudgetError):
                self.ledger.release_no_charge("validation", reason="provider_http_rejected_before_generation", http_status=invalid)
        self.ledger.release_no_charge("validation", reason="provider_http_rejected_before_generation", http_status=400)
        self.assertEqual(self.ledger.status()["reserved_usd"], "0.000000")
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute("SELECT reason FROM requests").fetchone()[0], "provider_http_rejected_before_generation:400")

    def test_cache_and_reasoning_not_double_counted(self):
        p = TextPricing("muse", "1", "3", "official-source", "2026-10-30", "0.5", "3")
        estimate = text_cost(p, input_tokens=1000, cached_input_tokens=400, output_tokens=200, reasoning_tokens=100)
        separate = text_cost(p, input_tokens=1000, cached_input_tokens=400, output_tokens=100, reasoning_tokens=100, reasoning_in_output=False)
        self.assertEqual(estimate.cost_usd, "0.001400")
        self.assertEqual(estimate.cost_usd, separate.cost_usd)
        self.assertEqual(estimate.usage["billable_output_tokens"], 200)
        conservative = estimate_text(p, input_tokens=1000, max_output_tokens=200)
        self.assertEqual(conservative.cost_usd, "0.001600")
        with self.assertRaises(BudgetError): text_cost(p, input_tokens=1, cached_input_tokens=2, output_tokens=2)
        with self.assertRaises(BudgetError): text_cost(p, input_tokens=1, output_tokens=2, reasoning_tokens=3)

    def test_sampler_minimum_and_unknown_price(self):
        p = SamPricing("sam3.1", "official-source", "2026-10-30", per_image_usd="0.001", per_video_frame_usd="0.002")
        self.assertEqual(estimate_sam(p, images=3).cost_usd, "0.003000")
        self.assertEqual(estimate_sam(p, video_frames=1).cost_usd, "0.100000")
        self.assertEqual(estimate_sam(p, video_frames=51).cost_usd, "0.102000")
        self.assertEqual(estimate_sam(p, video_frames=1).usage["billable_video_frames"], 50)
        with self.assertRaisesRegex(BudgetError, "unknown_pricing"):
            estimate_sam(SamPricing("sam", "official", "2026-10-30"), images=1)
        with self.assertRaises(BudgetError): estimate_sam(p)

    def test_unknown_or_mismatched_estimates_rejected(self):
        p = pricing(); estimate = estimate_text(p, input_tokens=100, max_output_tokens=100)
        with self.assertRaises(BudgetError): self.ledger.reserve("bad", estimate, model="other", endpoint="/v1/chat", kind="text")
        with self.assertRaises(BudgetError): estimate_text(TextPricing("m", None, "1", "official", "2026-10-30"), input_tokens=1, max_output_tokens=1)
        with self.assertRaises(BudgetError): self.ledger.reserve("new\nline", estimate, model=p.model, endpoint="/v1/chat", kind="text")
        with self.assertRaises(BudgetError):
            self.ledger.reserve("unknown", CostEstimate(0, {"model": p.model, "kind": "text"}, {"input_tokens": 0}), model=p.model, endpoint="/v1/chat", kind="text")
        malicious = CostEstimate(1000000, {**estimate.pricing, "prompt": "private-prompt"}, estimate.usage)
        with self.assertRaises(BudgetError):
            self.ledger.reserve("payload", malicious, model=p.model, endpoint="/v1/chat", kind="text")
        under = CostEstimate(0, estimate.pricing, estimate.usage)
        with self.assertRaisesRegex(BudgetError, "under_reserved_estimate"):
            self.ledger.reserve("under", under, model=p.model, endpoint="/v1/chat", kind="text")
        self.assertNotIn(b"private-prompt", self.path.read_bytes())


class BoundedUnknownBudgetTests(unittest.TestCase):
    setUp = BudgetTests.setUp
    reserve = BudgetTests.reserve
    """Fixtures authorize only a pinned, reviewed single-attempt text timeout."""

    def make_uncertain(self, request_id="bounded", *, ledger=None, rate="1.25", reason="timeout", http_status=None):
        ledger = ledger or self.ledger
        p = TextPricing("muse-spark-1.3", rate, "4.25", "https://dev.meta.ai/docs/pricing-rate-limits", "2026-10-30", cached_input_per_million_usd="0.15")
        estimate = estimate_text(p, input_tokens=1048576, max_output_tokens=16000)
        ledger.reserve(request_id, estimate, model=p.model, endpoint="/v1/chat/completions", kind="text")
        ledger.mark_started(request_id); ledger.mark_uncertain(request_id, reason)
        evidence = Path(self.tmp.name) / (request_id + "-evidence"); evidence.mkdir()
        receipt = {"request_id":request_id,"provider":"meta_api","inference_location":"hosted_Meta_Model_API",
                   "model":p.model,"endpoint":"/v1/chat/completions","kind":"text","estimate_usd":estimate.cost_usd,
                   "pricing":p.snapshot(),"status":"transport_error_charge_uncertain","failure_phase":"transport",
                   "automatic_retries":False,"contributor_mode":False,"request_payload_saved":False,
                   "payload_sha256":"a"*64,"elapsed_s":300.01}
        if http_status is not None: receipt["http_status"] = http_status
        receipt_path = evidence / "meta-request.json"
        source_path = evidence / "metadata.json"
        receipt_path.write_text(json.dumps(receipt))
        source_path.write_text(json.dumps({"source_sha256":"b"*64,"frame_index_sha256":"c"*64}))
        transport_path = Path(__file__).resolve().parents[1] / "meta_api.py"
        return {"request_receipt_path":str(receipt_path),"request_receipt_sha256":hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                "source_record_path":str(source_path),"source_record_sha256":hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "source_sha256":"b"*64,"frame_index_sha256":"c"*64,"request_payload_sha256":"a"*64,
                "transport_source_sha256":hashlib.sha256(transport_path.read_bytes()).hexdigest(),
                "context_input_tokens":1048576,"max_output_tokens":16000,"attempt_count":1,"n":1,
                "tools":False,"background":False,"other_billable_modalities":False,
                "total_output_limit_includes_reasoning":True,"pricing_tier":"standard","timeout_confirmed":True}

    def ack(self, proof, request_id="bounded", ledger=None):
        return (ledger or self.ledger).acknowledge_bounded_text_uncertainty(
            request_id,proof=proof,authorization_reference="user-20261030-estimated100usd")

    def test_bounded_ack_preserves_all_money_unknown_status_and_immutable_audit(self):
        proof=self.make_uncertain();before=self.ledger.status()
        result=self.ack(proof);after=self.ledger.status()
        self.assertTrue(result["newly_acknowledged"]);self.assertTrue(result["unknown_usage"])
        self.assertEqual(result["reserved_usd"],"1.378720")
        self.assertEqual(before["reserved_usd"],after["reserved_usd"])
        self.assertEqual(before["remaining_estimated_usd"],after["remaining_estimated_usd"])
        self.assertEqual(after["settled_usd"],"0.000000")
        self.assertFalse(after["blocked"]);self.assertTrue(after["unknown_usage"])
        self.assertEqual(after["bounded_unknown_count"],1);self.assertEqual(after["bounded_unknown_usd"],"1.378720")
        self.assertEqual(after["unknown_usage_count"],1)
        self.assertEqual(read_budget_status(self.path),after)
        reopened=BudgetLedger(self.path);self.assertEqual(reopened.status(),after)
        with sqlite3.connect(self.path) as db:
            row=db.execute("SELECT state,actual,usage_json,reserved FROM requests").fetchone()
            self.assertEqual(row,("bounded_unknown",None,None,1378720))
            self.assertEqual(db.execute("SELECT retained_reservation FROM bounded_unknown_audits").fetchone()[0],1378720)
            with self.assertRaises(sqlite3.IntegrityError):db.execute("UPDATE bounded_unknown_audits SET retained_reservation=0")
            with self.assertRaises(sqlite3.IntegrityError):db.execute("DELETE FROM bounded_unknown_audits")

    def test_ack_is_idempotent_and_cannot_retry_or_release(self):
        proof=self.make_uncertain();self.ack(proof)
        result=self.ack(proof);self.assertFalse(result["newly_acknowledged"])
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM bounded_unknown_audits").fetchone()[0],1)
        with self.assertRaises(BudgetError):self.ledger.mark_started("bounded")
        for reason in ("provider_confirmed_no_charge","cancelled_before_transport"):
            with self.assertRaises(BudgetError):self.ledger.release_no_charge("bounded",reason=reason)
        modified=dict(proof,n=2)
        with self.assertRaisesRegex(BudgetError,"bounded_acknowledgement_conflict"):self.ack(modified)
        self.assertEqual(self.ledger.status()["reserved_usd"],"1.378720")

    def test_bounded_hold_remains_in_concurrent_admission_until_cap(self):
        proof=self.make_uncertain();self.ack(proof)
        def worker(index):
            ledger=BudgetLedger(self.path)
            try:self.reserve("later-"+str(index),"2",ledger=ledger);return True
            except BudgetExceeded:return False
        with ThreadPoolExecutor(max_workers=12) as pool: admitted=list(pool.map(worker,range(60)))
        self.assertEqual(sum(admitted),49)
        status=self.ledger.status()
        self.assertEqual(status["reserved_usd"],"99.378720")
        self.assertEqual(status["remaining_estimated_usd"],"0.621280")
        self.reserve("exact-final","0.621280")
        self.assertTrue(self.ledger.status()["blocked"])
        with self.assertRaises(BudgetExceeded):self.reserve("over-final","0.000001")
        self.assertEqual(self.ledger.status()["reserved_usd"],"100.000000")
        self.assertEqual(self.ledger.status()["bounded_unknown_usd"],"1.378720")

    def test_late_real_usage_can_settle_while_audit_remains(self):
        proof=self.make_uncertain();self.ack(proof)
        self.ledger.settle("bounded","0.000210",usage={"input_tokens":100,"output_tokens":20})
        status=self.ledger.status()
        self.assertEqual(status["settled_usd"],"0.000210")
        self.assertEqual(status["reserved_usd"],"0.000000")
        self.assertFalse(status["unknown_usage"]);self.assertEqual(status["bounded_unknown_count"],0)
        self.assertEqual(status["bounded_unknown_usd"],"0.000000")
        with sqlite3.connect(self.path) as db:self.assertEqual(db.execute("SELECT COUNT(*) FROM bounded_unknown_audits").fetchone()[0],1)

    def test_bad_proof_and_unknown_pricing_remain_blocked(self):
        mutations=[{"attempt_count":2},{"n":2},{"n":True},{"tools":True},{"background":True},
                   {"other_billable_modalities":True},{"total_output_limit_includes_reasoning":False},
                   {"context_input_tokens":1000},{"max_output_tokens":24001},{"max_output_tokens":100},
                   {"pricing_tier":"unknown"},{"timeout_confirmed":False},
                   {"request_payload_sha256":"d"*64},{"source_sha256":"d"*64},
                   {"transport_source_sha256":"d"*64},{"request_receipt_sha256":"d"*64}]
        proof=self.make_uncertain()
        for mutation in mutations:
            with self.subTest(mutation=mutation),self.assertRaises(BudgetError):self.ack(dict(proof,**mutation))
            self.assertTrue(self.ledger.status()["blocked"])
        with sqlite3.connect(self.path) as db:self.assertEqual(db.execute("SELECT COUNT(*) FROM bounded_unknown_audits").fetchone()[0],0)
        other_path=Path(self.tmp.name)/"other.sqlite3";other=BudgetLedger(other_path)
        bad_price=self.make_uncertain("wrong-price",ledger=other,rate="1.00")
        with self.assertRaisesRegex(BudgetError,"frozen_estimate_not_verified_full_context_bound"):
            self.ack(bad_price,"wrong-price",ledger=other)
        self.assertTrue(other.status()["blocked"])

    def test_sam_generic_unknown_and_5xx_are_ineligible(self):
        proof=self.make_uncertain("http5xx",http_status=500)
        with self.assertRaisesRegex(BudgetError,"http_error_not_eligible"):
            self.ack(proof,"http5xx")
        other=BudgetLedger(Path(self.tmp.name)/"generic.sqlite3")
        proof2=self.make_uncertain("generic",ledger=other,reason="unknown_charge")
        with self.assertRaisesRegex(BudgetError,"request_not_eligible"):
            self.ack(proof2,"generic",ledger=other)
        sam=BudgetLedger(Path(self.tmp.name)/"sam.sqlite3")
        p=SamPricing("sam-3.1","official-source","2026-10-30",per_image_usd="0.0025")
        sam.reserve("sam",estimate_sam(p,images=1),model=p.model,endpoint="/v1/responses",kind="sam")
        sam.mark_started("sam");sam.mark_uncertain("sam","timeout")
        with self.assertRaisesRegex(BudgetError,"request_not_eligible"):
            self.ack(proof,"sam",ledger=sam)
        self.assertTrue(sam.status()["blocked"])

    def test_tampered_or_unbound_source_evidence_is_rejected(self):
        proof=self.make_uncertain()
        source=Path(proof["source_record_path"])
        source.write_text(json.dumps({"source_sha256":"d"*64,"frame_index_sha256":"c"*64}))
        with self.assertRaisesRegex(BudgetError,"bounded_proof_file_hash_mismatch"):self.ack(proof)
        changed=dict(proof,source_record_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        with self.assertRaisesRegex(BudgetError,"source_record_hash_identity_mismatch"):self.ack(changed)
        self.assertTrue(self.ledger.status()["blocked"])

    def test_stale_preupgrade_admission_connections_fail_closed(self):
        proof=self.make_uncertain();self.ack(proof)
        # Simulates an old loaded module, whose connections do not register the
        # reviewed policy function and whose totals would omit the new state.
        with sqlite3.connect(self.path) as old:
            with self.assertRaises(sqlite3.OperationalError):
                old.execute("INSERT INTO requests (request_id,identity,model,endpoint,kind,state,reserved,estimate_json) SELECT 'old-admission',identity,model,endpoint,kind,'reserved',reserved,estimate_json FROM requests WHERE request_id='bounded'")
        self.assertEqual(self.ledger.status()["request_count"],1)
        # Current reviewed connections still account the bound and admit safely.
        self.reserve("current-admission","1")
        self.assertEqual(self.ledger.status()["reserved_usd"],"2.378720")

    def test_acknowledgement_failure_rolls_back_audit_and_state_together(self):
        proof=self.make_uncertain()
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TRIGGER fail_bounded_state BEFORE UPDATE ON requests WHEN NEW.state='bounded_unknown' BEGIN SELECT RAISE(ABORT,'offline_state_failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):self.ack(proof)
        self.assertTrue(self.ledger.status()["blocked"])
        self.assertEqual(self.ledger.status()["reserved_usd"],"1.378720")
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM bounded_unknown_audits").fetchone()[0],0)
            self.assertEqual(db.execute("SELECT state FROM requests").fetchone()[0],"uncertain")



if __name__ == "__main__": unittest.main()
