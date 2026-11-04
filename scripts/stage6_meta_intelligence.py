#!/usr/bin/env python3
"""Plan/extract offline or explicitly run budgeted Meta task analysis and Q&A."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from task_intelligence import IntelligenceError, build_context, cached_analyses, plan_analysis, run_analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["extract", "plan", "run", "ask", "cached"])
    parser.add_argument("--interaction", required=True)
    parser.add_argument("--question")
    parser.add_argument("--output-root", default="data/task-intelligence")
    parser.add_argument("--max-images", type=int, default=12)
    parser.add_argument("--max-output-tokens", type=int, default=12000)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--allow-paid-api", action="store_true", help="Authorize one call on cache miss under the unchanged shared USD100/USD2 estimate guard.")
    args = parser.parse_args()
    try:
        if args.command == "extract":
            result = build_context(args.interaction, max_images=args.max_images)
        elif args.command == "cached":
            result = cached_analyses(args.interaction, output_root=args.output_root)
        else:
            mode = "ask" if args.command == "ask" or args.question is not None else "analysis"
            options = dict(mode=mode, question=args.question, max_images=args.max_images, max_output_tokens=args.max_output_tokens)
            if args.command == "plan":
                result = plan_analysis(args.interaction, **options)
                result = {k: v for k, v in result.items() if k not in {"prompt", "schema", "provider_schema", "context"}}
                result["network_used"] = False
            else:
                result = run_analysis(args.interaction, output_root=args.output_root,
                                      allow_paid_api=args.allow_paid_api, timeout=args.timeout_s, **options)
        print(json.dumps(result, indent=2, allow_nan=False))
    except Exception as exc:
        # Never dump arbitrary provider errors or environment values.
        safe = str(exc) if isinstance(exc, IntelligenceError) else type(exc).__name__
        print(json.dumps({"status": "failed", "error": safe}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
