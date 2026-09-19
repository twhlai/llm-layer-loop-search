#!/usr/bin/env python3
"""
RYS Looped Layer Evaluator

Systematically evaluates looped attention layer configurations from
list of candidate layers:
1. Generate modified GGUF with looped layers
2. Start llama-server with the modified model
3. Run math + EQ probes
4. Score and record results
5. Print live results table
6. Kill server, repeat

Usage:
    python sweep.py \
        --model /path/to/model.gguf \
        --llama-server /path/to/llama-server \
        --tmpdir /dev/shm/rys \
        --candidates 8..24 \
        --results results.jsonl
"""

import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

from gguf_surgery import build_gguf_from_path
from layer_path import parse_layer_list
from ls_utils import wait_for_server, start_server, stop_server, dump_server_log, query_model
from math_probe import MATH_QUESTIONS, score_math_response
from eq_probe import EQ_SCENARIOS, build_eq_prompt, parse_eq_response, score_eq_response
from reasoning_probe import REASONING_QUESTIONS, score_reasoning_response


# Server config
DEFAULT_PORT = 8099


def run_math_probe(port: int) -> float:
    """Run all math questions and return average score (0-1)."""
    scores = []
    for question, answer in MATH_QUESTIONS:
        response = query_model(question, port, max_tokens=48)
        if response is not None:
            score = score_math_response(answer, response)
            scores.append(score)
        else:
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0


def run_eq_probe(port: int) -> float:
    """Run all EQ scenarios and return average score (0-100)."""
    scores = []
    for scenario in EQ_SCENARIOS:
        prompt = build_eq_prompt(scenario)
        response = query_model(prompt, port, max_tokens=48)
        if response is not None:
            predicted = parse_eq_response(response, len(scenario["emotions"]))
            score = score_eq_response(scenario["reference"], predicted)
            scores.append(score)
        else:
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0


def run_reasoning_probe(port: int) -> dict:
    """Run all reasoning questions, return scores by category and overall."""
    by_category = {}
    for q in REASONING_QUESTIONS:
        cat = q["type"]
        if cat not in by_category:
            by_category[cat] = []
        response = query_model(q["prompt"], port, max_tokens=512)
        score = score_reasoning_response(q, response)
        by_category[cat].append(score)

    # Per-category averages
    cat_scores = {}
    for cat, scores in by_category.items():
        cat_scores[cat] = sum(scores) / len(scores) if scores else 0.0

    # Overall reasoning score (0-1)
    all_scores = [s for scores in by_category.values() for s in scores]
    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0

    return {"categories": cat_scores, "overall": overall}


def run_evaluation(port: int) -> dict:
    """Run all probes and return results."""
    math_score = run_math_probe(port)
    eq_score = run_eq_probe(port)
    reasoning = run_reasoning_probe(port)
    return {
        "math_score": math_score,
        "eq_score": eq_score,
        "reasoning_score": reasoning["overall"],
        "reasoning_cats": reasoning["categories"],
    }


def print_results_table(results: list[dict], baseline: dict | None = None):
    """Print a live-updating results table."""
    width = 90
    print("\n" + "=" * width)
    print(f"{'Config':>14} {'Math':>8} {'EQ':>8} {'Reason':>8} "
          f"{'Math Δ':>8} {'EQ Δ':>8} {'Reas Δ':>8} {'Combined Δ':>11}")
    print("-" * width)

    if baseline:
        brs = baseline.get('reasoning_score', 0)
        print(f"{'BASELINE':>14} "
              f"{baseline['math_score']:>8.4f} {baseline['eq_score']:>8.2f} {brs:>8.2%} "
              f"{'---':>8} {'---':>8} {'---':>8} {'---':>11}")
        print("-" * width)

    for r in results:
        config = f"{set(r['repeat_layers'])}x{r['repeat_factor']}"
        rs = r.get('reasoning_score', 0)

        if baseline:
            math_delta = r['math_score'] - baseline['math_score']
            eq_delta = r['eq_score'] - baseline['eq_score']
            reas_delta = rs - baseline.get('reasoning_score', 0)
            # Combined: weight EQ and reasoning more than math
            combined = eq_delta + (reas_delta * 100)
            math_d = f"{math_delta:>+8.4f}"
            eq_d = f"{eq_delta:>+8.2f}"
            reas_d = f"{reas_delta:>+8.2%}"
            comb_d = f"{combined:>+11.2f}"
            all_pos = "*" if math_delta > 0 and eq_delta > 0 and reas_delta > 0 else " "
        else:
            math_d = eq_d = reas_d = comb_d = "---"
            all_pos = " "

        print(f"{config:>14} "
              f"{r['math_score']:>8.4f} {r['eq_score']:>8.2f} {rs:>8.2%} "
              f"{math_d} {eq_d} {reas_d} {comb_d}  {all_pos}")

    print("=" * width)
    sys.stdout.flush()


def generate_layer_path(n_layers: int, layer_subset: tuple[int,...], repeat_factor: int) -> list[int]:
    new_list = []
    for i in range(n_layers):
        if i in layer_subset:
            new_list.extend([-i] * (repeat_factor-1))
        new_list.append(i)
    return new_list


def main():
    parser = argparse.ArgumentParser(description="RYS Looped Layer Evaluator")
    parser.add_argument("--model", required=True, help="Path to input GGUF model")
    parser.add_argument("--llama-server", required=True, help="Path to llama-server binary")
    parser.add_argument("--tmpdir", default="/dev/shm/rys",
                        help="Temp directory for modified GGUFs (use tmpfs/RAM)")
    parser.add_argument("--results", default="rys_results.jsonl",
                        help="Output results file (JSONL)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--candidates", required=True, help="List of candidate layers to evaluate")
    parser.add_argument("--num-loops", type=int, default=1,
                        help="Number of looped single layers to include")
    parser.add_argument("--repeat-factor", type=int, default=2,
                        help="Number of times layer attention is repeated")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline run (use if already in results)")
    parser.add_argument("--server-args", nargs=argparse.REMAINDER, default=[],
                        help="Extra args to pass to llama-server (must be last)")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    tmpdir = Path(args.tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    candidate_list = parse_layer_list(args.candidates)

    results_path = Path(args.results)
    results = []
    baseline = None

    # Load existing results if resuming
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    if entry.get("is_baseline"):
                        baseline = entry
                    else:
                        results.append(entry)
        print(f"Loaded {len(results)} existing results + baseline={baseline is not None}")

    # Run baseline (unmodified model)
    if not args.skip_baseline and baseline is None:
        print("\n>>> Running BASELINE evaluation...")
        proc = start_server(args.llama_server, str(model_path), tmpdir, args.port, args.server_args)
        try:
            if not wait_for_server(args.port):
                print("ERROR: Server failed to start for baseline", file=sys.stderr)
                dump_server_log(proc)
                stop_server(proc)
                sys.exit(1)

            print("  Server ready. Running probes...")
            eval_result = run_evaluation(args.port)
            baseline = {
                "is_baseline": True,
                "math_score": eval_result["math_score"],
                "eq_score": eval_result["eq_score"],
                "reasoning_score": eval_result["reasoning_score"],
                "reasoning_cats": eval_result.get("reasoning_cats", {}),
                "timestamp": datetime.now().isoformat(),
            }

            with open(results_path, "a") as f:
                f.write(json.dumps(baseline) + "\n")

            brs = baseline['reasoning_score']
            print(f"  Baseline: math={baseline['math_score']:.4f} eq={baseline['eq_score']:.2f} reasoning={brs:.2%}")
        finally:
            stop_server(proc)

    # Get model layer count from the GGUF metadata
    from gguf import GGUFReader
    reader = GGUFReader(str(model_path), 'r')
    arch_field = reader.get_field('general.architecture')
    arch = arch_field.contents()
    block_count_field = reader.get_field(f'{arch}.block_count')
    n_layers = block_count_field.contents()
    print(f"\nModel: {model_path.name}")
    print(f"Architecture: {arch}, Layers: {n_layers}")

    # Generate sweep configurations
    configs = list(itertools.combinations(candidate_list, args.num_loops))

    # Filter out already-completed configs
    done = {tuple(r["repeat_layers"]) for r in results}
    configs = [c for c in configs if c not in done]

    print(f"Configs to test: {len(configs)}")

    print_results_table(results, baseline)

    for layer_subset in configs:
        config_str = f"{str(layer_subset).replace(" ","")}x{args.repeat_factor}"
        print(f"\n>>> Testing config {config_str}")

        # Generate modified GGUF
        modified_path = tmpdir / f"rys_{config_str}.gguf"
        print(f"  Generating modified GGUF...")
        try:
            layer_path = generate_layer_path(n_layers, layer_subset, args.repeat_factor)
            build_gguf_from_path(
                str(model_path), str(modified_path),
                layer_path, verbose=False
            )
        except Exception as e:
            print(f"  ERROR generating GGUF: {e}", file=sys.stderr)
            continue

        # Start server with modified model
        print(f"  Starting server...")
        proc = start_server(
            args.llama_server, str(modified_path), tmpdir, args.port, args.server_args
        )

        try:
            if not wait_for_server(args.port):
                print(f"  ERROR: Server failed to start for {config_str}", file=sys.stderr)
                dump_server_log(proc)
                print(f"  Check server log above for details.", file=sys.stderr)
                continue

            print(f"  Server ready. Running probes...")
            eval_result = run_evaluation(args.port)

            entry = {
                "repeat_layers": layer_subset,
                "repeat_factor": args.repeat_factor,
                "math_score": eval_result["math_score"],
                "eq_score": eval_result["eq_score"],
                "reasoning_score": eval_result["reasoning_score"],
                "reasoning_cats": eval_result.get("reasoning_cats", {}),
                "timestamp": datetime.now().isoformat(),
            }

            results.append(entry)

            # Append to results file
            with open(results_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

            print_results_table(results, baseline)

        finally:
            stop_server(proc)

            # Clean up modified GGUF to free tmpfs space
            if modified_path.exists():
                modified_path.unlink()
                print(f"  Cleaned up {modified_path.name}")

    print("\n\nSweep complete!")
    print_results_table(results, baseline)


if __name__ == "__main__":
    main()
