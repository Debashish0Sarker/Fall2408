from __future__ import annotations
import argparse
import os
import pandas as pd
from .data import load_dataset
from .config import ScoreConfig
from .oracle import compute_scores, build_oracle_contexts
from .train import train_all_models
from .evaluate import evaluate_and_plot

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to dataset_30k.csv")
    ap.add_argument("--out", default="artifacts", help="Artifact output directory")
    ap.add_argument("--results", default="results", help="Results output directory")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.results, exist_ok=True)

    cfg = ScoreConfig()

    print("[1/4] Loading dataset...")
    df = load_dataset(args.csv)

    print("[2/4] Computing scores + oracle labels...")
    scores = compute_scores(df, cfg)
    ctx, _best = build_oracle_contexts(df, scores)

    # Save artifacts
    ctx_path = os.path.join(args.out, "contexts.csv")
    scores_path = os.path.join(args.out, "scores.csv")
    ctx.to_csv(ctx_path, index=False)
    scores.to_csv(scores_path, index=False)
    print("  saved:", ctx_path)
    print("  saved:", scores_path)

    print("[3/4] Training models...")
    train_all_models(ctx, outdir=args.out, seed=args.seed)

    print("[4/4] Evaluating + plotting...")
    scores_reload = pd.read_csv(scores_path)
    ctx_reload = pd.read_csv(ctx_path)
    results = evaluate_and_plot(ctx_reload, scores_reload, artifact_dir=args.out, results_dir=args.results)

    print("\nDone. Key outputs:")
    print(" -", os.path.join(args.results, "model_comparison.csv"))
    print(" -", os.path.join(args.results, "plots"))
    print(results)

if __name__ == "__main__":
    main()
