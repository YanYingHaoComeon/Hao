# SPDX-License-Identifier: MIT
"""Command-line entry point; authorized input/output paths are mandatory."""
from pathlib import Path
import argparse, pickle
import pandas as pd
from hao.common import load_config, output_directory, module, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["train", "nested", "explain"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", help="Local CSV, train/nested phases only")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", help="Trusted local fitted-model directory, explain phase")
    args = parser.parse_args()
    cfg = load_config(args.config); out = output_directory(args.output)
    if args.phase == "explain":
        if not args.model_dir: parser.error("--model-dir is required")
        folder = Path(args.model_dir)
        # Only deserialize models that you created and trust.
        with (folder / "model.pkl").open("rb") as handle: model = pickle.load(handle)
        if getattr(model, "positive_label", None) != cfg["schema"]["positive_label"]:
            raise ValueError("Explanation positive label differs from the fitted model.")
        if not model.columns:
            raise ValueError("A training-prior-only model has no feature SHAP plot.")
        background = pd.read_csv(folder / "background.csv")
        cases = pd.read_csv(folder / "explanation_input.csv")
        values, base, error = module("07_model_interpretation", "shap_values").explain(model, background, cases, cfg["interpretation"])
        pd.DataFrame(values, columns=cases.columns).to_csv(out / "shap_values.csv", index=False)
        write_json(out / "explanation.json", {"base_probability": base, "maximum_additivity_error": error,
            "positive_label": cfg["schema"]["positive_label"]})
        module("07_model_interpretation", "plots").plot_explanation(values, cases, base, out)
    else:
        if not args.data: parser.error("--data is required")
        x, y = module("01_data_preprocessing", "schema").read_data(args.data, cfg["schema"])
        write_json(out / "configuration.json", cfg)
        if args.phase == "train":
            from hao.workflow import train_holdout
            train_holdout(x, y, cfg, out)
        else: module("05_model_validation", "nested_cv").validate(x, y, cfg, out)


if __name__ == "__main__": main()
