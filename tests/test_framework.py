# SPDX-License-Identifier: MIT
"""Structural tests use generated generic observations, never research data."""
from pathlib import Path
import copy, json, os, tempfile, unittest
import numpy as np
import pandas as pd
from hao.common import ROOT, load_config, module, output_directory
from hao.workflow import train_holdout


def synthetic(seed=81, n=120):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n); b = rng.normal(size=n)
    category = rng.integers(0, 2, size=n)
    probability = 1/(1+np.exp(-(.6*a-.4*b+.3*category)))
    y = rng.binomial(1, probability)
    x = pd.DataFrame({"measure_a": a, "measure_b": b, "category_a": category.astype(float)})
    return x, y


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "config/example.json")
        self.x, self.y = synthetic()

    def test_configurable_column_mapping_and_positive_label(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = self.x.rename(columns={"measure_a":"custom_a", "measure_b":"custom_b", "category_a":"custom_c"})
            frame["custom_target"] = np.where(self.y, "event", "reference")
            path = Path(directory) / "synthetic.csv"; frame.to_csv(path, index=False)
            schema = {"continuous":["custom_a","custom_b"], "categorical":["custom_c"],
                "outcome":"custom_target", "positive_label":"event"}
            read = module("01_data_preprocessing", "schema").read_data
            x, y = read(path, schema)
            np.testing.assert_array_equal(y, self.y)
            schema["positive_label"] = "reference"
            np.testing.assert_array_equal(read(path, schema)[1], 1-self.y)
            self.assertEqual(list(x.columns), ["custom_a","custom_b","custom_c"])

    def test_category_mapping_is_training_only(self):
        encode = module("01_data_preprocessing", "schema").encode_partition
        a = pd.DataFrame({"c":["alpha","beta",None]})
        b = pd.DataFrame({"c":["alpha",None]})
        x, z, mapping = encode(a, b, ["c"])
        self.assertTrue(np.isnan(z.c.iloc[1])); self.assertEqual(x.c.iloc[0], 0)
        with self.assertRaises(ValueError): encode(a, pd.DataFrame({"c":["new"]}), ["c"])
        with self.assertRaises(ValueError): encode(pd.DataFrame({"c":["a","b","c"]}), pd.DataFrame({"c":["a"]}), ["c"])

    def test_split_seed_reproducibility(self):
        split = module("01_data_preprocessing", "schema").split_indices
        a, b = split(self.y, self.cfg["split"])
        c, d = split(self.y, self.cfg["split"])
        np.testing.assert_array_equal(a, c); self.assertFalse(set(a) & set(b))
        changed = dict(self.cfg["split"], seed=903)
        self.assertFalse(np.array_equal(a, split(self.y, changed)[0]))

    def test_all_seven_algorithms_and_configured_parameters(self):
        registry = module("04_model_development", "registry")
        self.assertEqual(set(registry.FILES), set(self.cfg["models"]))
        for name, options in self.cfg["models"].items():
            settings = dict(options["params"])
            if name == "ann": settings["max_iter"] = 150
            predictor = registry.Predictor(name, settings, 904).fit(self.x, self.y)
            p = predictor.predict(self.x)
            self.assertEqual(p.shape, (len(self.y),)); self.assertTrue(np.isfinite(p).all())
            self.assertTrue(((p>=0)&(p<=1)).all())
            if name != "lr": self.assertEqual(predictor.model.get_params()["random_state"], 904)

    def test_standardizer_uses_training_only(self):
        predictor = module("04_model_development", "registry").Predictor("svm", {"kernel":"linear"}, 3).fit(self.x, self.y)
        np.testing.assert_allclose(predictor.scaler.mean_, self.x.mean())
        before = predictor.scaler.mean_.copy()
        predictor.predict(self.x+100)
        np.testing.assert_array_equal(before, predictor.scaler.mean_)

    def test_empty_selection_is_explicit_prior(self):
        predictor = module("04_model_development", "registry").Predictor("dt", {}, 3).fit(self.x.iloc[:, :0], self.y)
        np.testing.assert_allclose(predictor.predict(self.x), self.y.mean())
        result = module("06_model_evaluation", "calibration").calibration(self.y, predictor.predict(self.x), 1e-6)
        self.assertIsNone(result["calibration_slope"])

    def test_stable_offset_calibration_and_ci(self):
        p = np.tile([0., .05, .3, .7], len(self.y)//4)
        result = module("06_model_evaluation", "calibration").calibration(self.y, p, 1e-6)
        self.assertLess(abs(result["offset_score_residual"]), 1e-8)
        self.assertLess(result["intercept_ci_lower"], result["calibration_intercept"])
        self.assertGreater(result["intercept_ci_upper"], result["calibration_intercept"])

    def test_threshold_counts_and_dca_boundaries(self):
        y = np.array([0,0,1,1]); p = np.array([.1,.4,.3,.9])
        table = module("06_model_evaluation", "thresholds").classification(y, p, [.3,.5])
        self.assertEqual(table.iloc[0].tp, 2); self.assertEqual(table.iloc[1].fn, 1)
        dca = module("06_model_evaluation", "dca").decision_curve
        frame = dca(y, p, [.2,.5])
        self.assertAlmostEqual(frame.model.iloc[1], .25)
        with self.assertRaises(ValueError): dca(y, p, [1.])

    def test_frozen_bootstrap_is_seeded(self):
        evaluate = module("06_model_evaluation", "metrics").frozen_bootstrap_ci
        p = np.linspace(.1,.9,len(self.y))
        self.assertEqual(evaluate(self.y,p,20,61), evaluate(self.y,p,20,61))

    def test_rubin_scalar_and_exact_shap_additivity(self):
        pooling = module("06_model_evaluation", "pooling").rubin_scalar
        pooled = pooling([.2,.3,.1], [.1,.12,.09], 80)
        self.assertAlmostEqual(pooled["estimate"], .2)
        predictor = module("04_model_development", "registry").Predictor("dt", {"max_depth":2}, 21).fit(self.x.iloc[:70,:2], self.y[:70])
        options = dict(self.cfg["interpretation"], method="exact", max_exact_features=3)
        values, base, error = module("07_model_interpretation", "shap_values").explain(predictor, self.x.iloc[:20,:2], self.x.iloc[70:73,:2], options)
        self.assertEqual(values.shape, (3,2)); self.assertLess(error, 1e-10)
        kernel_options = dict(options, method="kernel", kernel_samples=50)
        kernel, _, kernel_error = module("07_model_interpretation", "shap_values").explain(predictor, self.x.iloc[:20,:2], self.x.iloc[70:73,:2], kernel_options)
        np.testing.assert_allclose(values, kernel, atol=1e-8)
        self.assertLess(kernel_error, 1e-8)
        with tempfile.TemporaryDirectory() as temp:
            module("07_model_interpretation", "plots").plot_explanation(values, self.x.iloc[70:73,:2], base, Path(temp))
            self.assertEqual(len(list(Path(temp).glob("*.pdf"))), 4)

    def test_public_output_protection(self):
        with self.assertRaises(ValueError): output_directory(ROOT / "outputs")


@unittest.skipUnless(os.environ.get("HAO_TEST_R") == "1", "R integration explicitly enabled with HAO_TEST_R=1")
class RIntegrationTests(unittest.TestCase):
    def settings(self):
        cfg = load_config(ROOT / "config/example.json")
        cfg["imputation"].update(m=2, iterations=2)
        cfg["selection"]["mode"] = "none"
        return cfg

    def test_two_mice_datasets_and_training_exclusion(self):
        x, y = synthetic(); x.loc[::8,"measure_a"] = np.nan
        cfg = self.settings()
        with tempfile.TemporaryDirectory() as tmp:
            prepare = module("02_missing_data_imputation", "bridge").prepare
            batches, maps = prepare(x.iloc[:90], x.iloc[90:], y[:90], cfg, Path(tmp))
            self.assertEqual(len(batches), 2)
            for a,b,features in batches:
                self.assertFalse(a.isna().any().any()); self.assertFalse(b.isna().any().any())
                np.testing.assert_allclose(a.measure_a.drop(index=a.index[::8]), x.iloc[:90].measure_a.drop(index=x.iloc[:90].index[::8]))
            # Perturb assessment data; training posterior draws must remain identical.
            changed = x.iloc[90:].copy(); changed.measure_b += 25
            second, _ = prepare(x.iloc[:90], changed, y[:90], cfg, Path(tmp)/"other")
            for (a,_,_),(c,_,_) in zip(batches,second): np.testing.assert_allclose(a,c)

    def test_lasso_and_boruta_sources_run(self):
        x,y = synthetic(n=140); x.loc[::9,"measure_b"] = np.nan
        cfg = self.settings(); cfg["imputation"]["m"] = 1
        cfg["selection"]["mode"] = "intersection"
        cfg["selection"]["lasso"]["folds"] = 3
        cfg["selection"]["boruta"]["max_runs"] = 11
        with tempfile.TemporaryDirectory() as tmp:
            batches,_ = module("02_missing_data_imputation", "bridge").prepare(x.iloc[:100],x.iloc[100:],y[:100],cfg,Path(tmp))
            selected=json.loads((Path(tmp)/"selection.json").read_text())[0]
            self.assertEqual(set(batches[0][2]),set(selected["lasso"])&set(selected["boruta"]))

    def test_configured_passive_formula(self):
        x,y = synthetic(); x.measure_b = 2*x.measure_a
        x.loc[::7,"measure_b"] = np.nan
        cfg = self.settings(); cfg["imputation"]["m"] = 1
        cfg["imputation"]["passive"] = {"measure_b":"~I(2 * measure_a)"}
        with tempfile.TemporaryDirectory() as tmp:
            batches,_ = module("02_missing_data_imputation", "bridge").prepare(x.iloc[:90],x.iloc[90:],y[:90],cfg,Path(tmp))
            for a,b,_ in batches:
                np.testing.assert_allclose(a.measure_b, 2*a.measure_a)
                np.testing.assert_allclose(b.measure_b, 2*b.measure_a)

    def test_multiple_imputation_training_and_nested_cv(self):
        x,y = synthetic(n=120); x.loc[::11,"measure_a"] = np.nan
        cfg = self.settings(); cfg["models"] = {k:cfg["models"][k] for k in ["lr","dt"]}
        cfg["imputation"]["m"] = 5
        cfg["validation"].update(outer_folds=2, outer_repeats=1, inner_folds=2, imputations=1)
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp); rows=train_holdout(x,y,cfg,directory)
            self.assertEqual(len(rows),10)
            self.assertTrue((directory/"lr_rubin_pooling.json").is_file())
            nested=directory/"nested"; nested.mkdir()
            cfg["selection"]["mode"] = "intersection"
            cfg["selection"]["lasso"]["folds"] = 3
            cfg["selection"]["boruta"]["max_runs"] = 11
            records=module("05_model_validation","nested_cv").validate(x,y,cfg,nested)
            self.assertEqual(len(records),2)
            oof=pd.read_csv(nested/"oof_predictions.csv")
            self.assertTrue((oof.groupby("model").size()==84).all())


if __name__ == "__main__": unittest.main()
