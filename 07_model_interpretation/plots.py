# SPDX-License-Identifier: MIT
"""Generic saved-explanation plots; never fit a prediction model here."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_explanation(values, cases, base, directory):
    import shap
    explanation = shap.Explanation(values=values, data=cases.to_numpy(),
        base_values=np.full(len(cases), base), feature_names=list(cases.columns))
    for name, function in [("importance", shap.plots.bar), ("beeswarm", shap.plots.beeswarm)]:
        function(explanation, show=False)
        plt.savefig(directory / (name+".pdf"), bbox_inches="tight"); plt.close()
    shap.plots.waterfall(explanation[0], show=False)
    plt.savefig(directory / "waterfall.pdf", bbox_inches="tight"); plt.close()
    count = len(cases.columns)
    if count:
        fig, axes = plt.subplots(int(np.ceil(count/3)), min(count,3), squeeze=False,
            figsize=(min(count,3)*3.2, int(np.ceil(count/3))*2.8))
        for j, ax in enumerate(axes.flat):
            if j >= count: ax.axis("off"); continue
            ax.scatter(cases.iloc[:,j], values[:,j], s=12)
            ax.axhline(0, linestyle="--", color="gray", linewidth=.7)
            ax.set(xlabel=cases.columns[j], ylabel="SHAP value")
        fig.tight_layout(); fig.savefig(directory / "dependence.pdf"); plt.close(fig)
