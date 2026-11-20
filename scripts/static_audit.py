# SPDX-License-Identifier: MIT
"""Read source only. Fail on data/artifacts, personal paths or credential literals."""
from pathlib import Path
import ast, json, re

ROOT = Path(__file__).resolve().parents[1]


def audit():
    forbidden = {".csv", ".xlsx", ".xls", ".rds", ".rdata", ".pkl", ".npy", ".npz", ".pdf", ".png", ".tiff", ".tif", ".log", ".ipynb"}
    problems = []; count = 0
    for p in ROOT.rglob("*"):
        if any(x in p.relative_to(ROOT).parts for x in [".git", "__pycache__", ".venv"]): continue
        if p.is_symlink(): problems.append("Source symlink: " + str(p.relative_to(ROOT))); continue
        if not p.is_file(): continue
        count += 1
        if p.suffix.lower() in forbidden: problems.append("Non-source artifact: " + str(p.relative_to(ROOT)))
        try: text = p.read_text(encoding="utf-8")
        except UnicodeError: problems.append("Non-UTF8/binary file"); continue
        if "\x00" in text: problems.append("Binary data")
        patterns = [r"/(?:Users|home)/[^/\s]+", r"\bghp_[A-Za-z0-9]{15,}",
            r"\bgithub_pat_[A-Za-z0-9_]{15,}", r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"]
        if any(re.search(pattern, text) for pattern in patterns): problems.append("Path/credential candidate: " + str(p.relative_to(ROOT)))
        if p.suffix == ".py": ast.parse(text, filename=str(p.relative_to(ROOT)))
        if p.suffix == ".json": json.loads(text)
        if p.name.endswith((".py.md", ".R.md")): problems.append("Executable source disguised as Markdown")
    expected = ["01_data_preprocessing", "02_missing_data_imputation", "03_feature_selection",
        "04_model_development", "05_model_validation", "06_model_evaluation", "07_model_interpretation"]
    for name in expected:
        if not any(p.suffix in {".py", ".R"} for p in (ROOT/name).iterdir()): problems.append("Empty code module: " + name)
    meta = json.loads((ROOT/".zenodo.json").read_text())
    if meta["license"] != "mit" or "license: MIT" not in (ROOT/"CITATION.cff").read_text(): problems.append("License metadata mismatch")
    if not (ROOT/"LICENSE").read_text().startswith("MIT License"): problems.append("Wrong license text")
    return {"files_checked": count, "problems": problems}


if __name__ == "__main__":
    result = audit(); print(json.dumps(result, indent=2))
    raise SystemExit(bool(result["problems"]))
