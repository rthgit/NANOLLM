import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Nano artifact tests on 3B/7B/14B artifacts.")
    parser.add_argument("--baseline-mode", choices=["auto", "none", "8bit", "4bit"], default="auto")
    parser.add_argument("--min-cosine", type=float, default=0.99)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--report", default="nano_matrix_test_report.json")
    parser.add_argument("--artifacts-root", default="/kaggle/working")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.artifacts_root)
    test_script = Path(__file__).with_name("nano_artifact_test.py").resolve()
    targets: List[Path] = [
        root / "final_artifact_3B",
        root / "final_artifact_7B",
        root / "final_artifact_Qwen2.5-14B-Instruct",
    ]

    matrix: Dict[str, Dict[str, object]] = {}
    for artifact_dir in targets:
        key = artifact_dir.name
        if not artifact_dir.exists():
            matrix[key] = {"found": False, "pass": False, "reason": "artifact directory missing"}
            continue

        out = artifact_dir / "test_report.json"
        cmd = [
            sys.executable,
            str(test_script),
            "--artifact-dir",
            str(artifact_dir),
            "--baseline-mode",
            args.baseline_mode,
            "--min-cosine",
            str(args.min_cosine),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--max-length",
            str(args.max_length),
            "--output",
            str(out),
        ]
        print(f"\nRunning test: {artifact_dir}")
        result = subprocess.run(cmd, text=True, capture_output=True)
        if result.returncode != 0:
            matrix[key] = {
                "found": True,
                "pass": False,
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-1000:],
                "stderr_tail": result.stderr[-1000:],
            }
            continue

        report = json.loads(out.read_text(encoding="utf-8"))
        matrix[key] = {
            "found": True,
            "pass": bool(report.get("pass", False)),
            "checks": report.get("checks", {}),
            "cosine_summary": report.get("cosine_summary", {}),
            "report_path": str(out),
        }

    matrix_path = Path(args.report).resolve()
    matrix_path.write_text(json.dumps(matrix, indent=2, ensure_ascii=False), encoding="utf-8")
    global_pass = all(v.get("pass", False) for v in matrix.values() if v.get("found"))
    print(f"\nMATRIX_PASS={global_pass}")
    print(f"Matrix report: {matrix_path}")


if __name__ == "__main__":
    main()
