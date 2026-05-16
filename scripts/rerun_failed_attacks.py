"""
rerun_failed_attacks.py
=======================

Find every attack run that has attack_failed.json but no attack_metrics.json,
re-launch it with the exact same command line, and replace the failure with
real metrics if it succeeds this time.

Use case: the orchestrator launched attacks before some training cells had
finished writing their config.yaml. Now that all the prerequisites exist,
the failed attacks should succeed if re-run.

Run
---
    # Dry-run: list what would be re-attacked.
    python rerun_failed_attacks.py --dry-run

    # Actually re-run.
    python rerun_failed_attacks.py

    # Optional: limit to a specific directory subtree.
    python rerun_failed_attacks.py --filter noniid_alpha_01_dp
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def find_failed_attacks(results_root: Path, filter_substr: str | None) -> list[Path]:
    """Return attack directories that contain attack_failed.json but no attack_metrics.json."""
    failed = []
    for failed_json in results_root.rglob("attack_failed.json"):
        attack_dir = failed_json.parent
        if (attack_dir / "attack_metrics.json").exists():
            # Already succeeded later somehow; nothing to do.
            continue
        if filter_substr and filter_substr not in str(attack_dir):
            continue
        failed.append(attack_dir)
    return sorted(failed)


def load_command(attack_dir: Path) -> list[str] | None:
    """Read the original command from attack_failed.json."""
    try:
        with open(attack_dir / "attack_failed.json") as f:
            data = json.load(f)
        cmd = data.get("command")
        if not cmd or not isinstance(cmd, list):
            return None
        return cmd
    except Exception as e:
        print(f"  ! could not read attack_failed.json: {e}", file=sys.stderr)
        return None


def archive_failure(attack_dir: Path) -> None:
    """Move the failure artifacts aside so we can tell old from new."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = attack_dir / f"_previous_failure_{ts}"
    archive_dir.mkdir(exist_ok=True)
    for name in ("attack_failed.json", "attack_stderr.txt", "attack_stdout.txt"):
        src = attack_dir / name
        if src.exists():
            shutil.move(str(src), str(archive_dir / name))


def rerun_one(attack_dir: Path, dry_run: bool) -> str:
    """Re-run a single attack. Return one of: 'dry', 'success', 'still_failed', 'skip'."""
    cmd = load_command(attack_dir)
    if cmd is None:
        return "skip"

    # Sanity check: the experiment_dir from the original command must now have config.yaml.
    # Find --experiment-dir flag.
    try:
        idx = cmd.index("--experiment-dir")
        exp_dir = Path(cmd[idx + 1])
    except (ValueError, IndexError):
        print(f"  ! could not parse --experiment-dir from cmd", file=sys.stderr)
        return "skip"

    if not (exp_dir / "config.yaml").exists():
        print(f"  ! config.yaml still missing in {exp_dir}; skipping", file=sys.stderr)
        return "skip"
    if not (exp_dir / "final_model.pt").exists():
        print(f"  ! final_model.pt missing in {exp_dir}; skipping", file=sys.stderr)
        return "skip"

    if dry_run:
        print(f"  [dry-run] would re-run: {' '.join(cmd)}")
        return "dry"

    archive_failure(attack_dir)

    print(f"  -> running...")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path.cwd()),
            capture_output=True,
            text=True,
            timeout=60 * 60,  # 1 hour per attack; should be far faster
        )
    except subprocess.TimeoutExpired:
        print(f"  ! timeout after 1 hour", file=sys.stderr)
        return "still_failed"
    except Exception as e:
        print(f"  ! subprocess error: {e}", file=sys.stderr)
        return "still_failed"

    # Persist stdout/stderr for inspection.
    (attack_dir / "attack_stdout.txt").write_text(result.stdout)
    (attack_dir / "attack_stderr.txt").write_text(result.stderr)

    if result.returncode == 0 and (attack_dir / "attack_metrics.json").exists():
        print(f"  ✓ success")
        return "success"

    # Record a fresh failure if it crashed again.
    failure = {
        "timestamp": datetime.now().isoformat(),
        "command": cmd,
        "returncode": result.returncode,
        "note": "rerun also failed; see attack_stderr.txt",
    }
    with open(attack_dir / "attack_failed.json", "w") as f:
        json.dump(failure, f, indent=2)
    print(f"  ✗ still failing (returncode={result.returncode})")
    return "still_failed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run attack subprocesses that failed because their config.yaml didn't exist yet.",
    )
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="If set, only re-run attack dirs whose path contains this substring.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)

    failed = find_failed_attacks(results_root, args.filter)

    if not failed:
        print("No failed attacks found that need rerunning.")
        return

    print(f"Found {len(failed)} failed attack run(s):")
    for d in failed:
        print(f"  - {d.relative_to(results_root.parent if results_root.parent.exists() else Path.cwd())}")
    print()

    if args.dry_run:
        print("Dry-run mode; not actually rerunning.")
        for d in failed:
            rerun_one(d, dry_run=True)
        return

    counts = {"success": 0, "still_failed": 0, "skip": 0}
    for i, d in enumerate(failed, start=1):
        print(f"[{i}/{len(failed)}] {d}")
        outcome = rerun_one(d, dry_run=False)
        counts[outcome] = counts.get(outcome, 0) + 1

    print()
    print("Summary:")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()