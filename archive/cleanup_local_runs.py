from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


KEEP_TOP_LEVEL_DIRS = {
    "Llama-3.2-3B-NANO-Canonical-v3-Safe88-Embed8",
    "Qwen2.5-7B-NANO-Canonical-v3-Safe88-Embed8",
    "l3x",
    "research_logs",
}

LOG_NAMES = {
    "search_results.json",
    "scan_results.json",
    "results.json",
    "combo_results.json",
    "sonar_results.json",
}


@dataclass
class CleanupPlan:
    kept: list[Path] = field(default_factory=list)
    archived: list[tuple[Path, Path]] = field(default_factory=list)
    deleted: list[Path] = field(default_factory=list)
    deleted_bytes: int = 0


def dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def load_best_l3x_paths(l3x_dir: Path) -> tuple[Path | None, Path | None]:
    combo_path = l3x_dir / "combo_results.json"
    if not combo_path.exists():
        return None, None
    data = json.loads(combo_path.read_text(encoding="utf-8"))
    best = data.get("best_pass") or {}
    artifact_dir = best.get("artifact_dir")
    map_path = best.get("map_path")
    return (
        Path(artifact_dir) if artifact_dir else None,
        Path(map_path) if map_path else None,
    )


def should_archive_json(path: Path) -> bool:
    if path.name in LOG_NAMES:
        return True
    return "maps" in path.parts and path.suffix == ".json"


def archive_files(
    source_root: Path,
    archive_root: Path,
    files: Iterable[Path],
    apply: bool,
    plan: CleanupPlan,
) -> None:
    for source in files:
        rel = source.relative_to(source_root)
        target = archive_root / rel
        plan.archived.append((source, target))
        if apply:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def remove_path(path: Path, apply: bool, plan: CleanupPlan) -> None:
    if not path.exists():
        return
    plan.deleted.append(path)
    plan.deleted_bytes += dir_size(path)
    if not apply:
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def prune_l3x(l3x_dir: Path, archive_root: Path, apply: bool, plan: CleanupPlan) -> None:
    best_artifact_dir, best_map_path = load_best_l3x_paths(l3x_dir)
    if not l3x_dir.exists():
        return

    keep_paths = {l3x_dir / "combo_results.json"}
    if best_artifact_dir:
        keep_paths.add(best_artifact_dir)
    if best_map_path:
        keep_paths.add(best_map_path)

    archive_candidates = [
        path
        for path in l3x_dir.rglob("*.json")
        if should_archive_json(path) and path not in keep_paths
    ]
    archive_files(l3x_dir, archive_root / "l3x", archive_candidates, apply, plan)

    for child in l3x_dir.iterdir():
        if child.name == "artifacts":
            for artifact_dir in child.iterdir():
                if artifact_dir != best_artifact_dir:
                    remove_path(artifact_dir, apply, plan)
            continue
        if child.name == "maps":
            for map_file in child.iterdir():
                if map_file != best_map_path:
                    remove_path(map_file, apply, plan)
            continue
        if child not in keep_paths:
            remove_path(child, apply, plan)

    plan.kept.extend(sorted(p for p in keep_paths if p.exists() or not apply))


def build_plan(local_runs: Path, archive_dir_name: str, apply: bool) -> CleanupPlan:
    plan = CleanupPlan()
    archive_root = local_runs / archive_dir_name
    l3x_dir = local_runs / "l3x"

    prune_l3x(l3x_dir, archive_root, apply, plan)

    for child in sorted(local_runs.iterdir()):
        if child.name in KEEP_TOP_LEVEL_DIRS:
            if child.name != "l3x":
                plan.kept.append(child)
            continue
        if child.is_file():
            plan.kept.append(child)
            continue

        archive_candidates = [
            path for path in child.rglob("*.json") if should_archive_json(path)
        ]
        archive_files(local_runs, archive_root, archive_candidates, apply, plan)
        remove_path(child, apply, plan)

    return plan


def print_plan(plan: CleanupPlan, apply: bool) -> None:
    mode = "APPLY" if apply else "DRY-RUN"
    print(f"[{mode}] keep={len(plan.kept)} archive={len(plan.archived)} delete={len(plan.deleted)}")
    if plan.kept:
        print("\nKept:")
        for path in plan.kept:
            print(f"  {path}")
    if plan.archived:
        print("\nArchived:")
        for source, target in plan.archived:
            print(f"  {source} -> {target}")
    if plan.deleted:
        print("\nDeleted:")
        for path in plan.deleted:
            print(f"  {path}")
    print(f"\nSpace to free: {plan.deleted_bytes:,} bytes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean heavy candidate artifacts under local_runs.")
    parser.add_argument(
        "--local-runs",
        type=Path,
        default=Path("local_runs"),
        help="Path to the local_runs directory.",
    )
    parser.add_argument(
        "--archive-dir-name",
        default="research_logs",
        help="Subdirectory under local_runs used to keep small result logs.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually perform the cleanup. Default is dry-run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_runs = args.local_runs.resolve()
    if not local_runs.exists():
        raise FileNotFoundError(local_runs)
    plan = build_plan(local_runs, args.archive_dir_name, args.apply)
    print_plan(plan, args.apply)


if __name__ == "__main__":
    main()
