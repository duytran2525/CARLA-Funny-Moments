from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _audit_town(town_dir: Path) -> bool:
    csv_path = town_dir / "driving_log.csv"
    if not csv_path.exists():
        print(f"{town_dir.name}: missing driving_log.csv")
        return False

    rows = _read_rows(csv_path)
    center_files = {path.stem for path in (town_dir / "images_center").glob("*.jpg")}
    left_files = {path.stem for path in (town_dir / "images_left").glob("*.jpg")}
    right_files = {path.stem for path in (town_dir / "images_right").glob("*.jpg")}

    command_counts = Counter(row.get("command", "") for row in rows)
    route_command_counts = Counter(row.get("route_command", "") for row in rows if "route_command" in row)
    command_source_counts = Counter(row.get("command_source", "") for row in rows if "command_source" in row)
    junction_ahead_counts = Counter(row.get("junction_ahead", "") for row in rows if "junction_ahead" in row)
    img_ids = {row.get("img_id", "") for row in rows}
    history_ids = {row.get("img_id_tm06", "") for row in rows} | {row.get("img_id_tm03", "") for row in rows}
    all_referenced_ids = img_ids | history_ids

    missing_center = sorted(all_referenced_ids - center_files)
    missing_left = sorted(all_referenced_ids - left_files)
    missing_right = sorted(all_referenced_ids - right_files)
    frame_id_filename_matches = len({row.get("frame_id", "") for row in rows} & center_files)

    print(f"{town_dir.name}: rows={len(rows)} images={len(center_files)}/{len(left_files)}/{len(right_files)}")
    print(f"{town_dir.name}: command_counts={dict(sorted(command_counts.items()))}")
    if route_command_counts:
        print(f"{town_dir.name}: route_command_counts={dict(sorted(route_command_counts.items()))}")
    if command_source_counts:
        print(f"{town_dir.name}: command_source_counts={dict(sorted(command_source_counts.items()))}")
    if junction_ahead_counts:
        print(f"{town_dir.name}: junction_ahead_counts={dict(sorted(junction_ahead_counts.items()))}")
    print(
        f"{town_dir.name}: referenced_img_missing center={len(missing_center)} "
        f"left={len(missing_left)} right={len(missing_right)}"
    )
    print(
        f"{town_dir.name}: frame_id_filename_matches={frame_id_filename_matches} "
        "(expected 0; use img_id/image_filename to load images)"
    )

    suspicious_turns = 0
    turn_without_junction = 0
    for row in rows:
        if row.get("command") not in {"1", "2"}:
            continue
        try:
            if abs(float(row.get("wp_5_y", "0"))) < 1.0:
                suspicious_turns += 1
        except ValueError:
            suspicious_turns += 1
        if row.get("junction_ahead") == "0":
            turn_without_junction += 1
    if suspicious_turns:
        print(f"{town_dir.name}: suspicious_turn_commands={suspicious_turns} (turn command with |wp_5_y| < 1m)")
    if turn_without_junction:
        print(f"{town_dir.name}: turn_without_junction_ahead={turn_without_junction}")

    return not (missing_center or missing_left or missing_right)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit collected CARLA dataset CSV/image linkage.")
    parser.add_argument("root", nargs="?", default="data/collected", help="Dataset root containing TownXX folders.")
    parser.add_argument("--town", action="append", default=None, help="Town folder name to audit. Can be repeated.")
    args = parser.parse_args()

    root = Path(args.root)
    towns = [root / town for town in args.town] if args.town else sorted(path for path in root.iterdir() if path.is_dir())
    ok = True
    for town_dir in towns:
        ok = _audit_town(town_dir) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
