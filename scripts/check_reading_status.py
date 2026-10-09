#!/usr/bin/env python3
"""Check local note/index status drift; this does not assess paper correctness or mastery.

Only reads docs. Existing check_frontmatter.py/check_links.py are independent.
Run: python scripts/check_reading_status.py [--root PATH] [--json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import yaml


def note_status(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^---\s*$", text, maxsplit=2, flags=re.MULTILINE)
    if len(parts) != 3 or parts[0].strip():
        raise ValueError("missing or malformed frontmatter")
    metadata = yaml.safe_load(parts[1])
    if not isinstance(metadata, dict) or not isinstance(metadata.get("status"), str) or metadata["status"] not in {"to-read", "reading", "done"}:
        raise ValueError("invalid note status")
    return metadata, parts[2]


def visible_status(cell: str) -> str | None:
    candidates = set()
    if "笔记完成" in cell or "已完成" in cell:
        candidates.add("done")
    if "精读中" in cell:
        candidates.add("reading")
    if any(word in cell for word in ("待读", "已派工单", "待精读")):
        candidates.add("to-read")
    return next(iter(candidates)) if len(candidates) == 1 else None


def cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def inspect(root: Path) -> dict:
    docs = root / "docs"
    errors, inventory = [], []
    notes = {}
    for path in sorted(docs.glob("topics/*/*/README.md")):
        relative = path.relative_to(docs).as_posix()
        try:
            metadata, body = note_status(path)
        except (ValueError, yaml.YAMLError) as exc:
            errors.append(f"{relative}: {exc}")
            continue
        status = metadata["status"]
        if status == "done":
            sections = re.findall(r"^##\s+([1-6])\.\s+", body, flags=re.MULTILINE)
            if sections != ["1", "2", "3", "4", "5", "6"]:
                errors.append(f"{relative}: done note must contain the ordered six sections")
            if re.search(r"<待精读[^>]*>|本页为占位骨架", body):
                errors.append(f"{relative}: done note still contains skeleton placeholders")
        notes[relative] = metadata
        inventory.append({"path": relative, "status": status, "read_date": str(metadata.get("read_date") or ""),
                          "body_lines": len(body.splitlines()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if not notes:
        errors.append("no paper notes found")
    index = (docs / "index.md").read_text(encoding="utf-8")
    listed = {}
    for number, line in enumerate(index.splitlines(), 1):
        links = re.findall(r"\]\((topics/[^/]+/[^/]+/README\.md)\)", line)
        if not links or not line.startswith("|"):
            continue
        row = cells(line)
        if len(links) != 1 or len(row) != 6:
            errors.append(f"index.md:{number}: malformed paper table row")
            continue
        path = links[0]
        if path in listed:
            errors.append(f"index.md:{number}: duplicate paper link {path}")
        listed[path] = visible_status(row[4])
        if path not in notes or listed[path] != notes[path]["status"]:
            errors.append(f"index.md:{number}: paper status does not match {path}")
    for path in notes:
        if path not in listed:
            errors.append(f"index.md: missing paper {path}")
    topic_counts = {}
    for number, line in enumerate(index.splitlines(), 1):
        links = re.findall(r"\]\(topics/([^/]+)/README\.md\)", line)
        if not links or not line.startswith("|"):
            continue
        row = cells(line)
        if len(links) != 1 or len(row) != 4 or not row[2].isdigit() or not row[3].isdigit():
            errors.append(f"index.md:{number}: malformed topic row; expected completed and total columns")
            continue
        topic = links[0]
        if topic in topic_counts:
            errors.append(f"index.md:{number}: duplicate topic row {topic}")
        expected = [m for path, m in notes.items() if path.split("/")[1] == topic]
        count = (sum(m["status"] == "done" for m in expected), len(expected))
        topic_counts[topic] = count
        if (int(row[2]), int(row[3])) != count:
            errors.append(f"index.md:{number}: {topic} counts should be {count[0]}/{count[1]}")
    for overview in sorted(docs.glob("topics/*/README.md")):
        topic = overview.parent.name
        if topic not in topic_counts:
            errors.append(f"index.md: missing valid topic row {topic}")
        for path, metadata in notes.items():
            if path.split("/")[1] != topic:
                continue
            target = "/".join(path.split("/")[2:])
            rows = [line for line in overview.read_text(encoding="utf-8").splitlines() if line.startswith("|") and f"]({target})" in line]
            if len(rows) != 1 or len(cells(rows[0])) != 5:
                errors.append(f"{overview.relative_to(docs)}: missing or malformed row for {target}")
            elif visible_status(cells(rows[0])[3]) != metadata["status"]:
                errors.append(f"{overview.relative_to(docs)}: status differs from {target}")
    queue = (docs / "reading-list.md").read_text(encoding="utf-8")
    for path, metadata in notes.items():
        linked = [line for line in queue.splitlines() if f"]({path})" in line]
        if metadata["status"] == "done":
            if any("骨架" in line or "待接单" in line for line in linked):
                errors.append(f"reading-list.md: completed note still marked as pending: {path}")
            dated = [line for line in linked if line.startswith("|") and str(metadata.get("read_date")) in line]
            if not dated:
                errors.append(f"reading-list.md: missing completion date for {path}")
    return {"status": "failed" if errors else "pass", "scope": "mechanical local status consistency, not semantic review or reproduction",
            "paper_count": len(notes), "completed_count": sum(n["status"] == "done" for n in notes.values()),
            "inventory": inventory, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = inspect(args.root)
    except (OSError, UnicodeError) as exc:
        result = {"status": "failed", "errors": [str(exc)]}
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"阅读状态检查：{result['status']}")
        for error in result["errors"]:
            print(error)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
