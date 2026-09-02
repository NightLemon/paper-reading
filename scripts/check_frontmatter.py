#!/usr/bin/env python3
"""校验笔记的 YAML frontmatter 是否合规。

扫描范围：
- 论文笔记 docs/topics/<主题>/<年份>-<短名>/README.md
- 概念页   docs/concepts/*.md（排除 README.md）

在仓库根目录直接运行：python scripts/check_frontmatter.py
全部通过时以 0 退出，否则逐条打印 `<相对路径>: <问题描述>` 并以 1 退出。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

# Windows 控制台默认 cp1252，中文输出会抛 UnicodeEncodeError，强制走 UTF-8
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docs"

TOPICS = {
    "foundations",
    "llm-serving",
    "llm-training",
    "distributed-systems",
    "networking",
    "llm-applications",
}
STATUSES = {"to-read", "reading", "done"}

KEBAB = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
PAPER_DIR = re.compile(r"^(\d{4})-([a-z0-9]+(?:-[a-z0-9]+)*)$")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def load_frontmatter(path: Path, problems: list[str]) -> dict | None:
    """读出 frontmatter；缺失或格式错误时记录问题并返回 None。"""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        problems.append(f"{rel(path)}: 缺少 YAML frontmatter（文件未以 `---` 开头）")
        return None

    # 跳过首行的 ---，找结束分隔符
    parts = re.split(r"^---\s*$", text, maxsplit=2, flags=re.MULTILINE)
    if len(parts) < 3:
        problems.append(f"{rel(path)}: frontmatter 没有闭合的 `---` 分隔符")
        return None

    try:
        data = yaml.safe_load(parts[1])
    except yaml.YAMLError as exc:
        problems.append(f"{rel(path)}: frontmatter YAML 解析失败：{exc}")
        return None

    if data is None:
        problems.append(f"{rel(path)}: frontmatter 为空")
        return None
    if not isinstance(data, dict):
        problems.append(f"{rel(path)}: frontmatter 顶层必须是键值映射，实际为 {type(data).__name__}")
        return None
    return data


def is_blank(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def check_paper(path: Path, problems: list[str]) -> None:
    data = load_frontmatter(path, problems)
    if data is None:
        return

    for field in ("title", "authors", "venue", "year", "topic", "status"):
        if field not in data or is_blank(data[field]):
            problems.append(f"{rel(path)}: 缺少必填字段 `{field}`")

    if "authors" in data and not isinstance(data["authors"], list):
        problems.append(f"{rel(path)}: `authors` 必须是列表")

    year = data.get("year")
    if year is not None and not isinstance(year, int):
        problems.append(f"{rel(path)}: `year` 必须是整数，实际为 {year!r}")

    topic = data.get("topic")
    if topic is not None and topic not in TOPICS:
        problems.append(
            f"{rel(path)}: `topic` 取值 {topic!r} 不在允许集合 {sorted(TOPICS)} 内"
        )

    status = data.get("status")
    if status is not None and status not in STATUSES:
        problems.append(
            f"{rel(path)}: `status` 取值 {status!r} 不在允许集合 {sorted(STATUSES)} 内"
        )

    paper_dir = path.parent
    dir_topic = paper_dir.parent.name
    if topic is not None and topic != dir_topic:
        problems.append(
            f"{rel(path)}: `topic` 字段 {topic!r} 与所在目录名 {dir_topic!r} 不一致"
        )

    match = PAPER_DIR.match(paper_dir.name)
    if not match:
        problems.append(
            f"{rel(path)}: 目录名 {paper_dir.name!r} 不符合 `<4位年份>-<kebab-case>` 约定"
        )
    elif isinstance(year, int) and int(match.group(1)) != year:
        problems.append(
            f"{rel(path)}: 目录名年份 {match.group(1)} 与 `year` 字段 {year} 不一致"
        )

    if status == "done":
        rating = data.get("rating")
        if is_blank(rating):
            problems.append(f"{rel(path)}: `status: done` 时 `rating` 必须存在且非空")
        elif not isinstance(rating, int) or not 1 <= rating <= 5:
            problems.append(f"{rel(path)}: `rating` 必须是 1–5 的整数，实际为 {rating!r}")

        read_date = data.get("read_date")
        if is_blank(read_date):
            problems.append(f"{rel(path)}: `status: done` 时 `read_date` 必须存在且非空")
        else:
            # YAML 会把裸日期解析成 datetime.date，转成字符串再校验格式
            if not ISO_DATE.match(str(read_date)):
                problems.append(
                    f"{rel(path)}: `read_date` 必须是 `YYYY-MM-DD` 格式，实际为 {read_date!r}"
                )


def check_concept(path: Path, problems: list[str]) -> None:
    if not KEBAB.match(path.stem):
        problems.append(f"{rel(path)}: 文件名 {path.stem!r} 不是 kebab-case")

    data = load_frontmatter(path, problems)
    if data is None:
        return

    for field in ("concept", "tags"):
        if field not in data or is_blank(data[field]):
            problems.append(f"{rel(path)}: 缺少必填字段 `{field}`")

    if "tags" in data and not isinstance(data["tags"], list):
        problems.append(f"{rel(path)}: `tags` 必须是列表")


def main() -> int:
    problems: list[str] = []

    papers = sorted(DOCS.glob("topics/*/*/README.md"))
    concepts = sorted(p for p in DOCS.glob("concepts/*.md") if p.name != "README.md")

    for path in papers:
        check_paper(path, problems)
    for path in concepts:
        check_concept(path, problems)

    checked = len(papers) + len(concepts)
    if not checked:
        print("未找到任何待检查的文件，请确认在仓库根目录运行本脚本。", file=sys.stderr)
        return 1

    if problems:
        for problem in problems:
            print(problem)
        print(
            f"\n检查了 {checked} 个文件（论文笔记 {len(papers)} 个，概念页 {len(concepts)} 个），"
            f"发现 {len(problems)} 个问题。",
            file=sys.stderr,
        )
        return 1

    print(
        f"frontmatter 检查通过：共 {checked} 个文件"
        f"（论文笔记 {len(papers)} 个，概念页 {len(concepts)} 个）。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
