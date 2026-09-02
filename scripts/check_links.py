#!/usr/bin/env python3
"""跑 `mkdocs build --clean` 并把内部链接告警当作失败。

背景：本仓库的坏链接目前以 INFO 级别输出，`--strict` 只会把 WARNING 升级为错误，
因此单靠 `--strict` 拦不住。本脚本改为解析构建输出，命中链接类问题即以非零码退出。
`mkdocs.yml` 归主线所有，本脚本不修改它。

在仓库根目录直接运行：python scripts/check_links.py
"""

from __future__ import annotations

import subprocess
import sys

# Windows 控制台默认 cp1252，中文输出会抛 UnicodeEncodeError，强制走 UTF-8
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

# MkDocs 报告内部链接问题时使用的措辞
LINK_MARKERS = (
    "contains a link",
    "is not found among documentation files",
    "does not contain an anchor",
    "contains an unrecognized relative link",
    "but the target is not found",
)


def main() -> int:
    proc = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--clean", "--strict"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    output = (proc.stdout or "") + (proc.stderr or "")
    print(output, end="" if output.endswith("\n") else "\n")

    if proc.returncode != 0:
        print(f"\nmkdocs build 失败（退出码 {proc.returncode}）。", file=sys.stderr)
        return proc.returncode

    offenders = [
        line
        for line in output.splitlines()
        if any(marker in line for marker in LINK_MARKERS)
    ]

    if offenders:
        print("\n检测到内部链接问题：", file=sys.stderr)
        for line in offenders:
            print(f"  {line.strip()}", file=sys.stderr)
        print(f"\n共 {len(offenders)} 条链接告警，视为构建失败。", file=sys.stderr)
        return 1

    print("\n链接检查通过：mkdocs build 无内部链接告警。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
