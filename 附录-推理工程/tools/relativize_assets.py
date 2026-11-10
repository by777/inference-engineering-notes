#!/usr/bin/env python3
"""把镜像 HTML 里的绝对资源路径 /_next/... 改成相对路径。

背景：wget -k 只转换了它抓到的链接，页面内嵌的若干 "/_next/..." 仍是绝对路径。
绝对路径在本地 http 服务下没问题，但在 file:// 下会指向文件系统根目录而 404。
按页面深度补 "../" 前缀后，http 与 file 两种打开方式都能解析到同一个文件。

用法（在 learn-inference/ 目录下）：
    python3 tools/relativize_assets.py
"""
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "site", "learn-inference.com")


def depth_prefix(dirpath: str) -> str:
    rel = os.path.relpath(dirpath, ROOT)
    if rel == ".":
        return ""
    return "../" * len(rel.split(os.sep))


def main() -> int:
    if not os.path.isdir(ROOT):
        print(f"找不到镜像目录: {ROOT}", file=sys.stderr)
        return 1

    changed = 0
    total = 0
    for dirpath, _, files in os.walk(ROOT):
        prefix = depth_prefix(dirpath)
        for name in files:
            if not name.endswith(".html"):
                continue
            total += 1
            path = os.path.join(dirpath, name)
            with open(path, encoding="utf-8") as fh:
                text = fh.read()

            # 只重写属性/JSON 字符串里的引用，避免误伤已经正确的相对路径
            new = text.replace('"/_next/', f'"{prefix}_next/')
            new = new.replace("'/_next/", f"'{prefix}_next/")

            if new != text:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(new)
                changed += 1

    print(f"扫描 {total} 个 HTML，重写 {changed} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
