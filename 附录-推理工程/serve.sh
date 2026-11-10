#!/usr/bin/env bash
# 在本地起一个静态服务阅读离线镜像。
#
# 为什么需要它：站点的 JS 运行时会用 "/_next/..." 这类根路径拼资源地址，
# file:// 下根路径会解析到文件系统根目录而 404。用 http 服务最稳。
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-8123}"

echo "服务目录: $DIR/site"
echo "阅读入口: http://127.0.0.1:${PORT}/learn-inference.com/index.html"
echo "按 Ctrl+C 停止"
echo

cd "$DIR/site"
exec python3 -m http.server "$PORT" --bind 127.0.0.1
