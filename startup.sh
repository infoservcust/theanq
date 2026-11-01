#!/bin/sh
cd "$(dirname "$0")"
if curl -sf -o /dev/null --max-time 1 http://127.0.0.1:8080/; then
  exit 0
fi
npm run dev > /tmp/dev.log 2>&1 &
