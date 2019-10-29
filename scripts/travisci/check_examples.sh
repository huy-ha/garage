#!/usr/bin/env bash
find examples/ -name '*.py' -exec sh -c 'echo "$1"; python "$1"' _ {} \;
