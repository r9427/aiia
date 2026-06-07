#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
echo "script directory: ${SCRIPT_DIR}"

"${SCRIPT_DIR}"/app