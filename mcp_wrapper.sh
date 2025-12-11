#!/bin/bash
# Debug wrapper for MCP server - logs to file without touching stdout
LOG=/tmp/mcp_debug.log
echo "=== Starting at $(date) ===" >> "$LOG"
echo "PWD: $PWD" >> "$LOG"
echo "Args: $*" >> "$LOG"
echo "PATH: $PATH" >> "$LOG"

# Run python with stderr going to log file, stdout stays clean for JSON-RPC
/home/david/dev/galaxybrain/.venv/bin/python "$@" 2>> "$LOG"
EXIT_CODE=$?

echo "Exit code: $EXIT_CODE" >> "$LOG"
exit $EXIT_CODE
