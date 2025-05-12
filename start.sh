#!/bin/bash

source ./.venv/bin/activate
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)/src"

if [ -z "$1" ]; then
    echo "Usage: $0 [agent|llm|rag|cli]"
    exit 1
fi

if [ "$1" == "agent" ]; then
    python src/main.py --start-agent $2

elif [ "$1" == "llm" ]; then
    python src/main.py --start-llm $2

elif [ "$1" == "rag" ]; then
    python src/services/rag_tool_service.py $2
elif [ "$1" == "cli" ]; then
    python src/cli_client.py $2
else
    echo "Unknown command: $1"
    echo "Usage: $0 [agent|llm|rag|cli]"
    exit 1
fi