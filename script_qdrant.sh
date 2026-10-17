#!/bin/bash

CONTAINER_NAME="docuquery-qdrant"

case "$1" in
  start)
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
      docker start "$CONTAINER_NAME"
    else
      docker run -d \
        --name "$CONTAINER_NAME" \
        -p 6333:6333 \
        -p 6334:6334 \
        -v "$(cd "$(dirname "$0")" && pwd)/qdrant_storage:/qdrant/storage" \
        qdrant/qdrant
    fi
    ;;

  stop)
    docker stop "$CONTAINER_NAME"
    ;;

  status)
    docker ps -a --filter "name=$CONTAINER_NAME"
    ;;

  *)
    echo "Usage: ./script_qdrant.sh {start|stop|status}"
    exit 1
    ;;
esac
