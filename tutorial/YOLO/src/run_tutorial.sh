#!/bin/bash

# Run the container for tutorial
docker run \
  --rm \
  --network kind \
  --device /dev/video0 \
  --name tutorial_container \
  tutorial
