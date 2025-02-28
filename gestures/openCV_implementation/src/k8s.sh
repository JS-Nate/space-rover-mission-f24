#!/bin/bash

# Allow Docker containers to access the host display
xhost +local:docker

# Run the container with X11 forwarding
docker run --rm \
    --network kind \
    --device /dev/video0 \
    --device /dev/input \
    --env DISPLAY=$DISPLAY \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --name gesture_container -it gesture
