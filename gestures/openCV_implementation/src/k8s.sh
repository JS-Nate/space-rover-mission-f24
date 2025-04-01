#!/bin/bash

xhost +local:docker

docker run --rm \
    --gpus all \
    --runtime=nvidia \
    --network kind \
    --device /dev/video0 \
    --device /dev/video2 \
    --device /dev/input \
    --env DISPLAY=$DISPLAY \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --name gesture_container \
    -p 5000:5000 \
    -it gesture
