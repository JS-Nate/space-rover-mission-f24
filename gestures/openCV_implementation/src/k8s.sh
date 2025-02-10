xhost +local:root
docker run --rm \
    --network kind \
    --device /dev/video0 \
    --device /dev/input \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --name gesture_container -it gesture


# docker run --rm --network kind --device /dev/video0 --name gesture_container -it gesture
# docker run --rm --network kind --device /dev/video0 --name gesture_container gesture 
