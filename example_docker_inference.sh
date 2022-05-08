#!/usr/bin/env bash

# You need to have build our docker image begore running this script
# You also need to change the paths below
SUB_NAME="sub-test"
TEST_INPUT_IMG="/home/lucasf/data/test_FeTA_inference/${SUB_NAME}"  # contains /anat
TEST_INPUT_META="/home/lucasf/data/test_FeTA_inference/${SUB_NAME}"
TEAM_NAME=trabit  # this will also be the container name
RESULT_LOCATION="/home/lucasf/data/test_FeTA_inference/${SUB_NAME}"

# Create the docker container
docker run -dit \
    -v ${TEST_INPUT_IMG}:/input_img/:ro \
    -v ${TEST_INPUT_META}:/input_meta/:ro \
    -v /output \
    --name ${TEAM_NAME} \
    --gpus all \
    feta_challenge/${TEAM_NAME}

# Run the segmentation algorithm
docker exec ${TEAM_NAME} python /feta_seg/example.py

# Copy the result
docker cp ${TEAM_NAME}:/output ${RESULT_LOCATION}

# Stop and delete the docker container
# otherwise we cannot reuse the name 'trabit' for a new container
docker stop ${TEAM_NAME}
docker rm -v ${TEAM_NAME}
