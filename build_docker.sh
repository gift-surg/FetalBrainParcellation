#!/usr/bin/env bash

# Build the docker image
docker build -f docker/Dockerfile -t lucasfidon/fetal_brain_segmentation:0.1 --rm .
