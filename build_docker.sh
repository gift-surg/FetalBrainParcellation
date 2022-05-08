#!/usr/bin/env bash
TEAM_NAME=trabit

# Build the docker image
docker build -f docker/Dockerfile -t feta_challenge/${TEAM_NAME} --rm .
