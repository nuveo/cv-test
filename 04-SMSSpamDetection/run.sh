#!/bin/bash

cd docker
docker-compose up -d --build
docker attach docker_spamfilter_1