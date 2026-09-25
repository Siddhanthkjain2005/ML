#!/bin/bash
set -eu
shutdown -h +720
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-venv python3-pip unzip rsync
mkdir -p /home/ubuntu/amazonml
chown ubuntu:ubuntu /home/ubuntu/amazonml
