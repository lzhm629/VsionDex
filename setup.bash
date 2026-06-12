#!/bin/bash
set -e

ENV_NAME="visiondex"

echo "================================="
echo "Creating conda env: $ENV_NAME"
echo "================================="

conda env create -f environment.yml

echo "================================="
echo "Activating env"
echo "================================="

source activate $ENV_NAME

echo "================================="
echo "Installing pip dependencies"
echo "================================="

pip install --upgrade pip
pip install -r requirements.txt

echo "================================="
echo "DONE"
echo "Run: conda activate visiondex"
echo "================================="