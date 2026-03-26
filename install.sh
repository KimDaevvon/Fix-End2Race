#!/bin/bash

cd f1tenth_gym
python -m pip install .
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install quadprog==0.1.13 --no-deps
python -m pip install scipy==1.7.3 --no-deps
cd ..