#!/bin/sh
cd
cd CS239Project
. .venv/bin/activate
mpirun python demo/fedavg.py
