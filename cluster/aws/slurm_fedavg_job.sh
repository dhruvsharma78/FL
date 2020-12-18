#!/bin/sh
rm -rf /shared/CS239Project
cp -r ${HOME}/CS239Project /shared/
cd /shared/CS239Project
. .venv/bin/activate
mpirun python demo/fedavg.py
