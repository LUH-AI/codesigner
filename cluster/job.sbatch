#!/bin/bash
# The job a run becomes. ui/services/cluster.py renders this file by filling in
# the doubled-at markers below, so an operator can change how work is submitted
# by editing here rather than in Python.
#
# Those markers rather than Python's own str.format braces: this is a shell
# script, and shell uses braces for its own purposes. A perfectly reasonable
# edit adding a ${SLURM_JOB_ID} would otherwise break the render with a KeyError
# pointing at nothing in particular.
#
# No --gres. These runs are CPU-bound: the expensive half is SMAC's own ask
# (a surrogate refit and an acquisition maximisation per trial), and the models
# are scikit-learn. Reserving a GPU would queue behind work that needs one, to
# leave it idle.
#SBATCH --job-name=codesigner-@@RUN_ID@@
#SBATCH --partition=@@PARTITION@@
#SBATCH --time=@@TIME_LIMIT@@
#SBATCH --cpus-per-task=@@CPUS@@
#SBATCH --output=@@WORKDIR@@/job.out
#SBATCH --error=@@WORKDIR@@/job.err

set -euo pipefail

# Bound the thread pools scikit-learn and its BLAS reach for. Left alone they
# size themselves from the *machine*, not from the allocation, so a model with
# n_jobs=-1 on a many-core node opens a thread per core inside a much smaller
# cgroup and spends its time in contention.
export OMP_NUM_THREADS=@@CPUS@@
export OPENBLAS_NUM_THREADS=@@CPUS@@
export MKL_NUM_THREADS=@@CPUS@@

exec "@@ROOT@@/.venv/bin/python" "@@ROOT@@/cluster/run_experiment.py" "@@WORKDIR@@"
