#!/bin/bash
# Script to setup interactive env on Euler

echo "Loading modules"
module load eth_proxy

echo "Loading python environment"
. $HOME/python_envs/dsl-research-assistant/bin/activate

echo "Changing to research assistant directory"
cd $HOME/workspace/dsl/dsl-research-assistant/src

# srun --ntasks=1 --cpus-per-task=4 --mem-per-cpu=8G --gpus=1 --time=4:00:00 --pty bash
# python -m semantic_search.store.create_store --store_name=hybrid-dev
# python -m semantic_search.store.create_store --store_name=hybrid-dev2 --doc_store_columns=all
# python -m semantic_search.benchmarking.benchmark_single --store_name=hybrid-dev3 --experiment_name=hybrid1