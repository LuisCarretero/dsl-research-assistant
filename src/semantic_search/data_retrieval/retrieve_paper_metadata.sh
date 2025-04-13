#!/bin/bash
#SBATCH --job-name=semanticScholar-retrival
#SBATCH --output=/cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/src/semantic_search/retrieve_paper_metadata.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4:00:00

module load eth_proxy

source $HOME/python_envs/dsl-research-assistant/bin/activate

cd $HOME/workspace/dsl/dsl-research-assistant/src/semantic_search

echo ":- Starting preprocessing job..."
echo ":- Python environment: $VIRTUAL_ENV"
echo ":- Current directory: $(pwd)"
echo ":- Running script: retrieve_paper_metadata.py"

python retrieve_paper_metadata.py

echo ":- Preprocessing job completed."
