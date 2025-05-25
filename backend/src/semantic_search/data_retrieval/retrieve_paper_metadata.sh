#!/bin/bash
#SBATCH --job-name=semanticScholar-retrival
#SBATCH --output=/Users/luis/Desktop/ETH/Courses/SS25-DSL/src/semantic_search/data_retrieval/retrieve_paper_metadata.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4:00:00

module load eth_proxy

source $HOME/python_envs/dsl-research-assistant/bin/activate

cd $HOME/workspace/dsl/dsl-research-assistant/src

echo ":- Starting preprocessing job..."
echo ":- Start time: $(date)"
echo ":- Python environment: $VIRTUAL_ENV"
echo ":- Current directory: $(pwd)"
echo ":- Running script: -m src.semantic_search.data_retrieval.retrieve_paper_metadata"

python -m src.semantic_search.data_retrieval.retrieve_paper_metadata \
    --raw_dir /Users/luis/Desktop/ETH/Courses/SS25-DSL/dsl-research-assistant/data/Conversions/opencvf-data/txt \
    --metadata_dir /Users/luis/Desktop/ETH/Courses/SS25-DSL/dsl-research-assistant/data/metadata3 \
    --max_papers -1

echo ":- Preprocessing job completed."
