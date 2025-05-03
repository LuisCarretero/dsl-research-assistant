
# all-MiniLM-L6-v2
python -m semantic_search.store.create_store \
    --model_name sentence-transformers/all-MiniLM-L6-v2 \
    --index_metric ip \
    --max_refs -1

python -m semantic_search.benchmarking.benchmark_single \
    --model_name sentence-transformers/all-MiniLM-L6-v2 \
    --metadata_dirpath /cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/raw-data/metadata3 \
    --store_dirpath /cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/db/sentence-transformers_all-MiniLM-L6-v2 \
    --results_dirpath /cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/benchmark_results

specter2
python -m semantic_search.store.create_store \
    --model_name allenai/specter2 \
    --index_metric ip \
    --max_refs -1

python -m semantic_search.benchmarking.benchmark_single \
    --model_name allenai/specter2 \
    --metadata_dirpath /cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/raw-data/metadata3 \
    --store_dirpath /cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/db/allenai_specter2 \
    --results_dirpath /cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/benchmark_results

# prdev/mini-gte
# python -m semantic_search.store.create_store \
#     --model_name prdev/mini-gte \
#     --index_metric ip \
#     --max_refs 1000

# python -m semantic_search.benchmarking.benchmark_single \
#     --model_name prdev/mini-gte \
#     --metadata_dirpath /cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/raw-data/metadata3 \
#     --store_dirpath /cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/db/prdev_mini-gte \
#     --results_dirpath /cluster/home/lcarretero/workspace/dsl/dsl-research-assistant/benchmark_results

