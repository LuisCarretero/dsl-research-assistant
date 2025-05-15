
# all-MiniLM-L6-v2
# python -m semantic_search.store.create_store \
#     --model_name sentence-transformers/all-MiniLM-L6-v2 \
#     --max_refs -1

# python -m semantic_search.benchmarking.benchmark_single \
#     --store_name sentence-transformers/all-MiniLM-L6-v2

# # specter2
# python -m semantic_search.store.create_store \
#     --model_name allenai/specter2 \
#     --max_refs -1

# python -m semantic_search.benchmarking.benchmark_single \
#     --store_name allenai/specter2

# prdev/mini-gte
# python -m semantic_search.store.create_store \
#     --model_name prdev/mini-gte \
#     --store_name prdev_mini-gte_meanPooling \
#     --max_refs -1

python -m semantic_search.benchmarking.benchmark_single \
    --store_name prdev_mini-gte_meanPooling
