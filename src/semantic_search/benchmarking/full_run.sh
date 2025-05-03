
# all-MiniLM-L6-v2
python -m semantic_search.store.create_store \
    --model_name sentence-transformers/all-MiniLM-L6-v2 \
    --index_metric l2 \
    --max_refs -1

python -m semantic_search.benchmarking.benchmark_single \
    --model_name sentence-transformers/all-MiniLM-L6-v2

# # specter2
# python -m semantic_search.store.create_store \
#     --model_name allenai/specter2 \
#     --index_metric ip \
#     --max_refs -1

# python -m semantic_search.benchmarking.benchmark_single \
#     --model_name allenai/specter2

# # prdev/mini-gte
# python -m semantic_search.store.create_store \
#     --model_name prdev/mini-gte \
#     --index_metric ip \
#     --max_refs 1000

# python -m semantic_search.benchmarking.benchmark_single \
#     --model_name prdev/mini-gte
