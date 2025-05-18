

#!/bin/bash

# Define arrays for store names and retrieval methods
STORE_NAMES=("granite" "mini_gte" "MiniLM" "specter2_base")
RETRIEVAL_METHODS=("hybrid" "embedding" "keyword")

# Loop through each store name and retrieval method
for store in "${STORE_NAMES[@]}"; do
    for method in "${RETRIEVAL_METHODS[@]}"; do
        echo "Running benchmark for store: $store, method: $method"
        python -m semantic_search.benchmarking.benchmark_single \
            --store_name "$store" --experiment_name "$store-$method" --retrieval_method "$method"
    done
done

