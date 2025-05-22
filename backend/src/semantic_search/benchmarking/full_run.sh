#!/bin/bash

# # Define arrays for store names and retrieval methods

# # Loop through each store name and retrieval method
STORE_NAMES=("granite" "mini_gte" "MiniLM" "specter2_base")
RETRIEVAL_METHODS=("hybrid" "embedding" "keyword")
for store in "${STORE_NAMES[@]}"; do
    for method in "${RETRIEVAL_METHODS[@]}"; do
        echo "Running benchmark for store: $store, method: $method"
        python -m semantic_search.benchmarking.benchmark_single \
            --store_name "$store" --experiment_name "$store-$method" --retrieval_method "$method"
    done
done

# Second test: mini_gte with hybrid retrieval method, iterating over ranker types
# Test with RFR ranker with different k values
K_VALUES=(10 20 30 60 90 120)
for k in "${K_VALUES[@]}"; do
    echo "Running benchmark for store: mini_gte, method: hybrid, ranker: RFR, k: $k"
    python -m semantic_search.benchmarking.benchmark_single \
        --store_name "mini_gte" \
        --experiment_name "mini_gte-hybrid-RFR-k$k" \
        --retrieval_method "hybrid" \
        --hybrid_ranker_type "RFR" \
        --hybrid_ranker_k $k
done

# Test with weighted ranker using different weight combinations
WEIGHT_COMBINATIONS=("0.9,0.1" "0.8,0.2" "0.7,0.3" "0.6,0.4" "0.5,0.5" "0.4,0.6" "0.3,0.7" "0.2,0.8" "0.1,0.9")
for weights in "${WEIGHT_COMBINATIONS[@]}"; do
    echo "Running benchmark for store: mini_gte, method: hybrid, ranker: weighted, weights: $weights"
    python -m semantic_search.benchmarking.benchmark_single \
        --store_name "mini_gte" \
        --experiment_name "mini_gte-hybrid-weighted-$weights" \
        --retrieval_method "hybrid" \
        --hybrid_ranker_type "weighted" \
        --hybrid_ranker_weights "[$weights]"
done

