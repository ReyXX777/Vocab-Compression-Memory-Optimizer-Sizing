# Vocabulary Sizing Through Compression–Memory Optimization

Modular C++20 implementation of the algorithm described in the assignment.

## Core Objective

Find V that maximizes compression efficiency rho(V)=B/T(V) while minimizing memory M(V)=V*d*b.

Utility: U(V) = rho(V) - lambda * M(V)

## Architecture

- `include/common/` : types, UTF-8, config, concepts
- `corpus/` : loading, zero-copy slicing with string_view, train/eval split
- `tokenizer/` : Vocab (flat layout), Trie (fast lookup), BPE with pmr
- `metrics/` : compression, memory, marginal gains
- `optimization/` : optimizer (global max) + knee detector (elbow)
- `reporting/` : CSV + console table

## Key C++17/20 Techniques

1. **string_view** zero-copy slicing throughout corpus pipeline
2. **span** (C++20) for token sequences
3. **reserve()** aggressively for dynamic buffers
4. **Flat layouts / Trie** for vocab lookups
5. **std::pmr** monotonic_buffer_resource for BPE training
6. **UTF-8 boundary handling**
7. **Concepts** to constrain Tokenizer
8. **Parallel evaluation** via <execution>
9. **No temporaries** in scoring loops
10. **Memory vs compression** quantification

## Build

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run

```bash
# with sample data
./build/vocab_sizing --train tests/data/sample_en.txt --eval tests/data/sample_multi.txt --candidates 256,512,1000,2000,4000 --dim 4096 --bytes 2 --lambda 0.0008 --out results.csv

# help
./build/vocab_sizing --help
```

Output: console table + CSV with rho, tau, MiB, U, delta_rho, delta_rho_per_MiB, knee.

## Formal Guarantees Implemented

- Monotonicity only assumed when vocabularies are nested (shared BPE trajectory) – implemented via prefix evaluation.
- Diminishing returns treated as empirical, measured.
- Finite optimum via bounded rho + linear penalty.
- First-order condition checked via neighbor conditions.
