#!/bin/bash
set -e
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
echo "Running demo with small vocab sizes..."
./build/vocab_sizing --train tests/data/sample_en.txt --eval tests/data/sample_multi.txt --candidates 256,512,1000,2000,4000,8000 --dim 4096 --bytes 2 --lambda 0.0008 --out results.csv
echo "Done. Check results.csv"
cat results.csv
