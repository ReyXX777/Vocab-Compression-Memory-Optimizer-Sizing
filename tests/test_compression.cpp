
#include "tokenizer/bpe_tokenizer.hpp"
#include "metrics/compression.hpp"
#include "metrics/memory.hpp"
#include <iostream>
#include <cassert>

int main() {
    std::vector<std::string> train = {
        "the quick brown fox jumps over the lazy dog",
        "compression is bytes per token",
        "embedding table memory grows linearly",
        "vocabulary sizing balances compression and memory"
    };
    std::vector<std::string> eval = {
        "the quick brown fox",
        "vocabulary sizing"
    };
    vocab_opt::tokenizer::BpeTokenizer tok(1000);
    tok.train(train, 2);
    auto comp = vocab_opt::metrics::evaluate_parallel(tok, eval);
    auto mem = vocab_opt::metrics::estimate_memory(1000, {4096,2,false});
    std::cout << "B=" << comp.bytes << " T=" << comp.tokens << " rho=" << comp.rho << "\n";
    std::cout << "M_embed_MiB=" << mem.embed_mib << " M_total_MiB=" << mem.total_mib << "\n";
    assert(comp.bytes>0);
    assert(comp.tokens>0);
    assert(mem.total_mib>0);
    std::cout << "Compression test PASSED\n";
    return 0;
}
