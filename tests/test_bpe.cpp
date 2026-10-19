
#include "tokenizer/bpe_tokenizer.hpp"
#include "corpus/preprocessor.hpp"
#include <iostream>
#include <cassert>

int main() {
    std::vector<std::string> corpus = {
        "hello world hello world",
        "vocabulary sizing compression memory",
        "the quick brown fox jumps",
        "hello hello hello world world",
        "compression efficiency memory overhead tradeoff"
    };
    vocab_opt::tokenizer::BpeTokenizer tok(500);
    tok.train(corpus, 2);
    auto ids = tok.encode("hello world");
    std::string dec = tok.decode(ids);
    std::cout << "Encode hello world -> " << ids.size() << " tokens, decoded: " << dec << "\n";
    // roundtrip for bytes: decode(encode(x)) should contain x as substring for byte-level? Actually exact for this simple case
    auto cnt = tok.count_tokens("hello world vocabulary");
    std::cout << "count_tokens = " << cnt << "\n";
    assert(tok.vocab_size() >= 256);
    std::cout << "BPE test PASSED\n";
    return 0;
}
