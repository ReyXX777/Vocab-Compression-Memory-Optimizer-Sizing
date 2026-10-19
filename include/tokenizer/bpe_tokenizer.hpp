
#pragma once
#include "common/types.hpp"
#include "tokenizer/vocab.hpp"
#include "tokenizer/trie.hpp"
#include <string_view>
#include <vector>
#include <memory_resource>
#include <unordered_map>

namespace vocab_opt::tokenizer {

class BpeTokenizer {
public:
    explicit BpeTokenizer(int vocab_size = 256);
    explicit BpeTokenizer(Vocab vocab);

    const Vocab& vocab() const { return vocab_; }
    std::size_t vocab_size() const { return vocab_.size(); }

    // Train from corpus (vector<string> owned). Uses pmr for zero-alloc training.
    // Returns self for chaining.
    // min_freq: minimum pair frequency to merge
    void train(const std::vector<std::string>& corpus_texts, int min_freq = 2);

    // Train with shared trajectory: caller provides already trained large vocab
    // and we evaluate prefix of its merges to guarantee nested vocab property
    // for Proposition 1 (monotonicity)
    static std::vector<BpeTokenizer> train_trajectory(
        const std::vector<std::string>& corpus,
        const std::vector<int>& candidate_vs,
        int min_freq = 2);

    // Encoding: byte-level BPE with merge ranks
    // Embrace std::span for token sequences, string_view for input
    TokenSeq encode(std::string_view text) const;

    // Batch encode for parallel corpus evaluation (eliminate temporaries)
    TokenCount count_tokens(std::string_view text) const; // fast count without alloc

    std::string decode(TokenSpan ids) const;
    std::string decode(const TokenSeq& ids) const { return decode(TokenSpan(ids)); }

    // For optimizer: create tokenizer with first N merges (nested vocab)
    BpeTokenizer truncate(int new_vocab_size) const;

private:
    Vocab vocab_;
    VocabTrie trie_;

    // Helpers for training
    using Pair = uint64_t;
    static Pair make_pair(TokenId a, TokenId b) { return (uint64_t)(uint32_t)a << 32 | (uint32_t)b; }

    // Encode helpers: merge using ranks
    void apply_merges(std::vector<TokenId>& ids) const;

    void rebuild_trie();
};

} // namespace vocab_opt::tokenizer
