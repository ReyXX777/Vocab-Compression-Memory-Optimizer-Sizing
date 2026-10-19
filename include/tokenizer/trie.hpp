
#pragma once
#include "common/types.hpp"
#include <vector>
#include <array>
#include <string_view>
#include <optional>

namespace vocab_opt::tokenizer {

// Trie for flat, fast vocab lookup (byte-level)
// Optimized for 256-ary branching
struct TrieNode {
    std::array<int32_t, 256> children;
    TokenId token_id = -1;
    TrieNode() { children.fill(-1); }
};

class VocabTrie {
public:
    VocabTrie();

    void clear();
    void reserve(std::size_t nodes);

    // Insert token bytes
    void insert(std::string_view token, TokenId id);

    // Longest prefix match from position pos in s
    struct Match { TokenId id; std::size_t length; };
    std::optional<Match> longest_match(std::string_view s, std::size_t pos) const;

    // For BPE: quick existence check
    bool contains(std::string_view token) const;

private:
    std::vector<TrieNode> nodes_;
};

} // namespace vocab_opt::tokenizer
