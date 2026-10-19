
#pragma once
#include "common/types.hpp"
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace vocab_opt::tokenizer {

// Optimize vocabulary lookups with flat layouts
class Vocab {
public:
    explicit Vocab(int target_size = 256);

    void reserve(std::size_t n);

    // Adds token if not exists, returns id
    TokenId add_token(std::string token);
    TokenId add_token_if_not_exists(std::string_view token);

    bool contains(std::string_view token) const;
    TokenId get_id(std::string_view token) const; // -1 if not found
    const std::string& get_token(TokenId id) const;

    std::size_t size() const { return id_to_token_.size(); }
    std::size_t target_size() const { return target_size_; }

    // Merge management for BPE
    void add_merge(TokenId a, TokenId b, TokenId merged_id);
    int merge_rank(TokenId a, TokenId b) const; // lower = earlier merge, INT_MAX if not merged
    const std::vector<std::pair<TokenId, TokenId>>& merges() const { return merges_; }

    // For encoding: need fast lookup
    const std::vector<std::string>& id_to_token() const { return id_to_token_; }

    // Validate round-trip and reserved tokens
    bool is_valid_id(TokenId id) const;

private:
    static uint64_t pair_key(TokenId a, TokenId b) { return (uint64_t)(uint32_t)a << 32 | (uint32_t)b; }

    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, TokenId> token_to_id_;
    std::vector<std::pair<TokenId, TokenId>> merges_; // in order
    std::unordered_map<uint64_t, int> merge_rank_; // pair_key -> rank
    std::size_t target_size_;
};

} // namespace vocab_opt::tokenizer
