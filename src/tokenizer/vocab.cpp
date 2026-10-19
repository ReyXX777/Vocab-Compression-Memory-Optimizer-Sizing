
#include "tokenizer/vocab.hpp"
#include <stdexcept>

namespace vocab_opt::tokenizer {

Vocab::Vocab(int target_size) : target_size_(target_size) {
    reserve(target_size);
    // Byte-level base: 256 tokens
    for (int i=0;i<256;++i) {
        std::string s(1, static_cast<char>(i));
        id_to_token_.push_back(s);
        token_to_id_[s] = i;
    }
}

void Vocab::reserve(std::size_t n) {
    id_to_token_.reserve(n);
    token_to_id_.reserve(n*2);
    merges_.reserve(n);
    merge_rank_.reserve(n*2);
}

TokenId Vocab::add_token(std::string token) {
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) return it->second;
    TokenId id = static_cast<TokenId>(id_to_token_.size());
    token_to_id_[token] = id;
    id_to_token_.push_back(std::move(token));
    return id;
}

TokenId Vocab::add_token_if_not_exists(std::string_view token) {
    // Avoid temporary creation in scoring loops: check with string_view lookup via temporary string only when needed
    // unordered_map with string key: create string only on miss path optimization via find with constructed string
    // For true zero-copy we'd use heterogeneous lookup (C++20), but keep simple
    auto it = token_to_id_.find(std::string(token));
    if (it != token_to_id_.end()) return it->second;
    return add_token(std::string(token));
}

bool Vocab::contains(std::string_view token) const {
    return token_to_id_.find(std::string(token)) != token_to_id_.end();
}

TokenId Vocab::get_id(std::string_view token) const {
    auto it = token_to_id_.find(std::string(token));
    if (it == token_to_id_.end()) return -1;
    return it->second;
}

const std::string& Vocab::get_token(TokenId id) const {
    if (id <0 || static_cast<std::size_t>(id) >= id_to_token_.size()) throw std::out_of_range("Vocab id out of range");
    return id_to_token_[id];
}

void Vocab::add_merge(TokenId a, TokenId b, TokenId merged_id) {
    merges_.emplace_back(a,b);
    int rank = static_cast<int>(merges_.size()-1);
    merge_rank_[pair_key(a,b)] = rank;
    (void)merged_id; // merged_id is implied by vocab addition order
}

int Vocab::merge_rank(TokenId a, TokenId b) const {
    auto it = merge_rank_.find(pair_key(a,b));
    if (it==merge_rank_.end()) return INT32_MAX;
    return it->second;
}

bool Vocab::is_valid_id(TokenId id) const {
    return id>=0 && static_cast<std::size_t>(id) < id_to_token_.size();
}

} // namespace vocab_opt::tokenizer
