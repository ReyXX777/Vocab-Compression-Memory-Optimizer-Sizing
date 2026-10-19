
#pragma once
#include <cstdint>
#include <vector>
#include <span>
#include <string>
#include <string_view>
#include <memory_resource>
#include <concepts>

namespace vocab_opt {

using TokenId = int32_t;
using ByteCount = std::size_t;
using TokenCount = std::size_t;

using TokenSeq = std::vector<TokenId>;
using TokenSpan = std::span<const TokenId>;
using TokenMutSpan = std::span<TokenId>;

// pmr aliases for training
namespace pmr = std::pmr;
using PmrTokenSeq = pmr::vector<TokenId>;

// Concept to clarify tokenizer constraints (C++20)
template<typename T>
concept EncodableTokenizer = requires(T tok, std::string_view sv, TokenSpan ts) {
    { tok.encode(sv) } -> std::same_as<TokenSeq>;
    { tok.decode(ts) } -> std::same_as<std::string>;
    { tok.vocab_size() } -> std::convertible_to<std::size_t>;
};

template<typename T>
concept MeasurableTokenizer = EncodableTokenizer<T> && requires(T tok) {
    { tok.vocab_size() } -> std::convertible_to<std::size_t>;
};

} // namespace vocab_opt
