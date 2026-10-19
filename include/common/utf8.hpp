
#pragma once
#include <string_view>
#include <string>
#include <cstdint>

namespace vocab_opt::utf8 {

// Correctly handle UTF-8 multi-byte boundaries
bool is_valid_utf8(std::string_view s);
std::size_t codepoint_len(unsigned char lead);
bool is_continuation(unsigned char b);
bool is_char_boundary(std::string_view s, std::size_t idx);

// Returns byte length of next codepoint, 0 on invalid
std::size_t next_codepoint_len(std::string_view s, std::size_t pos);

// Normalize: replace invalid sequences with U+FFFD, preserve otherwise
// Uses string_view zero-copy scan, reserve for output
std::string normalize_utf8(std::string_view input);

// Byte count of valid text (B in rho = B/T)
inline std::size_t byte_count(std::string_view s) { return s.size(); }

} // namespace vocab_opt::utf8
