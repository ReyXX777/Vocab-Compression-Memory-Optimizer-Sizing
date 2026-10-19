
#include "common/utf8.hpp"
#include <string>

namespace vocab_opt::utf8 {

bool is_continuation(unsigned char b) { return (b & 0xC0) == 0x80; }

std::size_t codepoint_len(unsigned char lead) {
    if ((lead & 0x80) == 0) return 1;
    if ((lead & 0xE0) == 0xC0) return 2;
    if ((lead & 0xF0) == 0xE0) return 3;
    if ((lead & 0xF8) == 0xF0) return 4;
    return 0; // invalid
}

bool is_char_boundary(std::string_view s, std::size_t idx) {
    if (idx == 0 || idx == s.size()) return true;
    unsigned char b = static_cast<unsigned char>(s[idx]);
    return !is_continuation(b);
}

bool is_valid_utf8(std::string_view s) {
    std::size_t i = 0;
    while (i < s.size()) {
        unsigned char lead = static_cast<unsigned char>(s[i]);
        auto len = codepoint_len(lead);
        if (len == 0 || i + len > s.size()) return false;
        for (std::size_t j = 1; j < len; ++j) {
            if (!is_continuation(static_cast<unsigned char>(s[i+j]))) return false;
        }
        i += len;
    }
    return true;
}

std::size_t next_codepoint_len(std::string_view s, std::size_t pos) {
    if (pos >= s.size()) return 0;
    unsigned char lead = static_cast<unsigned char>(s[pos]);
    auto len = codepoint_len(lead);
    if (len == 0) return 0;
    if (pos + len > s.size()) return 0;
    for (std::size_t j=1;j<len;++j) if (!is_continuation(static_cast<unsigned char>(s[pos+j]))) return 0;
    return len;
}

std::string normalize_utf8(std::string_view input) {
    // Aggressively reserve memory
    std::string out;
    out.reserve(input.size() + 16);
    std::size_t i = 0;
    const std::string repl = "ï¿½"; // U+FFFD
    while (i < input.size()) {
        unsigned char lead = static_cast<unsigned char>(input[i]);
        auto len = codepoint_len(lead);
        if (len == 0 || i + len > input.size()) {
            out += repl;
            ++i;
            continue;
        }
        bool ok = true;
        for (std::size_t j=1;j<len;++j) {
            if (!is_continuation(static_cast<unsigned char>(input[i+j]))) { ok=false; break; }
        }
        if (!ok) {
            out += repl;
            ++i;
        } else {
            // valid codepoint, copy slice zero-copy style
            out.append(input.data()+i, len);
            i += len;
        }
    }
    return out;
}

} // namespace vocab_opt::utf8
