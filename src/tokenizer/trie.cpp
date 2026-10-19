
#include "tokenizer/trie.hpp"

namespace vocab_opt::tokenizer {

VocabTrie::VocabTrie() { nodes_.emplace_back(); }

void VocabTrie::clear() { nodes_.clear(); nodes_.emplace_back(); }

void VocabTrie::reserve(std::size_t n) { nodes_.reserve(n); }

void VocabTrie::insert(std::string_view token, TokenId id) {
    int cur = 0;
    for (unsigned char ch: token) {
        int nxt = nodes_[cur].children[ch];
        if (nxt==-1) {
            nxt = static_cast<int>(nodes_.size());
            nodes_[cur].children[ch]=nxt;
            nodes_.emplace_back();
        }
        cur = nxt;
    }
    nodes_[cur].token_id = id;
}

std::optional<VocabTrie::Match> VocabTrie::longest_match(std::string_view s, std::size_t pos) const {
    int cur=0;
    std::optional<Match> best;
    for (std::size_t i=pos;i<s.size();++i) {
        unsigned char ch = static_cast<unsigned char>(s[i]);
        int nxt = nodes_[cur].children[ch];
        if (nxt==-1) break;
        cur=nxt;
        if (nodes_[cur].token_id!=-1) {
            best = Match{nodes_[cur].token_id, i-pos+1};
        }
    }
    return best;
}

bool VocabTrie::contains(std::string_view token) const {
    int cur=0;
    for (unsigned char ch: token) {
        int nxt = nodes_[cur].children[ch];
        if (nxt==-1) return false;
        cur=nxt;
    }
    return nodes_[cur].token_id!=-1;
}

} // namespace vocab_opt::tokenizer
