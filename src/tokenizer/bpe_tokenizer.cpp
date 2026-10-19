
#include "tokenizer/bpe_tokenizer.hpp"
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace vocab_opt::tokenizer {

BpeTokenizer::BpeTokenizer(int vocab_size) : vocab_(vocab_size) { rebuild_trie(); }

BpeTokenizer::BpeTokenizer(Vocab vocab) : vocab_(std::move(vocab)) { rebuild_trie(); }

void BpeTokenizer::rebuild_trie() {
    trie_.clear();
    trie_.reserve(vocab_.size()*2);
    for (std::size_t i=0;i<vocab_.size();++i) {
        trie_.insert(vocab_.get_token(static_cast<TokenId>(i)), static_cast<TokenId>(i));
    }
}

void BpeTokenizer::train(const std::vector<std::string>& corpus_texts, int min_freq) {
    // Leverage Polymorphic Memory Resources (pmr) for training buffers
    std::pmr::monotonic_buffer_resource pool(1024*1024*10); // 10 MB initial
    std::pmr::vector<std::pmr::vector<TokenId>> words(&pool);
    words.reserve(corpus_texts.size());

    // Convert each pre-tokenized word into sequence of byte ids
    for (auto& txt: corpus_texts) {
        std::pmr::vector<TokenId> ids(&pool);
        ids.reserve(txt.size());
        for (unsigned char b: txt) ids.push_back(static_cast<TokenId>(b));
        if (!ids.empty()) words.push_back(std::move(ids));
    }

    // Frequency map of pairs
    std::unordered_map<Pair, int64_t> pair_freq;
    pair_freq.reserve(8192);

    auto recompute_freq = [&](){
        pair_freq.clear();
        for (auto& w: words) {
            for (std::size_t i=0;i+1<w.size();++i) {
                Pair p = make_pair(w[i], w[i+1]);
                ++pair_freq[p];
            }
        }
    };

    recompute_freq();

    while (vocab_.size() < static_cast<std::size_t>(vocab_.target_size())) {
        if (pair_freq.empty()) break;
        // Find most frequent pair (eliminate temporaries in scoring loop)
        Pair best_pair = 0;
        int64_t best_count = -1;
        for (auto& kv: pair_freq) {
            if (kv.second > best_count) { best_count = kv.second; best_pair = kv.first; }
        }
        if (best_count < min_freq) break;

        TokenId a = static_cast<TokenId>(best_pair >> 32);
        TokenId b = static_cast<TokenId>(best_pair & 0xFFFFFFFFULL);
        if (!vocab_.is_valid_id(a) || !vocab_.is_valid_id(b)) {
            pair_freq.erase(best_pair);
            continue;
        }
        std::string merged_token = vocab_.get_token(a) + vocab_.get_token(b);
        TokenId new_id = vocab_.add_token(merged_token);
        vocab_.add_merge(a,b,new_id);

        // Merge all occurrences
        for (auto& w: words) {
            if (w.size()<2) continue;
            std::pmr::vector<TokenId> new_w(&pool);
            new_w.reserve(w.size());
            std::size_t i=0;
            while (i<w.size()) {
                if (i+1<w.size() && w[i]==a && w[i+1]==b) {
                    new_w.push_back(new_id);
                    i+=2;
                } else {
                    new_w.push_back(w[i]);
                    ++i;
                }
            }
            w = std::move(new_w);
        }
        recompute_freq();
    }
    rebuild_trie();
}

std::vector<BpeTokenizer> BpeTokenizer::train_trajectory(
    const std::vector<std::string>& corpus,
    const std::vector<int>& candidate_vs,
    int min_freq) {

    if (candidate_vs.empty()) return {};
    int max_v = *std::max_element(candidate_vs.begin(), candidate_vs.end());
    BpeTokenizer base(max_v);
    base.train(corpus, min_freq);

    // Now produce truncated tokenizers that share prefix merges -> nested vocab
    // This ensures monotonicity for Proposition 1 when using shared trajectory
    std::vector<BpeTokenizer> out;
    out.reserve(candidate_vs.size());
    for (int v: candidate_vs) {
        out.push_back(base.truncate(v));
    }
    return out;
}

BpeTokenizer BpeTokenizer::truncate(int new_vocab_size) const {
    if (new_vocab_size >= static_cast<int>(vocab_.size())) return *this;
    if (new_vocab_size < 256) new_vocab_size = 256;
    Vocab new_vocab(new_vocab_size);
    // Copy first new_vocab_size tokens (they are the base 256 + earliest merges)
    // Need to reconstruct merges for those
    // Our vocab id_to_token is ordered by creation, so first N is prefix
    for (int i=256; i<new_vocab_size && i < static_cast<int>(vocab_.size()); ++i) {
        const std::string& tok = vocab_.get_token(i);
        new_vocab.add_token(tok);
    }
    // Copy relevant merges: merges that produce tokens < new_vocab_size
    // merges_[k] corresponds to token id 256+k
    for (int k=0; k < new_vocab_size - 256 && k < static_cast<int>(vocab_.merges().size()); ++k) {
        auto [a,b] = vocab_.merges()[k];
        // only add if a,b already in new vocab
        if (a < new_vocab_size && b < new_vocab_size) {
            TokenId mid = 256 + k;
            new_vocab.add_merge(a,b,mid);
        }
    }
    return BpeTokenizer(std::move(new_vocab));
}

void BpeTokenizer::apply_merges(std::vector<TokenId>& ids) const {
    if (ids.size()<2) return;
    // BPE encode: repeatedly merge lowest-rank pair
    while (ids.size()>=2) {
        int best_rank = INT32_MAX;
        std::size_t best_pos = 0;
        bool found = false;
        for (std::size_t i=0;i+1<ids.size();++i) {
            int r = vocab_.merge_rank(ids[i], ids[i+1]);
            if (r < best_rank) {
                best_rank = r;
                best_pos = i;
                found = true;
                if (best_rank==0) break; // early exit, can't be better
            }
        }
        if (!found || best_rank==INT32_MAX) break;
        // Merge at best_pos
        TokenId a = ids[best_pos];
        TokenId b = ids[best_pos+1];
        std::string merged_str = vocab_.get_token(a) + vocab_.get_token(b);
        TokenId merged_id = vocab_.get_id(merged_str);
        if (merged_id==-1) break; // should not happen if rank exists
        std::vector<TokenId> new_ids;
        new_ids.reserve(ids.size()-1);
        for (std::size_t i=0;i<ids.size();++i) {
            if (i==best_pos) {
                new_ids.push_back(merged_id);
                ++i; // skip next
            } else {
                new_ids.push_back(ids[i]);
            }
        }
        ids.swap(new_ids);
    }
}

TokenSeq BpeTokenizer::encode(std::string_view text) const {
    if (text.empty()) return {};
    std::vector<TokenId> ids;
    ids.reserve(text.size()); // aggressive reserve
    for (unsigned char b: text) ids.push_back(static_cast<TokenId>(b));
    apply_merges(ids);
    return ids;
}

TokenCount BpeTokenizer::count_tokens(std::string_view text) const {
    // Fast path without allocating TokenSeq multiple times? Still need merges but avoid extra copies
    if (text.empty()) return 0;
    // Reuse stack buffer for small texts
    std::vector<TokenId> ids;
    ids.reserve(text.size());
    for (unsigned char b: text) ids.push_back(static_cast<TokenId>(b));
    apply_merges(ids);
    return ids.size();
}

std::string BpeTokenizer::decode(TokenSpan ids) const {
    std::string out;
    std::size_t total = 0;
    for (auto id: ids) if (vocab_.is_valid_id(id)) total += vocab_.get_token(id).size();
    out.reserve(total);
    for (auto id: ids) if (vocab_.is_valid_id(id)) out += vocab_.get_token(id);
    return out;
}

} // namespace vocab_opt::tokenizer
