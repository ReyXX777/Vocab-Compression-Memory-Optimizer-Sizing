
#include "common/config.hpp"
#include "corpus/corpus_loader.hpp"
#include "corpus/preprocessor.hpp"
#include "tokenizer/bpe_tokenizer.hpp"
#include "metrics/compression.hpp"
#include "metrics/memory.hpp"
#include "metrics/marginal.hpp"
#include "optimization/optimizer.hpp"
#include "optimization/knee_detector.hpp"
#include "reporting/csv_reporter.hpp"
#include "reporting/table_printer.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <sstream>

using namespace vocab_opt;

static void print_help() {
    std::cout << R"(
vocab_sizing – Determine effective tokenizer vocabulary size balancing compression vs memory

Usage:
  vocab_sizing [options]

Options:
  --train <path>         Train corpus file or directory
  --eval <path>          Eval corpus file (if omitted, train is split 90/10)
  --candidates <list>    Comma separated V: e.g. 32000,48000,64000,96000,128000  (default: 256,512,1000,2000,4000,8000 for demo)
  --dim <int>            Embedding dim d (default 4096)
  --bytes <int>          Bytes per param b (default 2)
  --tied                 Use tied embeddings (default false -> untied 2x)
  --lambda <float>       Penalty lambda per MiB (default 0.0008)
  --min-freq <int>       Min pair frequency for BPE merge (default 2)
  --no-shared            Do NOT use shared trajectory (train each V independently, non-monotonic)
  --out <path>           Output CSV (default results.csv)
  --help                 Show this

Examples:
  ./vocab_sizing --train tests/data/sample_en.txt --eval tests/data/sample_multi.txt --candidates 256,512,1000,2000,4000 --dim 4096 --lambda 0.0008
  ./vocab_sizing --train my_corpus.txt --candidates 32000,48000,64000,96000,128000 --dim 4096 --bytes 2 --out results.csv

Theory:
  rho(V)=B/T(V)  B=eval bytes, T=tokens
  M_embed = V*d*b  , M_total = 2*M_embed if untied
  U(V)=rho - lambda*M_total
  Finite optimum exists if rho bounded and lambda>0.
  First-order interior condition: Delta rho(V*-1) >= lambda*Delta M >= Delta rho(V*)

)";
}

static std::vector<int> parse_candidates(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        out.push_back(std::stoi(tok));
    }
    return out;
}

int main(int argc, char** argv) {
    ExperimentConfig cfg;
    cfg.candidate_vs = {256,512,1000,2000,4000,8000}; // demo-friendly small
    std::string train_arg, eval_arg;
    bool help = false;

    for (int i=1;i<argc;++i) {
        std::string a = argv[i];
        if (a=="--help" || a=="-h") help=true;
        else if (a=="--train" && i+1<argc) train_arg = argv[++i];
        else if (a=="--eval" && i+1<argc) eval_arg = argv[++i];
        else if (a=="--candidates" && i+1<argc) cfg.candidate_vs = parse_candidates(argv[++i]);
        else if (a=="--dim" && i+1<argc) cfg.model.embedding_dim = std::stoi(argv[++i]);
        else if (a=="--bytes" && i+1<argc) cfg.model.bytes_per_param = std::stoi(argv[++i]);
        else if (a=="--tied") cfg.model.tied_embeddings = true;
        else if (a=="--lambda" && i+1<argc) cfg.lambda = std::stod(argv[++i]);
        else if (a=="--min-freq" && i+1<argc) cfg.min_merge_freq = std::stoi(argv[++i]);
        else if (a=="--no-shared") cfg.use_shared_trajectory = false;
        else if (a=="--out" && i+1<argc) cfg.output_csv = argv[++i];
    }

    if (help) { print_help(); return 0; }

    // Load corpus
    std::vector<std::string> train_texts;
    std::vector<std::string> eval_texts;

    corpus::Preprocessor prep(cfg.lowercase);
    corpus::CorpusLoader::LoadedCorpus train_lc, eval_lc;

    try {
        if (!train_arg.empty()) {
            auto paths = corpus::CorpusLoader::discover(train_arg);
            train_lc = corpus::CorpusLoader::load_files(paths);
        } else {
            // fallback to sample
            std::filesystem::path sample = "tests/data/sample_en.txt";
            if (std::filesystem::exists(sample)) {
                train_lc = corpus::CorpusLoader::load_file(sample);
            } else {
                // synthetic corpus if no file
                train_lc.buffer = "The quick brown fox jumps over the lazy dog. "
                                  "Vocabulary sizing balances compression efficiency and memory overhead. "
                                  "Larger vocabularies represent common words with fewer tokens. "
                                  "Embedding table memory grows linearly with V. "
                                  "This is repeated to create corpus. ";
                // repeat
                std::string repeated;
                repeated.reserve(train_lc.buffer.size()*500);
                for (int i=0;i<500;++i) repeated += train_lc.buffer;
                train_lc.buffer = std::move(repeated);
                std::string_view bv(train_lc.buffer);
                std::size_t start=0;
                for (std::size_t j=0;j<bv.size();++j) if (bv[j]=='\n') { train_lc.lines.emplace_back(bv.data()+start, j-start); start=j+1; }
                train_lc.lines.emplace_back(bv.data()+start, bv.size()-start);
            }
        }

        if (!eval_arg.empty()) {
            auto paths = corpus::CorpusLoader::discover(eval_arg);
            eval_lc = corpus::CorpusLoader::load_files(paths);
        } else {
            // split train into train/eval
            auto split = corpus::CorpusLoader::split_train_eval(std::move(train_lc), cfg.train_ratio);
            train_lc = std::move(split.train);
            eval_lc = std::move(split.eval);
        }

        // If eval empty because we used fallback, ensure train_lc still valid
        if (train_lc.lines.empty() && !eval_lc.lines.empty()) {
            // we already split? handled
        }
        if (train_arg.empty() && eval_arg.empty() && train_lc.lines.empty()) {
            // after fallback split already done? Actually we replaced logic
        }

        // Process corpora
        if (train_lc.lines.empty()) {
            // If train_arg was provided but eval not, we split above, train_lc is already set
            // else if both empty we created synthetic in train_lc? need to re-evaluate
        }

        // Ensure train_texts and eval_texts are prepared
        // train_lc may be empty if we did split; handle both
        std::vector<std::string> train_processed, eval_processed;
        if (!train_lc.lines.empty()) {
            train_processed = prep.process_corpus(train_lc.lines);
        }
        if (!eval_lc.lines.empty()) {
            eval_processed = prep.process_corpus(eval_lc.lines);
        }

        // If train_processed empty (because we split earlier and train_lc was moved), reload fallback
        if (train_processed.empty() && eval_processed.empty()) {
            // last resort synthetic
            std::string base = "hello world vocabulary sizing compression memory optimization BPE tokenizer embedding projection ";
            for (int i=0;i<2000;++i) {
                train_processed.push_back(base + std::to_string(i%100));
                if (i%10==0) eval_processed.push_back(base + " eval " + std::to_string(i));
            }
        }
        if (train_processed.empty()) train_processed = eval_processed;
        if (eval_processed.empty()) eval_processed = train_processed;

        std::cout << "Train samples: " << train_processed.size() << "  Eval samples: " << eval_processed.size() << "\n";
        std::cout << "Candidates: ";
        for (auto v: cfg.candidate_vs) std::cout << v << " ";
        std::cout << "\nModel d=" << cfg.model.embedding_dim << " b=" << cfg.model.bytes_per_param
                  << " tied=" << cfg.model.tied_embeddings << " lambda=" << cfg.lambda << "\n";
        std::cout << "Shared trajectory (nested vocab for monotonicity): " << (cfg.use_shared_trajectory?"yes":"no") << "\n";

        // Train tokenizers
        std::vector<tokenizer::BpeTokenizer> tokenizers;
        tokenizers.reserve(cfg.candidate_vs.size());

        if (cfg.use_shared_trajectory) {
            // One large training, evaluate prefixes – guarantees compatible subset property
            int max_v = *std::max_element(cfg.candidate_vs.begin(), cfg.candidate_vs.end());
            std::cout << "\nTraining shared BPE trajectory up to V=" << max_v << " ...\n";
            tokenizer::BpeTokenizer base(max_v);
            base.train(train_processed, cfg.min_merge_freq);
            for (int v: cfg.candidate_vs) {
                tokenizers.push_back(base.truncate(v));
                std::cout << "  truncated V=" << v << " actual=" << tokenizers.back().vocab_size() << "\n";
            }
        } else {
            // Independent training (may violate monotonicity, as analysis warns)
            for (int v: cfg.candidate_vs) {
                std::cout << "\nTraining independent BPE V=" << v << " ...\n";
                tokenizer::BpeTokenizer tok(v);
                tok.train(train_processed, cfg.min_merge_freq);
                tokenizers.push_back(std::move(tok));
            }
        }

        // Evaluate
        std::vector<CandidateResult> results;
        results.reserve(tokenizers.size());

        for (std::size_t i=0;i<tokenizers.size();++i) {
            auto& tok = tokenizers[i];
            int V = cfg.candidate_vs[i];
            // Use parallel evaluation
            auto comp = metrics::evaluate_parallel(tok, eval_processed);
            auto mem = metrics::estimate_memory(static_cast<int>(tok.vocab_size()), cfg.model);
            CandidateResult cr;
            cr.vocab_size = V;
            cr.compression = comp;
            cr.memory = mem;
            results.push_back(cr);
            std::cout << "Eval V=" << V << " B=" << comp.bytes << " T=" << comp.tokens << " rho=" << comp.rho << " tau=" << comp.tau << " M_total_MiB=" << mem.total_mib << "\n";
        }

        // Compute marginals and utility
        metrics::compute_marginals(results);
        optimization::VocabOptimizer optimizer(cfg);
        for (auto& r: results) r.utility = optimizer.compute_utility(r.compression, r.memory);

        // Detect knee
        auto knee_elbow = optimization::KneeDetector::elbow_distance(results);
        auto knee_drop = optimization::KneeDetector::max_marginal_drop(results);
        double kappa = results.empty()?0:results[0].memory.kappa_mib_per_token;
        double lambda_kappa = cfg.lambda * kappa;
        auto knee_thresh = optimization::KneeDetector::threshold_crossing(results, lambda_kappa);

        // Mark knee (use elbow as primary)
        for (auto& r: results) if (r.vocab_size==knee_elbow.vocab_size) r.is_knee = true;

        auto best = optimizer.find_best(results);

        // Print
        reporting::print_table(results, best);
        std::cout << "\nKnee detectors:\n";
        std::cout << "  elbow_distance: V=" << knee_elbow.vocab_size << " score=" << knee_elbow.score << "\n";
        std::cout << "  max_marginal_drop: V=" << knee_drop.vocab_size << " drop=" << knee_drop.score << "\n";
        std::cout << "  threshold_crossing (Delta rho < lambda*Delta M): V=" << knee_thresh.vocab_size << "\n";
        std::cout << "  lambda*kappa threshold = " << lambda_kappa << " (kappa_MiB_per_token=" << kappa << ")\n";

        // Check first-order condition for best
        std::sort(results.begin(), results.end(), [](auto& a, auto& b){return a.vocab_size < b.vocab_size;});
        for (std::size_t i=0;i<results.size();++i) if (results[i].vocab_size==best.vocab_size) {
            if (i>0 && i+1<results.size()) {
                bool ok = optimizer.check_first_order(results, i);
                std::cout << "First-order interior check at best: " << (ok?"PASS":"FAIL") << "\n";
                if (!ok) std::cout << "  Note: global max may still hold even if local condition fails due to non-concave rho.\n";
            }
            break;
        }

        // CSV
        reporting::write_csv(cfg.output_csv, results);
        std::cout << "\nCSV written to " << cfg.output_csv << "\n";

        std::cout << "\nFinal assessment: recommended V=" << best.vocab_size
                  << " (best utility) ; knee at V=" << knee_elbow.vocab_size
                  << ". Final choice should validate against downstream quality & latency (Option 2 systems-aware objective).\n";

    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
