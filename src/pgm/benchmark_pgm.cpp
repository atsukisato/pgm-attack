#include <iostream>
#include <optional>
#include <vector>
#include <algorithm>
#include <random>
#include <limits>
#include <stdexcept>

#include "cli.hpp"
#include "io_uint64.hpp"
#include "timer.hpp"
#include "perf_counters.hpp"
#include <nlohmann/json.hpp>
#include <pgm/pgm_index.hpp>
#include <boost/sort/spreadsort/spreadsort.hpp>


// Helper class to access protected members of PGMIndex for getting segments per level
template<typename K, size_t Epsilon, size_t EpsilonRecursive, typename Floating = float>
class PGMIndexAccessor : public pgm::PGMIndex<K, Epsilon, EpsilonRecursive, Floating> {
public:
    using Base = pgm::PGMIndex<K, Epsilon, EpsilonRecursive, Floating>;
    
    template<typename RandomIt>
    PGMIndexAccessor(RandomIt first, RandomIt last) : Base(first, last) {}

    std::vector<size_t> get_segments_per_level() const {
        std::vector<size_t> result;
        if (this->levels_offsets.empty()) {
            return result;
        }
        size_t height = this->height();
        result.reserve(height);
        for (size_t i = 0; i < height; ++i) {
            size_t level_start = this->levels_offsets[i];
            size_t level_end = this->levels_offsets[i + 1];
            size_t segment_count = (level_end > level_start) ? (level_end - level_start - 1) : 0;
            result.push_back(segment_count);
        }
        
        return result;
    }
};


void verify_range(
    const std::vector<uint64_t>& keys,
    const pgm::ApproxPos& range,
    volatile uint64_t& valid_count,
    uint64_t query
) {
    auto it = std::lower_bound(keys.begin() + range.lo, keys.begin() + range.hi, query);
    if (it == keys.begin()) {
        if (query <= *it) {
            valid_count += 1;
        }
    } else if (it == keys.end()) {
        if (*(it - 1) < query) {
            valid_count += 1;
        }
    } else {
        if (*(it - 1) < query && query <= *it) {
            valid_count += 1;
        }
    }
}


template<size_t Epsilon>
nlohmann::json benchmark_with_epsilon(const std::vector<uint64_t>& keys, bool skip_perf = false) {
    if (keys.size() < 2) {
        throw std::runtime_error("At least two keys are required for benchmarking.");
    }

    constexpr size_t num_queries = 1'000'000;
    constexpr size_t warmup_num_queries = 1'000'000;
    constexpr size_t runs = 10;

    Timer build_timer;
    PGMIndexAccessor<uint64_t, Epsilon, Epsilon> index(keys.begin(), keys.end());
    double build_time = build_timer.elapsed_sec();

    nlohmann::json j;
    j["epsilon"] = Epsilon;
    j["levels"] = index.height();
    j["total_key_num"] = keys.size();
    j["min_key"] = keys.front();
    j["max_key"] = keys.back();

    auto segments_per_level = index.get_segments_per_level();
    j["segments_per_level"] = segments_per_level;

    size_t total_segments = 0;
    for (size_t c : segments_per_level) total_segments += c;
    j["total_segments"] = total_segments;

    size_t size_bytes = index.size_in_bytes();
    j["index_size_in_kb"] = size_bytes / 1024.0;
    j["build_time_sec"] = build_time;

    // ---- pre-generate queries (outside timing) ----
    std::vector<uint64_t> queries(num_queries + warmup_num_queries);
    {
        std::mt19937_64 rng(0);
        std::uniform_int_distribution<size_t> dist(0, keys.size() - 1);
        for (auto &x : queries) x = keys[dist(rng)];
    }

    // ---- warm-up (not measured) ----
    volatile uint64_t warmup_valid_count = 0;
    for (size_t i = 0; i < warmup_num_queries; ++i) {
        auto range = index.search(queries[i]);
        verify_range(keys, range, warmup_valid_count, queries[i]);
    }

    // ---- measure time ----
    volatile uint64_t valid_count = 0;
    nlohmann::json runs_json = nlohmann::json::array();
    std::vector<double> times;
    times.reserve(runs);

    for (size_t r = 0; r < runs; ++r) {
        Timer t;
        for (size_t i = warmup_num_queries; i < warmup_num_queries + num_queries; ++i) {
            auto range = index.search(queries[i]);
            verify_range(keys, range, valid_count, queries[i]);
        }
        double sec = t.elapsed_sec();
        times.push_back(sec);

        nlohmann::json one;
        one["query_time_sec"] = sec;
        one["avg_query_time_ns"] = (sec / num_queries) * 1e9;
        runs_json.push_back(one);
    }

    // ---- measure perf ----
    nlohmann::json perf_runs_json = nlohmann::json::array();
    if (!skip_perf) {
        try {
            PerfEventGroup perf;
            perf.open_common_counters();
            for (size_t r = 0; r < runs; ++r) {
                perf.start();
                for (size_t i = warmup_num_queries; i < warmup_num_queries + num_queries; ++i) {
                    auto range = index.search(queries[i]);
                    verify_range(keys, range, valid_count, queries[i]);
                }
                auto perf_res = perf.stop_and_read();

                nlohmann::json one;
                nlohmann::json perf_json;
                for (const auto& kv : perf_res.values) {
                    perf_json[kv.first] = kv.second;
                }
                one["perf"] = perf_json;

                // optional: per-query derived metrics
                if (perf_res.values.count("LLC-load-misses")) {
                    one["perf_derived"]["LLC-load-misses_per_query"] =
                        (double)perf_res.values["LLC-load-misses"] / (double)num_queries;
                }
                if (perf_res.values.count("instructions")) {
                    one["perf_derived"]["instructions_per_query"] =
                        (double)perf_res.values["instructions"] / (double)num_queries;
                }
                if (perf_res.values.count("cycles")) {
                    one["perf_derived"]["cycles_per_query"] =
                        (double)perf_res.values["cycles"] / (double)num_queries;
                }

                perf_runs_json.push_back(one);
            }
        } catch (const std::runtime_error& e) {
            // If perf events fail to open, skip perf measurements
            std::cerr << "Warning: Failed to open perf events: " << e.what() << std::endl;
            std::cerr << "Continuing without perf measurements..." << std::endl;
        }
    }

    // ---- write results ----
    j["num_queries"] = num_queries;
    j["warmup_num_queries"] = warmup_num_queries;
    j["runs"] = runs_json;
    j["perf_runs"] = perf_runs_json;

    std::vector<double> times_sorted = times;
    std::sort(times_sorted.begin(), times_sorted.end());
    double sum = 0.0;
    for (double v : times_sorted) sum += v;
    double mean = sum / runs;
    double median = (runs % 2) ? times_sorted[runs/2] : 0.5*(times_sorted[runs/2-1] + times_sorted[runs/2]);

    double var = 0.0;
    for (double v : times_sorted) var += (v - mean) * (v - mean);
    double stdev = std::sqrt(var / runs);

    j["query_time_summary"] = {
        {"mean_sec", mean},
        {"median_sec", median},
        {"min_sec", times_sorted.front()},
        {"max_sec", times_sorted.back()},
        {"stdev_sec", stdev},
        {"mean_avg_query_time_ns", (mean / num_queries) * 1e9},
        {"median_avg_query_time_ns", (median / num_queries) * 1e9}
    };

    j["warmup_valid_count"] = warmup_valid_count;
    j["valid_count"] = valid_count;

    return j;
}


int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);
    bool skip_perf = has_flag(args, "--skip-perf");
    std::string keys_path = get(args, "--keys");
    std::string n_str = get(args, "--n", "");
    std::string seed_str = get(args, "--seed", "");

    std::vector<uint64_t> keys;
    std::optional<std::size_t> dataset_start_pos;
    if (!n_str.empty() && !seed_str.empty()) {
        // Sample contiguous n keys from dataset (same logic as generate_poisons with --n)
        size_t n = std::stoull(n_str);
        uint64_t seed = std::stoull(seed_str);
        std::size_t total = read_uint64_bin_count(keys_path);
        if (total < n) {
            std::cerr << "Error: file has " << total << " elements, cannot read " << n << std::endl;
            return 1;
        }
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<std::size_t> dist(0, total - n);
        std::size_t start = dist(rng);
        dataset_start_pos = start;
        keys = read_uint64_bin_range(keys_path, start, n);
    } else {
        keys = read_uint64_bin(keys_path);
    }

    std::string poisons_path = get(args, "--poisons", "");
    if (!poisons_path.empty()) {
        auto poisons = read_uint64_bin(poisons_path);
        keys.insert(keys.end(), poisons.begin(), poisons.end());
    }

    // Ensure keys are sorted, and PGM-index reserves UINT64_MAX as a sentinel value, so we must exclude it
    if (!std::is_sorted(keys.begin(), keys.end())) {
        boost::sort::spreadsort::spreadsort(keys.begin(), keys.end());
    }
    constexpr uint64_t SENTINEL = std::numeric_limits<uint64_t>::max();
    keys.erase(
        std::remove(keys.begin(), keys.end(), SENTINEL),
        keys.end()
    );
    
    if (keys.empty()) {
        std::cerr << "Error: No valid keys after removing sentinel value" << std::endl;
        return 1;
    }

    // Get epsilon from command line argument
    std::string epsilon_str = get(args, "--epsilon", "");
    if (epsilon_str.empty()) {
        std::cerr << "Error: --epsilon argument is required" << std::endl;
        return 1;
    }
    
    size_t epsilon = std::stoull(epsilon_str);
    nlohmann::json j;
    
    switch (epsilon) {
        case 1:
            j = benchmark_with_epsilon<1>(keys, skip_perf);
            break;
        case 2:
            j = benchmark_with_epsilon<2>(keys, skip_perf);
            break;
        case 4:
            j = benchmark_with_epsilon<4>(keys, skip_perf);
            break;
        case 8:
            j = benchmark_with_epsilon<8>(keys, skip_perf);
            break;
        case 16:
            j = benchmark_with_epsilon<16>(keys, skip_perf);
            break;
        case 32:
            j = benchmark_with_epsilon<32>(keys, skip_perf);
            break;
        case 64:
            j = benchmark_with_epsilon<64>(keys, skip_perf);
            break;
        case 128:
            j = benchmark_with_epsilon<128>(keys, skip_perf);
            break;
        case 256:
            j = benchmark_with_epsilon<256>(keys, skip_perf);
            break;
        case 512:
            j = benchmark_with_epsilon<512>(keys, skip_perf);
            break;
        case 1024:
            j = benchmark_with_epsilon<1024>(keys, skip_perf);
            break;
        case 2048:
            j = benchmark_with_epsilon<2048>(keys, skip_perf);
            break;
        case 4096:
            j = benchmark_with_epsilon<4096>(keys, skip_perf);
            break;
        case 8192:
            j = benchmark_with_epsilon<8192>(keys, skip_perf);
            break;
        case 16384:
            j = benchmark_with_epsilon<16384>(keys, skip_perf);
            break;
        default:
            std::cerr << "Error: Unsupported epsilon value: " << epsilon << std::endl;
            std::cerr << "Supported values: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384" << std::endl;
            return 1;
    }

    if (dataset_start_pos.has_value()) {
        j["dataset_start_pos"] = dataset_start_pos.value();
    }

    std::cout << j.dump(2) << std::endl;
    return 0;
}
