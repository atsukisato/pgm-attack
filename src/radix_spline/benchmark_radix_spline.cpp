#include <iostream>
#include <optional>
#include <vector>
#include <algorithm>
#include <random>
#include <limits>
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <string>
#include <utility>

#include "cli.hpp"
#include "io_uint64.hpp"
#include "timer.hpp"
#include "perf_counters.hpp"
#include <nlohmann/json.hpp>
#include <rs/builder.h>
#include <rs/serializer.h>
#include <boost/sort/spreadsort/spreadsort.hpp>

namespace {

// Reads radix / spline sizes from the wire format produced by rs::Serializer<uint64_t>::ToBytes
// (same layout as third_party/RadixSpline/include/rs/serializer.h). Avoids editing RadixSpline.
std::pair<size_t, size_t> radix_spline_counts_from_serialized_bytes(const std::string& bytes) {
    constexpr size_t key_sz = sizeof(uint64_t);
    constexpr size_t sz_sz = sizeof(size_t);
    const size_t header = key_sz * 2 + sz_sz * 4;  // min, max, num_keys, bits, shift, max_error
    if (bytes.size() < header + sz_sz) {
        throw std::runtime_error("radix_spline: serialized blob too small for header + radix len");
    }
    size_t off = header;
    size_t radix_table_size = 0;
    std::memcpy(&radix_table_size, bytes.data() + off, sz_sz);
    off += sz_sz;
    const size_t radix_payload = radix_table_size * sizeof(uint32_t);
    if (bytes.size() < off + radix_payload + sz_sz) {
        throw std::runtime_error("radix_spline: serialized blob too small for radix payload + spline len");
    }
    off += radix_payload;
    size_t spline_points_size = 0;
    std::memcpy(&spline_points_size, bytes.data() + off, sz_sz);
    return {radix_table_size, spline_points_size};
}

void verify_search_bound(
    const std::vector<uint64_t>& keys,
    const rs::SearchBound& bound,
    volatile uint64_t& valid_count,
    uint64_t query
) {
    const size_t true_pos =
        static_cast<size_t>(std::lower_bound(keys.begin(), keys.end(), query) - keys.begin());
    if (true_pos >= bound.begin && true_pos < bound.end) {
        valid_count += 1;
    }
}

nlohmann::json benchmark_radix_spline_impl(
    const std::vector<uint64_t>& keys,
    size_t num_radix_bits,
    size_t max_error,
    bool skip_perf
) {
    if (keys.size() < 2) {
        throw std::runtime_error("At least two keys are required for benchmarking.");
    }

    constexpr size_t num_queries = 1'000'000;
    constexpr size_t warmup_num_queries = 1'000'000;
    constexpr size_t runs = 10;

    Timer build_timer;
    rs::Builder<uint64_t> rsb(keys.front(), keys.back(), num_radix_bits, max_error);
    for (const auto& k : keys) {
        rsb.AddKey(k);
    }
    rs::RadixSpline<uint64_t> rs = rsb.Finalize();
    double build_time = build_timer.elapsed_sec();

    std::string serialized;
    rs::Serializer<uint64_t>::ToBytes(rs, &serialized);
    const auto counts = radix_spline_counts_from_serialized_bytes(serialized);
    const size_t num_radix_table_entries = counts.first;
    const size_t num_spline_points = counts.second;
    const size_t num_spline_segments =
        num_spline_points > 0 ? num_spline_points - 1 : 0;

    nlohmann::json j;
    j["index_impl"] = "radix_spline";
    j["max_error"] = max_error;
    j["num_radix_bits"] = num_radix_bits;
    j["num_spline_points"] = num_spline_points;
    j["num_spline_segments"] = num_spline_segments;
    j["num_radix_table_entries"] = num_radix_table_entries;
    j["total_key_num"] = keys.size();
    j["min_key"] = keys.front();
    j["max_key"] = keys.back();

    size_t size_bytes = rs.GetSize();
    j["index_size_in_kb"] = size_bytes / 1024.0;
    j["build_time_sec"] = build_time;

    std::vector<uint64_t> queries(num_queries + warmup_num_queries);
    {
        std::mt19937_64 rng(0);
        std::uniform_int_distribution<size_t> dist(0, keys.size() - 1);
        for (auto& x : queries) {
            x = keys[dist(rng)];
        }
    }

    volatile uint64_t warmup_valid_count = 0;
    for (size_t i = 0; i < warmup_num_queries; ++i) {
        rs::SearchBound bound = rs.GetSearchBound(queries[i]);
        verify_search_bound(keys, bound, warmup_valid_count, queries[i]);
    }

    volatile uint64_t valid_count = 0;
    nlohmann::json runs_json = nlohmann::json::array();
    std::vector<double> times;
    times.reserve(runs);

    for (size_t r = 0; r < runs; ++r) {
        Timer t;
        for (size_t i = warmup_num_queries; i < warmup_num_queries + num_queries; ++i) {
            rs::SearchBound bound = rs.GetSearchBound(queries[i]);
            verify_search_bound(keys, bound, valid_count, queries[i]);
        }
        double sec = t.elapsed_sec();
        times.push_back(sec);

        nlohmann::json one;
        one["query_time_sec"] = sec;
        one["avg_query_time_ns"] = (sec / num_queries) * 1e9;
        runs_json.push_back(one);
    }

    nlohmann::json perf_runs_json = nlohmann::json::array();
    if (!skip_perf) {
        try {
            PerfEventGroup perf;
            perf.open_common_counters();
            for (size_t r = 0; r < runs; ++r) {
                perf.start();
                for (size_t i = warmup_num_queries; i < warmup_num_queries + num_queries; ++i) {
                    rs::SearchBound bound = rs.GetSearchBound(queries[i]);
                    verify_search_bound(keys, bound, valid_count, queries[i]);
                }
                auto perf_res = perf.stop_and_read();

                nlohmann::json one;
                nlohmann::json perf_json;
                for (const auto& kv : perf_res.values) {
                    perf_json[kv.first] = kv.second;
                }
                one["perf"] = perf_json;

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
            std::cerr << "Warning: Failed to open perf events: " << e.what() << std::endl;
            std::cerr << "Continuing without perf measurements..." << std::endl;
        }
    }

    j["num_queries"] = num_queries;
    j["warmup_num_queries"] = warmup_num_queries;
    j["runs"] = runs_json;
    j["perf_runs"] = perf_runs_json;

    std::vector<double> times_sorted = times;
    std::sort(times_sorted.begin(), times_sorted.end());
    double sum = 0.0;
    for (double v : times_sorted) {
        sum += v;
    }
    double mean = sum / runs;
    double median = (runs % 2) ? times_sorted[runs / 2]
                               : 0.5 * (times_sorted[runs / 2 - 1] + times_sorted[runs / 2]);

    double var = 0.0;
    for (double v : times_sorted) {
        var += (v - mean) * (v - mean);
    }
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

}  // namespace

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);
    bool skip_perf = has_flag(args, "--skip-perf");
    std::string keys_path = get(args, "--keys");
    std::string n_str = get(args, "--n", "");
    std::string seed_str = get(args, "--seed", "");

    std::vector<uint64_t> keys;
    std::optional<std::size_t> dataset_start_pos;
    if (!n_str.empty() && !seed_str.empty()) {
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

    std::string max_error_str = get(args, "--max-error", "");
    if (max_error_str.empty()) {
        max_error_str = get(args, "--epsilon", "");
    }
    if (max_error_str.empty()) {
        std::cerr << "Error: --max-error or --epsilon argument is required" << std::endl;
        return 1;
    }
    size_t max_error = std::stoull(max_error_str);

    std::string radix_bits_str = get(args, "--num-radix-bits", "");
    size_t num_radix_bits =
        radix_bits_str.empty() ? 18 : static_cast<size_t>(std::stoull(radix_bits_str));

    nlohmann::json j;
    try {
        j = benchmark_radix_spline_impl(keys, num_radix_bits, max_error, skip_perf);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    if (dataset_start_pos.has_value()) {
        j["dataset_start_pos"] = dataset_start_pos.value();
    }

    std::cout << j.dump(2) << std::endl;
    return 0;
}
