#include <iostream>
#include <random>
#include <vector>
#include <optional>

#include "cli.hpp"
#include "io_uint64.hpp"
#include "timer.hpp"
#include "pgm_utils/pgm_helpers.hpp"
#include "compute_upper_bound_fix_w_per_block.hpp"
#include <nlohmann/json.hpp>


int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);

    std::string keys_path = get(args, "--keys");
    std::string n_str = get(args, "--n", "");
    std::string seed_str = get(args, "--seed", "");
    size_t lambda = std::stoull(get(args, "--lambda"));
    size_t epsilon = std::stoull(get(args, "--epsilon"));
    std::string method = get(args, "--method", "simple");

    std::vector<uint64_t> keys;
    if (!n_str.empty()) {
        // Sample contiguous n keys from dataset (1M mode)
        size_t n = std::stoull(n_str);
        if (seed_str.empty()) {
            std::cerr << "Error: --seed is required when --n is specified" << std::endl;
            return 1;
        }
        uint64_t seed = std::stoull(seed_str);
        std::size_t total = read_uint64_bin_count(keys_path);
        if (total < n) {
            std::cerr << "Error: file has " << total << " elements, cannot read " << n << std::endl;
            return 1;
        }
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<std::size_t> dist(0, total - n);
        std::size_t start = dist(rng);
        keys = read_uint64_bin_range(keys_path, start, n);
    } else {
        keys = read_uint64_bin(keys_path);
    }

    Timer timer;

    // PGM-index reserves UINT64_MAX as a sentinel value
    constexpr uint64_t SENTINEL = std::numeric_limits<uint64_t>::max();
    
    // Ensure keys are sorted and remove sentinel value to get correct min/max
    std::sort(keys.begin(), keys.end());
    keys.erase(
        std::remove(keys.begin(), keys.end(), SENTINEL),
        keys.end()
    );
    
    size_t ub = 0;

    if (method == "simple") {
        ub = pgm_util::compute_m_opt(keys, epsilon) + lambda;
    } else if (method == "fix_w_per_block") {
        ub = upper_bound_fix_w_per_block::compute_m_opt_after_poisoning_upper_bound(keys, epsilon, lambda);
    } else {
        std::cerr << "Unknown method: " << method << std::endl;
        return 1;
    }
    
    double elapsed = timer.elapsed_sec();

    nlohmann::json j;
    j["lambda"] = lambda;
    j["epsilon"] = epsilon;
    j["method"] = method;
    j["m_opt_upper_bound"] = ub;
    j["computation_time_sec"] = elapsed;

    std::cout << j.dump(2) << std::endl;
}
