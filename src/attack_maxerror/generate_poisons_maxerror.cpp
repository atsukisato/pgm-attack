#include <iostream>
#include <random>
#include <vector>
#include <limits>
#include <algorithm>

#include "cli.hpp"
#include "io_uint64.hpp"
#include "timer.hpp"
#include <nlohmann/json.hpp>
#include <boost/sort/spreadsort/spreadsort.hpp>

#include "maximize_maxerror/random.hpp"
#include "maximize_maxerror/random_adjacent.hpp"
#include "maximize_maxerror/optimal.hpp"
#include "maximize_maxerror/consec.hpp"
#include "maximize_maxerror/dup_optimal.hpp"
#include "maximize_maxerror/greedy.hpp"
#include "maximize_maxerror/minimax_regression.hpp"

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);

    std::string input = get(args, "--keys");
    std::string output = get(args, "--output");
    std::string n_str = get(args, "--n", "");
    std::string seed_str = get(args, "--seed", "");
    size_t lambda = std::stoull(get(args, "--lambda", "0"));
    std::string method = get(args, "--method", "random");
    bool allow_duplicates = (get(args, "--allow-duplicates", "false") == "true");

    if (n_str.empty()) {
        std::cerr << "Error: --n is required" << std::endl;
        return 1;
    }
    if (seed_str.empty()) {
        std::cerr << "Error: --seed is required" << std::endl;
        return 1;
    }

    size_t n = std::stoull(n_str);
    uint64_t seed = std::stoull(seed_str);

    std::size_t total = read_uint64_bin_count(input);
    if (total < n) {
        std::cerr << "Error: file has " << total << " elements, cannot read " << n << std::endl;
        return 1;
    }

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::size_t> dist(0, total - n);
    std::size_t start = dist(rng);

    std::vector<uint64_t> keys = read_uint64_bin_range(input, start, n);

    constexpr uint64_t SENTINEL = std::numeric_limits<uint64_t>::max();
    keys.erase(
        std::remove(keys.begin(), keys.end(), SENTINEL),
        keys.end()
    );

    if (!std::is_sorted(keys.begin(), keys.end())) {
        boost::sort::spreadsort::spreadsort(keys.begin(), keys.end());
    }

    if (keys.empty()) {
        std::cerr << "Error: No valid keys after removing sentinel value" << std::endl;
        return 1;
    }

    if (!allow_duplicates) {
        if (std::adjacent_find(keys.begin(), keys.end()) != keys.end()) {
            std::cerr << "Error: Duplicates not allowed but input data contains duplicate keys. Use --allow-duplicates true for datasets with duplicates." << std::endl;
            return 1;
        }
    }

    std::vector<uint64_t> poisons;
    Timer timer;

    if (method == "random" || method == "maximize_maxerror_random") {
        maximize_maxerror::random(keys, poisons, lambda, seed, allow_duplicates);
    } else if (method == "random_adjacent" || method == "maximize_maxerror_random_adjacent") {
        maximize_maxerror::random_adjacent(keys, poisons, lambda, seed, allow_duplicates);
    } else if (method == "optimal" || method == "maximize_maxerror_optimal") {
        maximize_maxerror::optimal(keys, poisons, lambda, allow_duplicates);
    } else if (method == "consec" || method == "maximize_maxerror_consec") {
        maximize_maxerror::consec(keys, poisons, lambda, allow_duplicates);
    } else if (method == "dup_optimal" || method == "maximize_maxerror_dup_optimal") {
        maximize_maxerror::dup_optimal(keys, poisons, lambda);
    } else if (method == "greedy" || method == "maximize_maxerror_greedy") {
        maximize_maxerror::greedy(keys, poisons, lambda, allow_duplicates);
    } else {
        std::cerr << "Unknown method: " << method << std::endl;
        return 1;
    }

    double elapsed = timer.elapsed_sec();
    write_uint64_bin(output, poisons);

    std::vector<uint64_t> poisoned_keys = keys;
    poisoned_keys.insert(poisoned_keys.end(), poisons.begin(), poisons.end());
    boost::sort::spreadsort::spreadsort(poisoned_keys.begin(), poisoned_keys.end());
    auto [slope, intercept, max_error] = minimax_maxabs_regression_rank(poisoned_keys);

    nlohmann::json j;
    j["method"] = method;
    j["lambda"] = lambda;
    j["seed"] = seed;
    j["n"] = n;
    j["dataset_start_pos"] = start;
    j["generation_time_sec"] = elapsed;
    j["num_poisons_generated"] = poisons.size();
    j["slope"] = slope;
    j["intercept"] = intercept;
    j["max_error"] = max_error;
    j["allow_duplicates"] = allow_duplicates;
    std::cout << j.dump(2) << std::endl;
    return 0;
}
