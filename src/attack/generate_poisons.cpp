#include <iostream>
#include <optional>
#include <random>
#include <vector>
#include <limits>
#include <algorithm>

#include "cli.hpp"
#include "io_uint64.hpp"
#include "timer.hpp"
#include <nlohmann/json.hpp>
#include <boost/sort/spreadsort/spreadsort.hpp>

#include "pgm_utils/pgm_helpers.hpp"

#include "inject_poisons_to_minimize_segment_length_swing_lambda_with_theta.hpp"
#include "inject_poisons_to_minimize_segment_length_random.hpp"
#include "inject_poisons_to_minimize_segment_length_random_adjacent.hpp"


int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);

    std::string input = get(args, "--keys");
    std::string output = get(args, "--output");
    size_t lambda = std::stoull(get(args, "--lambda", "0"));
    std::string seed_str = get(args, "--seed", "");
    std::string n_str = get(args, "--n", "");
    std::string method = get(args, "--method", "random");
    std::string epsilon_str = get(args, "--epsilon", "");
    bool allow_duplicates = (get(args, "--allow-duplicates", "false") == "true");

    std::vector<uint64_t> keys;
    std::optional<std::size_t> dataset_start_pos;
    if (!n_str.empty()) {
        // Sample contiguous n keys from dataset (like generate_poisons_maxerror)
        size_t n = std::stoull(n_str);
        if (seed_str.empty()) {
            std::cerr << "Error: --seed is required when --n is specified" << std::endl;
            return 1;
        }
        uint64_t seed = std::stoull(seed_str);
        std::size_t total = read_uint64_bin_count(input);
        if (total < n) {
            std::cerr << "Error: file has " << total << " elements, cannot read " << n << std::endl;
            return 1;
        }
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<std::size_t> dist(0, total - n);
        std::size_t start = dist(rng);
        dataset_start_pos = start;
        keys = read_uint64_bin_range(input, start, n);
    } else {
        keys = read_uint64_bin(input);
    }

    uint64_t seed = seed_str.empty() ? 0 : std::stoull(seed_str);

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

    if (!allow_duplicates) {
        if (std::adjacent_find(keys.begin(), keys.end()) != keys.end()) {
            std::cerr << "Error: Duplicates not allowed but input data contains duplicate keys. Use --allow-duplicates true for datasets with duplicates." << std::endl;
            return 1;
        }
    }
    
    std::vector<uint64_t> poisons;

    size_t m_opt_after_poisoning = 0;
    double mu = 20.0;  // for inject_poisons_to_minimize_segment_length_swing_lambda_with_theta

    Timer timer;
    if (method == "random") {
        poisons.resize(lambda);
        std::mt19937_64 rng(seed);
        // Get min/max from keys
        uint64_t min_key = keys.front();
        uint64_t max_key = keys.back();
        std::uniform_int_distribution<uint64_t> dist(min_key, max_key);
        for (size_t i = 0; i < lambda; ++i) {
            uint64_t poison = dist(rng);
            poisons[i] = poison;
        }
    } else if (method == "inject_poisons_to_minimize_segment_length_swing_lambda_with_theta") {
        if (epsilon_str.empty()) {
            std::cerr << "Error: --epsilon is required for inject_poisons_to_minimize_segment_length_swing_lambda_with_theta method" << std::endl;
            return 1;
        }
        if (args.kv.find("--lambda") == args.kv.end()) {
            std::cerr << "Error: --lambda is required for inject_poisons_to_minimize_segment_length_swing_lambda_with_theta method" << std::endl;
            return 1;
        }
        size_t epsilon = std::stoull(epsilon_str);
        mu = std::stod(get(args, "--mu", "20.0"));
        keys.erase(
            std::remove(keys.begin(), keys.end(), SENTINEL),
            keys.end()
        );
        if (keys.empty()) {
            std::cerr << "Error: No valid keys after removing sentinel value" << std::endl;
            return 1;
        }
        m_opt_after_poisoning = inject_poisons_to_minimize_segment_length_swing_lambda_with_theta(keys, poisons, epsilon, lambda, mu, 100, allow_duplicates);
        boost::sort::spreadsort::spreadsort(poisons.begin(), poisons.end());

    } else if (method == "inject_poisons_to_minimize_segment_length_random") {
        if (epsilon_str.empty()) {
            std::cerr << "Error: --epsilon is required for inject_poisons_to_minimize_segment_length_random method" << std::endl;
            return 1;
        }
        if (args.kv.find("--lambda") == args.kv.end()) {
            std::cerr << "Error: --lambda is required for inject_poisons_to_minimize_segment_length_random method" << std::endl;
            return 1;
        }
        if (seed_str.empty()) {
            std::cerr << "Error: --seed is required for inject_poisons_to_minimize_segment_length_random method" << std::endl;
            return 1;
        }
        size_t epsilon = std::stoull(epsilon_str);
        keys.erase(
            std::remove(keys.begin(), keys.end(), SENTINEL),
            keys.end()
        );
        if (keys.empty()) {
            std::cerr << "Error: No valid keys after removing sentinel value" << std::endl;
            return 1;
        }
        m_opt_after_poisoning = inject_poisons_to_minimize_segment_length_random(keys, poisons, epsilon, lambda, seed, allow_duplicates);
        boost::sort::spreadsort::spreadsort(poisons.begin(), poisons.end());

    } else if (method == "inject_poisons_to_minimize_segment_length_random_adjacent") {
        if (epsilon_str.empty()) {
            std::cerr << "Error: --epsilon is required for inject_poisons_to_minimize_segment_length_random_adjacent method" << std::endl;
            return 1;
        }
        if (args.kv.find("--lambda") == args.kv.end()) {
            std::cerr << "Error: --lambda is required for inject_poisons_to_minimize_segment_length_random_adjacent method" << std::endl;
            return 1;
        }
        if (seed_str.empty()) {
            std::cerr << "Error: --seed is required for inject_poisons_to_minimize_segment_length_random_adjacent method" << std::endl;
            return 1;
        }
        size_t epsilon = std::stoull(epsilon_str);
        keys.erase(
            std::remove(keys.begin(), keys.end(), SENTINEL),
            keys.end()
        );
        if (keys.empty()) {
            std::cerr << "Error: No valid keys after removing sentinel value" << std::endl;
            return 1;
        }
        m_opt_after_poisoning =
            inject_poisons_to_minimize_segment_length_random_adjacent(keys, poisons, epsilon, lambda, seed, allow_duplicates);
        boost::sort::spreadsort::spreadsort(poisons.begin(), poisons.end());

    } else {
        std::cerr << "Unknown method: " << method << std::endl;
        return 1;
    }
    
    double elapsed = timer.elapsed_sec();
    write_uint64_bin(output, poisons);

    nlohmann::json j;
    j["lambda"] = lambda;
    if (method != "inject_poisons_to_minimize_segment_length_swing_lambda_with_theta") {
        j["seed"] = seed;
    }
    j["method"] = method;
    j["generation_time_sec"] = elapsed;
    if ((method == "inject_poisons_to_minimize_segment_length_swing_lambda_with_theta" ||
         method == "inject_poisons_to_minimize_segment_length_random" ||
         method == "inject_poisons_to_minimize_segment_length_random_adjacent") && !epsilon_str.empty()) {
        j["epsilon"] = std::stoull(epsilon_str);
        j["m_opt_after_poisoning"] = m_opt_after_poisoning;
    }
    if (method == "inject_poisons_to_minimize_segment_length_swing_lambda_with_theta") {
        j["mu"] = mu;
    }
    if (dataset_start_pos.has_value()) {
        j["dataset_start_pos"] = dataset_start_pos.value();
    }
    j["allow_duplicates"] = allow_duplicates;
    j["num_poisons_generated"] = poisons.size();

    std::cout << j.dump(2) << std::endl;
}
