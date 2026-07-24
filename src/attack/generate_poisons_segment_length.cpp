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

#include "pgm_utils/pgm_helpers.hpp"
#include "minimize_segment_length/minimize_segment_length.hpp"
#include "minimize_segment_length/minimize_segment_length_swing.hpp"
#include "minimize_segment_length/minimize_segment_length_random.hpp"
#include "minimize_segment_length/minimize_segment_length_random_adjacent.hpp"

/**
 * @brief Get a valid key at or near the specified position in the dataset
 * @param input Path to the dataset file
 * @param dataset_start_pos Starting position in the dataset
 * @param total Total number of elements in the dataset
 * @return A valid key (not sentinel) at or near dataset_start_pos
 */
uint64_t get_start_key(const std::string& input, size_t dataset_start_pos, size_t total) {
    constexpr uint64_t SENTINEL = std::numeric_limits<uint64_t>::max();
    
    // Try to get the key at dataset_start_pos
    std::vector<uint64_t> temp_keys = read_uint64_bin_range(input, dataset_start_pos, 1);
    if (!temp_keys.empty() && temp_keys[0] != SENTINEL) {
        return temp_keys[0];
    }
    
    // If the key at dataset_start_pos is sentinel, find the next valid key
    for (size_t i = dataset_start_pos + 1; i < total && i < dataset_start_pos + 1000; ++i) {
        temp_keys = read_uint64_bin_range(input, i, 1);
        if (!temp_keys.empty() && temp_keys[0] != SENTINEL) {
            return temp_keys[0];
        }
    }
    
    throw std::runtime_error("Could not find valid key near start position");
}

/**
 * @brief Read keys from dataset until we have a complete segment
 * @param input Path to the dataset file
 * @param dataset_start_pos Starting position in the dataset (index in the *full* dataset)
 * @param epsilon Epsilon parameter for PGM-index
 * @param total Total number of elements in the dataset
 * @param keys Output vector for the read keys
 * @param relative_start_pos Output position relative to the read range (index in `keys`)
 * @param read_count Output number of keys read (in the raw read range; after sentinel removal, keys.size() may differ)
 */
 void read_keys_until_complete_segment(
    const std::string& input,
    size_t dataset_start_pos,
    size_t epsilon,
    size_t total,
    std::vector<uint64_t>& keys,
    size_t& relative_start_pos,
    size_t& read_count
) {
    if (total == 0) throw std::invalid_argument("total is 0");
    if (dataset_start_pos >= total) throw std::invalid_argument("dataset_start_pos out of range");
    if (epsilon == 0) throw std::invalid_argument("epsilon must be >= 1");

    // initial read_count = 10 * epsilon^2 (clamped)
    {
        const size_t max_safe = std::numeric_limits<size_t>::max() / 10;
        size_t eps2 = epsilon;
        if (eps2 > 0 && eps2 > max_safe / eps2) eps2 = max_safe; // avoid overflow in eps2*eps2
        else eps2 = eps2 * eps2;
        if (eps2 > max_safe) eps2 = max_safe;
        read_count = 10 * eps2;
        if (read_count == 0) read_count = 1;
        if (read_count > total) read_count = total;
    }

    relative_start_pos = 0;

    const size_t kMaxIters = 64;
    for (size_t iter = 0; iter < kMaxIters; ++iter) {
        if (dataset_start_pos + read_count > total) {
            read_count = total - dataset_start_pos;
        }
        keys = read_uint64_bin_range(input, dataset_start_pos, read_count);
        // Sort if needed
        if (!std::is_sorted(keys.begin(), keys.end())) {
            boost::sort::spreadsort::spreadsort(keys.begin(), keys.end());
        }
        if (read_count == total - dataset_start_pos) {
            // keys, relative_start_pos, read_count
            return;
        }
        size_t segment_end = pgm_util::extend_segment_end(keys, relative_start_pos, epsilon, dataset_start_pos);
        if (segment_end >= keys.size() - 1) {
            read_count *= 2;
            continue;
        } else {
            // keys, relative_start_pos, read_count
            return;
        }
    }

    throw std::runtime_error("Exceeded max iterations while trying to read a complete segment");
}


int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);

    std::string input = get(args, "--keys");
    std::string output = get(args, "--output");
    std::string seed_str = get(args, "--seed", "");
    std::string epsilon_str = get(args, "--epsilon", "");
    std::string lambda_per_segment_str = get(args, "--lambda-per-segment", "");
    std::string method = get(args, "--method", "minimize_segment_length");
    bool allow_duplicates = (get(args, "--allow-duplicates", "false") == "true");

    if (seed_str.empty()) {
        std::cerr << "Error: --seed is required" << std::endl;
        return 1;
    }
    if (epsilon_str.empty()) {
        std::cerr << "Error: --epsilon is required" << std::endl;
        return 1;
    }
    if (lambda_per_segment_str.empty()) {
        std::cerr << "Error: --lambda-per-segment is required" << std::endl;
        return 1;
    }

    uint64_t seed = std::stoull(seed_str);
    size_t epsilon = std::stoull(epsilon_str);
    size_t lambda_per_segment = std::stoull(lambda_per_segment_str);

    // Get total dataset size
    std::size_t total = read_uint64_bin_count(input);
    if (total == 0) {
        std::cerr << "Error: Dataset is empty" << std::endl;
        return 1;
    }

    // Select random start_pos in the entire dataset
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::size_t> dist(0, total - 1);
    size_t dataset_start_pos = dist(rng);

    // Read keys incrementally until we have one complete segment
    std::vector<uint64_t> keys;
    size_t relative_start_pos = 0;
    size_t read_count = 0;
    
    try {
        read_keys_until_complete_segment(input, dataset_start_pos, epsilon, total, keys, relative_start_pos, read_count);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    if (!allow_duplicates) {
        if (std::adjacent_find(keys.begin(), keys.end()) != keys.end()) {
            std::cerr << "Error: Duplicates not allowed but input data contains duplicate keys. Use --allow-duplicates true for datasets with duplicates." << std::endl;
            return 1;
        }
    }

    // Calculate initial segment length (before attack)
    // Use dataset_start_pos as y_offset to represent the position in the full dataset
    size_t initial_segment_end = pgm_util::extend_segment_end(keys, relative_start_pos, epsilon, dataset_start_pos);
    size_t covered_keys_before_attack = initial_segment_end - relative_start_pos + 1;

    std::vector<uint64_t> poisons;
    Timer timer;

    // Generate poisons for the selected segment
    if (method == "minimize_segment_length_random") {
        minimize_segment_length::random(
            keys,
            relative_start_pos,
            initial_segment_end,
            poisons,
            lambda_per_segment,
            seed,
            allow_duplicates
        );
    } else if (method == "minimize_segment_length_random_adjacent") {
        minimize_segment_length::random_adjacent(
            keys,
            relative_start_pos,
            initial_segment_end,
            poisons,
            lambda_per_segment,
            seed,
            allow_duplicates
        );
    } else if (method == "minimize_segment_length") {
        auto result = compute_optimal_consecutive_multiple_poisons_minimize_segment_length(
            keys,
            relative_start_pos,
            epsilon,
            lambda_per_segment,
            dataset_start_pos,
            allow_duplicates
        );
        poisons = result.first;
    } else if (method == "minimize_segment_length_swing") {
        auto result = compute_consecutive_multiple_poisons_minimize_segment_length_swing(
            keys,
            relative_start_pos,
            epsilon,
            lambda_per_segment,
            dataset_start_pos,
            allow_duplicates
        );
        poisons = result.first;
    } else {
        std::cerr << "Unknown method: " << method << std::endl;
        return 1;
    }

    double elapsed = timer.elapsed_sec();
    write_uint64_bin(output, poisons);

    std::vector<uint64_t> original_keys = keys;
    size_t start_pos = relative_start_pos;
    size_t i_in_poisoned_data = dataset_start_pos;
    std::sort(poisons.begin(), poisons.end());
    size_t segment_end_in_keys;
    size_t covered_keys_after_attack;

    segment_end_in_keys = pgm_util::get_segment_end_in_keys(keys, original_keys, poisons, epsilon, i_in_poisoned_data, start_pos, allow_duplicates);
    covered_keys_after_attack = segment_end_in_keys - start_pos + 1;

    nlohmann::json j;
    j["method"] = method;
    j["lambda_per_segment"] = lambda_per_segment;
    j["seed"] = seed;
    j["epsilon"] = epsilon;
    j["dataset_start_pos"] = dataset_start_pos;
    j["relative_start_pos"] = relative_start_pos;
    j["keys_read_count"] = read_count;
    j["covered_keys_before_attack"] = covered_keys_before_attack;
    j["covered_keys_after_attack"] = covered_keys_after_attack;
    j["segment_end_in_keys"] = segment_end_in_keys;
    j["generation_time_sec"] = elapsed;
    j["allow_duplicates"] = allow_duplicates;
    j["num_poisons_generated"] = poisons.size();
    std::cout << j.dump(2) << std::endl;
    return 0;
}
