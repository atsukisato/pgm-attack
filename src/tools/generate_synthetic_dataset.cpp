#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <fstream>
#include <filesystem>
#include <sstream>

#include "io_uint64.hpp"
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

constexpr uint64_t SENTINEL = std::numeric_limits<uint64_t>::max();

/**
 * Generate Zipf-distributed integers using inverse transform sampling.
 * P(k) = 1/k^s / H_N,s  for k = 1, 2, ..., N
 */
std::vector<uint64_t> generate_zipf(size_t n, size_t N, double s, std::mt19937_64& rng) {
    std::vector<long double> cdf(N + 1, 0.0L);
    long double H = 0.0L;
    long double s_ld = static_cast<long double>(s);
    for (size_t i = 1; i <= N; ++i) {
        H += 1.0L / std::pow(static_cast<long double>(i), s_ld);
    }
    for (size_t k = 1; k <= N; ++k) {
        cdf[k] = cdf[k - 1] + (1.0L / std::pow(static_cast<long double>(k), s_ld)) / H;
    }

    std::vector<uint64_t> result;
    result.reserve(n);
    std::uniform_real_distribution<long double> uniform(0.0L, 1.0L);

    for (size_t i = 0; i < n; ++i) {
        long double u = uniform(rng);
        size_t lo = 1, hi = N;
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (cdf[mid] < u) lo = mid + 1;
            else hi = mid;
        }
        result.push_back(static_cast<uint64_t>(lo));
    }
    return result;
}

/**
 * Generate uniform integers in [0, max).
 */
std::vector<uint64_t> generate_uniform(size_t n, uint64_t max_val, std::mt19937_64& rng) {
    std::uniform_int_distribution<uint64_t> dist(0, max_val > 0 ? max_val - 1 : 0);
    std::vector<uint64_t> result;
    result.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        result.push_back(dist(rng));
    }
    return result;
}

/**
 * Generate lognormal(m, s) distributed integers.
 * If X ~ Lognormal(m, s), then log(X) ~ N(m, s).
 * Samples n doubles, scales so min->0 and max->(value_range-1), then rounds to integers.
 * Output is in [0, value_range) — same semantics as uniform (value_range = range size).
 * Sentinel is avoided.
 */
std::vector<uint64_t> generate_lognormal(size_t n, double m, double s, uint64_t value_range,
                                         std::mt19937_64& rng) {
    std::lognormal_distribution<long double> dist(static_cast<long double>(m), static_cast<long double>(s));
    std::vector<long double> raw(n);
    for (size_t i = 0; i < n; ++i) {
        raw[i] = dist(rng);
    }
    long double raw_min = *std::min_element(raw.begin(), raw.end());
    long double raw_max = *std::max_element(raw.begin(), raw.end());

    // value_range = range size; max_out = inclusive max = value_range - 1 (or SENTINEL-1 if overflow)
    uint64_t max_val = (value_range >= SENTINEL) ? (SENTINEL - 1) : value_range;
    uint64_t max_out = (max_val > 0) ? (max_val - 1) : 0;

    std::vector<uint64_t> result;
    result.reserve(n);
    if (raw_min == raw_max) {
        for (size_t i = 0; i < n; ++i) {
            result.push_back(0);
        }
    } else {
        long double scale = static_cast<long double>(max_out) / (raw_max - raw_min);
        for (size_t i = 0; i < n; ++i) {
            long double scaled = (raw[i] - raw_min) * scale;
            uint64_t v = static_cast<uint64_t>(std::round(scaled));
            if (v > max_out) v = max_out;
            result.push_back(v);
        }
    }
    return result;
}

/**
 * Generate N(μ, σ) normal distributed integers.
 * Samples n doubles, scales so min->0 and max->(value_range-1), then rounds to integers.
 * Output is in [0, value_range) — same semantics as uniform and lognormal.
 */
std::vector<uint64_t> generate_normal(size_t n, double mu, double sigma, uint64_t value_range,
                                     std::mt19937_64& rng) {
    std::normal_distribution<long double> dist(static_cast<long double>(mu), static_cast<long double>(sigma));
    std::vector<long double> raw(n);
    for (size_t i = 0; i < n; ++i) {
        raw[i] = dist(rng);
    }
    long double raw_min = *std::min_element(raw.begin(), raw.end());
    long double raw_max = *std::max_element(raw.begin(), raw.end());

    uint64_t max_val = (value_range >= SENTINEL) ? (SENTINEL - 1) : value_range;
    uint64_t max_out = (max_val > 0) ? (max_val - 1) : 0;

    std::vector<uint64_t> result;
    result.reserve(n);
    if (raw_min == raw_max) {
        for (size_t i = 0; i < n; ++i) {
            result.push_back(0);
        }
    } else {
        long double scale = static_cast<long double>(max_out) / (raw_max - raw_min);
        for (size_t i = 0; i < n; ++i) {
            long double scaled = (raw[i] - raw_min) * scale;
            uint64_t v = static_cast<uint64_t>(std::round(scaled));
            if (v > max_out) v = max_out;
            result.push_back(v);
        }
    }
    return result;
}

void print_usage(const char* prog) {
    std::cerr << "Usage:\n"
              << "  " << prog << " uniform <n> <max> [--output <name>] [--seed <seed>]\n"
              << "  " << prog << " zipf <n> <N> <s> [--output <name>] [--seed <seed>]\n"
              << "  " << prog << " lognormal <n> <mu> <sigma> <value_range> [--output <name>] [--seed <seed>]\n"
              << "  " << prog << " normal <n> <mu> <sigma> <value_range> [--output <name>] [--seed <seed>]\n"
              << "\n"
              << "Distributions:\n"
              << "  uniform   : n elements in [0, max) (max is exclusive)\n"
              << "  zipf     : n elements, domain [1, N], exponent s (P(k) ∝ 1/k^s)\n"
              << "  lognormal: n elements, Lognormal(μ, σ), scaled to [0, value_range)\n"
              << "  normal   : n elements, N(μ, σ), scaled to [0, value_range)\n"
              << "\n"
              << "Options:\n"
              << "  --output <name>  Output dataset name (default: <dist>_<params>_<n>)\n"
              << "  --seed <seed>    Random seed (default: 42)\n"
              << "\n"
              << "Examples:\n"
              << "  " << prog << " uniform 1000000 4194304 --output uniform_2^22\n"
              << "  " << prog << " zipf 1000000 1000000 1.0 --output zipf_s1\n"
              << "  " << prog << " lognormal 1000000 0.0 0.5 4294967296 --output lognormal_05\n"
              << "  " << prog << " normal 1000000 0.0 1.0 4294967296 --output normal_std\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string dist = argv[1];
    uint64_t seed = 42;
    std::string output_name;

    // Parse optional args first
    std::vector<std::string> positional;
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output" && i + 1 < argc) {
            output_name = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoull(argv[++i]);
        } else if (arg.substr(0, 2) != "--") {
            positional.push_back(arg);
        }
    }

    std::mt19937_64 rng(seed);
    std::vector<uint64_t> data;
    std::string description;

    if (dist == "uniform") {
        if (positional.size() < 2) {
            std::cerr << "Error: uniform requires <n> <max>\n";
            print_usage(argv[0]);
            return 1;
        }
        size_t n = std::stoull(positional[0]);
        uint64_t max_val = std::stoull(positional[1]);
        data = generate_uniform(n, max_val, rng);
        description = "Uniform [0, " + std::to_string(max_val) + ")";
        if (output_name.empty()) {
            output_name = "uniform_" + std::to_string(max_val) + "_" + std::to_string(n);
        }
    } else if (dist == "zipf") {
        if (positional.size() < 3) {
            std::cerr << "Error: zipf requires <n> <N> <s>\n";
            print_usage(argv[0]);
            return 1;
        }
        size_t n = std::stoull(positional[0]);
        size_t N = std::stoull(positional[1]);
        double s = std::stod(positional[2]);
        data = generate_zipf(n, N, s, rng);
        std::ostringstream oss;
        oss << "Zipf(s=" << s << ") domain [1," << N << "]";
        description = oss.str();
        if (output_name.empty()) {
            output_name = "zipf_s" + positional[2] + "_" + std::to_string(n);
        }
    } else if (dist == "lognormal") {
        if (positional.size() < 4) {
            std::cerr << "Error: lognormal requires <n> <mu> <sigma> <value_range>\n";
            print_usage(argv[0]);
            return 1;
        }
        size_t n = std::stoull(positional[0]);
        double mu = std::stod(positional[1]);
        double sigma = std::stod(positional[2]);
        uint64_t value_range = std::stoull(positional[3]);
        data = generate_lognormal(n, mu, sigma, value_range, rng);
        std::ostringstream oss;
        oss << "Lognormal(μ=" << mu << ", σ=" << sigma << ") scaled to [0," << value_range << ")";
        description = oss.str();
        if (output_name.empty()) {
            output_name = "lognormal_" + positional[2] + "_" + std::to_string(n);
        }
    } else if (dist == "normal") {
        if (positional.size() < 4) {
            std::cerr << "Error: normal requires <n> <mu> <sigma> <value_range>\n";
            print_usage(argv[0]);
            return 1;
        }
        size_t n = std::stoull(positional[0]);
        double mu = std::stod(positional[1]);
        double sigma = std::stod(positional[2]);
        uint64_t value_range = std::stoull(positional[3]);
        data = generate_normal(n, mu, sigma, value_range, rng);
        std::ostringstream oss;
        oss << "N(μ=" << mu << ", σ=" << sigma << ") scaled to [0," << value_range << ")";
        description = oss.str();
        if (output_name.empty()) {
            output_name = "normal_mu" + positional[1] + "_sigma" + positional[2] + "_" + std::to_string(n);
        }
    } else {
        std::cerr << "Error: Unknown distribution '" << dist << "'\n";
        print_usage(argv[0]);
        return 1;
    }

    // Sort and remove sentinel (PGM-index requires no sentinel in data)
    std::sort(data.begin(), data.end());
    data.erase(std::remove(data.begin(), data.end(), SENTINEL), data.end());

    // Output paths
    std::string data_name = output_name + "_uint64";
    std::string data_path = "data/" + data_name;
    std::string config_name = output_name;
    std::string config_path = "configs/datasets/" + config_name + ".json";

    // Create directories
    fs::create_directories(fs::path(data_path).parent_path());
    fs::create_directories(fs::path(config_path).parent_path());

    // Write binary
    write_uint64_bin(data_path, data);
    std::cout << "Generated " << data.size() << " elements -> " << data_path << std::endl;

    // Write config
    nlohmann::json config;
    config["name"] = data_name;
    config["type"] = "uint64";
    config["path"] = data_path;
    config["description"] = description + " (" + std::to_string(data.size()) + " entries)";

    std::ofstream config_file(config_path);
    if (!config_file) {
        std::cerr << "Error: Failed to create config file: " << config_path << std::endl;
        return 1;
    }
    config_file << config.dump(2) << std::endl;
    std::cout << "Created config: " << config_path << std::endl;

    return 0;
}
