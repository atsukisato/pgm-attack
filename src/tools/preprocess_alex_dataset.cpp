#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "io_uint64.hpp"

namespace {

constexpr std::uint64_t VALUE_RANGE = 1ULL << 63;  // [0, 2^63)
constexpr std::uint64_t MAX_OUT = VALUE_RANGE - 1;

/**
 * Determine the number of elements from the file size and load.
 * Assume a raw array without a header.
 */
template <class T>
bool load_binary_vec(std::vector<T>& out, const std::string& file_path) {
    std::ifstream is(file_path, std::ios::binary | std::ios::ate);
    if (!is) return false;

    const std::size_t file_bytes = static_cast<std::size_t>(is.tellg());
    if (file_bytes % sizeof(T) != 0) return false;

    const std::size_t n = file_bytes / sizeof(T);
    out.resize(n);

    is.seekg(0, std::ios::beg);
    is.read(reinterpret_cast<char*>(out.data()),
            static_cast<std::streamsize>(file_bytes));
    return is.good();
}

/**
 * Scale doubles to [0, 2^63) and round.
 * Same as normal/lognormal in generate_synthetic_dataset.cpp.
 */
void scale_doubles_to_uint64(const std::vector<double>& data,
                             std::vector<std::uint64_t>& out) {
    out.resize(data.size());
    if (data.empty()) return;

    double raw_min = *std::min_element(data.begin(), data.end());
    double raw_max = *std::max_element(data.begin(), data.end());

    if (raw_min == raw_max) {
        std::fill(out.begin(), out.end(), 0);
        return;
    }

    double scale = static_cast<double>(MAX_OUT) / (raw_max - raw_min);
    for (std::size_t i = 0; i < data.size(); ++i) {
        double scaled = (data[i] - raw_min) * scale;
        std::uint64_t v = static_cast<std::uint64_t>(std::round(scaled));
        if (v > MAX_OUT) v = MAX_OUT;
        out[i] = v;
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    std::ios::sync_with_stdio(false);

    std::string input_path;
    std::string output_path;
    std::string type_str;  // "double" or "uint64"

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--type" && i + 1 < argc) {
            type_str = argv[++i];
        }
    }

    if (input_path.empty() || output_path.empty() || type_str.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " --input <bin> --output <bin> --type double|uint64\n"
                  << "  double: ALEX longitudes/longlat (8-byte floats)\n"
                  << "  uint64: ALEX YCSB (8-byte unsigned ints)\n";
        return 1;
    }

    if (type_str == "double") {
        std::vector<double> data;
        if (!load_binary_vec(data, input_path)) {
            std::cerr << "Error: Failed to load " << input_path << "\n";
            return 1;
        }
        std::cout << "Loaded " << data.size() << " doubles from " << input_path << "\n";

        std::vector<std::uint64_t> out;
        scale_doubles_to_uint64(data, out);
        std::sort(out.begin(), out.end());
        write_uint64_bin(output_path, out);
    } else if (type_str == "uint64") {
        std::vector<std::uint64_t> data;
        if (!load_binary_vec(data, input_path)) {
            std::cerr << "Error: Failed to load " << input_path << "\n";
            return 1;
        }
        std::cout << "Loaded " << data.size() << " uint64 from " << input_path << "\n";

        std::sort(data.begin(), data.end());
        write_uint64_bin(output_path, data);
    } else {
        std::cerr << "Error: --type must be 'double' or 'uint64'\n";
        return 1;
    }

    std::cout << "Wrote to " << output_path << "\n";
    return 0;
}
