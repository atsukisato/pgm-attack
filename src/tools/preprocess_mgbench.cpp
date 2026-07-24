#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "io_uint64.hpp"

namespace {

inline bool is_digit(char c) {
    return static_cast<unsigned>(c - '0') < 10;
}

inline int parse2(const char* p) {
    return (p[0] - '0') * 10 + (p[1] - '0');
}

inline int parse3(const char* p) {
    return (p[0] - '0') * 100 + (p[1] - '0') * 10 + (p[2] - '0');
}

inline int parse4(const char* p) {
    return (p[0] - '0') * 1000 + (p[1] - '0') * 100 + (p[2] - '0') * 10 + (p[3] - '0');
}

// Howard Hinnant's days-from-civil algorithm
// Returns number of days since 1970-01-01.
constexpr int64_t days_from_civil(int y, unsigned m, unsigned d) noexcept {
    y -= (m <= 2);
    const int era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);               // [0, 399]
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;    // [0, 365]
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;              // [0, 146096]
    return era * 146097 + static_cast<int>(doe) - 719468;
}

/**
 * Fast parser:
 *   bench2: "2012-01-01 00:00:00"      -> seconds
 *   bench3: "2017-09-07 01:56:47.073"  -> milliseconds
 *
 * Note:
 * - This treats the timestamp as a civil time and converts it deterministically
 *   to Unix time without relying on local timezone / DST.
 * - For learned-index experiments, this is usually preferable because ordering is preserved.
 */
bool parse_log_time_fast(const char* s, size_t n, uint64_t& out, bool use_milliseconds) {
    if (n < 19) return false;

    // Check fixed separators
    if (s[4] != '-' || s[7] != '-' || s[10] != ' ' || s[13] != ':' || s[16] != ':') {
        return false;
    }

    // Check digits in YYYY-MM-DD HH:MM:SS
    const int digit_pos[] = {0,1,2,3,5,6,8,9,11,12,14,15,17,18};
    for (int pos : digit_pos) {
        if (!is_digit(s[pos])) return false;
    }

    const int year   = parse4(s + 0);
    const int month  = parse2(s + 5);
    const int day    = parse2(s + 8);
    const int hour   = parse2(s + 11);
    const int minute = parse2(s + 14);
    const int second = parse2(s + 17);

    if (month < 1 || month > 12) return false;
    if (day < 1 || day > 31) return false;
    if (hour < 0 || hour > 23) return false;
    if (minute < 0 || minute > 59) return false;
    if (second < 0 || second > 59) return false;

    int ms = 0;
    if (use_milliseconds) {
        if (n >= 23 && s[19] == '.' && is_digit(s[20]) && is_digit(s[21]) && is_digit(s[22])) {
            ms = parse3(s + 20);
        } else {
            ms = 0;
        }
    }

    const int64_t days = days_from_civil(year, static_cast<unsigned>(month), static_cast<unsigned>(day));
    if (days < 0) return false;

    const uint64_t sec =
        static_cast<uint64_t>(days) * 86400ULL +
        static_cast<uint64_t>(hour) * 3600ULL +
        static_cast<uint64_t>(minute) * 60ULL +
        static_cast<uint64_t>(second);

    out = use_milliseconds ? sec * 1000ULL + static_cast<uint64_t>(ms) : sec;
    return true;
}

} // namespace

int main(int argc, char* argv[]) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::string input_path;
    std::string output_path;
    bool use_milliseconds = false;
    size_t reserve_n = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--unit") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --unit requires seconds or milliseconds\n";
                return 1;
            }
            std::string unit = argv[++i];
            if (unit == "seconds") {
                use_milliseconds = false;
            } else if (unit == "milliseconds") {
                use_milliseconds = true;
            } else {
                std::cerr << "Error: --unit must be 'seconds' or 'milliseconds'\n";
                return 1;
            }
        } else if (arg == "--reserve" && i + 1 < argc) {
            reserve_n = static_cast<size_t>(std::stoull(argv[++i]));
        }
    }

    if (input_path.empty() || output_path.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " --input <csv> --output <bin> --unit seconds|milliseconds [--reserve N]\n";
        return 1;
    }

    std::ifstream ifs(input_path, std::ios::binary);
    if (!ifs) {
        std::cerr << "Error: Failed to open input: " << input_path << "\n";
        return 1;
    }

    // Larger file buffer
    std::vector<char> filebuf(1 << 20); // 1 MiB
    ifs.rdbuf()->pubsetbuf(filebuf.data(), static_cast<std::streamsize>(filebuf.size()));

    std::vector<uint64_t> timestamps;
    if (reserve_n > 0) timestamps.reserve(reserve_n);

    std::string line;
    line.reserve(512);

    // Skip header
    if (!std::getline(ifs, line)) {
        std::cerr << "Error: Empty input file\n";
        return 1;
    }

    size_t bad_lines = 0;

    while (std::getline(ifs, line)) {
        if (line.empty()) continue;

        uint64_t ts;
        if (!parse_log_time_fast(line.data(), line.size(), ts, use_milliseconds)) {
            ++bad_lines;
            continue;
        }
        timestamps.push_back(ts);

        if (timestamps.size() % 1'000'000 == 0) {
            std::cout << "Parsed " << timestamps.size() << " timestamps from " << input_path << "\n";
        }
    }

    std::cout << "Parsed " << timestamps.size() << " timestamps from " << input_path << "\n";
    if (bad_lines > 0) {
        std::cout << "Skipped " << bad_lines << " malformed lines\n";
    }

    std::sort(timestamps.begin(), timestamps.end());
    write_uint64_bin(output_path, timestamps);
    std::cout << "Wrote to " << output_path << "\n";

    return 0;
}