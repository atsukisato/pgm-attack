#pragma once
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * Binary format: first 8 bytes is the element count n (uint64_t), followed by n*8 bytes is the data.
 * Return vector does not include the element count, only the data (length n).
 */
inline std::vector<uint64_t> read_uint64_bin(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Failed to open for read: " + path);

    std::uint64_t n = 0;
    if (!ifs.read(reinterpret_cast<char*>(&n), sizeof(std::uint64_t))) {
        throw std::runtime_error("Failed to read element count: " + path);
    }

    if (n > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("Element count too large for this platform: " + path);
    }

    std::vector<std::uint64_t> v(static_cast<std::size_t>(n));
    if (n == 0) return v;

    if (!ifs.read(reinterpret_cast<char*>(v.data()),
                  static_cast<std::streamsize>(v.size() * sizeof(std::uint64_t)))) {
        throw std::runtime_error("read() failed: " + path);
    }
    return v;
}

/**
 * Read only the first 8 bytes of the binary format, return the element count n.
 * Share the format with read_uint64_bin_range.
 */
inline std::size_t read_uint64_bin_count(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Failed to open for read: " + path);

    std::uint64_t n = 0;
    if (!ifs.read(reinterpret_cast<char*>(&n), sizeof(std::uint64_t))) {
        throw std::runtime_error("Failed to read element count: " + path);
    }

    if (n > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("Element count too large for this platform: " + path);
    }
    return static_cast<std::size_t>(n);
}

/**
 * Read count elements from the file at path, starting from from_index (0-indexed).
 * Throw if the number of elements is less than count (from_index + count > n).
 */
inline std::vector<uint64_t> read_uint64_bin_range(const std::string& path,
                                                   std::size_t from_index,
                                                   std::size_t count) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Failed to open for read: " + path);

    std::uint64_t n = 0;
    if (!ifs.read(reinterpret_cast<char*>(&n), sizeof(std::uint64_t))) {
        throw std::runtime_error("Failed to read element count: " + path);
    }

    if (from_index + count > static_cast<std::size_t>(n)) {
        throw std::runtime_error("Range out of bounds: path=" + path +
                                " from_index=" + std::to_string(from_index) +
                                " count=" + std::to_string(count) +
                                " total_elements=" + std::to_string(static_cast<std::uint64_t>(n)));
    }

    if (count == 0) return {};

    const std::streamoff offset = static_cast<std::streamoff>(sizeof(std::uint64_t) + from_index * sizeof(std::uint64_t));
    if (!ifs.seekg(offset, std::ios::beg)) {
        throw std::runtime_error("seekg() failed: " + path);
    }

    std::vector<std::uint64_t> v(count);
    if (!ifs.read(reinterpret_cast<char*>(v.data()),
                  static_cast<std::streamsize>(v.size() * sizeof(std::uint64_t)))) {
        throw std::runtime_error("read() failed: " + path);
    }
    return v;
}

/**
 * Binary format: write the element count (v.size()) to the first 8 bytes, followed by the contents of v.
 */
inline void write_uint64_bin(const std::string& path, const std::vector<uint64_t>& v) {
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) throw std::runtime_error("Failed to open for write: " + path);

    const std::uint64_t n = static_cast<std::uint64_t>(v.size());
    if (!ofs.write(reinterpret_cast<const char*>(&n), sizeof(std::uint64_t))) {
        throw std::runtime_error("write() count failed: " + path);
    }
    if (n > 0 && !ofs.write(reinterpret_cast<const char*>(v.data()),
                            static_cast<std::streamsize>(v.size() * sizeof(std::uint64_t)))) {
        throw std::runtime_error("write() data failed: " + path);
    }
    ofs.flush();
    if (!ofs) throw std::runtime_error("flush() failed: " + path);
}
