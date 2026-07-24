#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>

#ifdef __linux__

#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstring>

class PerfEventGroup {
public:
    struct Result {
        // name -> value
        std::unordered_map<std::string, uint64_t> values;
    };

    PerfEventGroup(bool exclude_kernel = true, bool exclude_hv = true)
        : exclude_kernel_(exclude_kernel), exclude_hv_(exclude_hv) {}

    ~PerfEventGroup() { close_all(); }

    PerfEventGroup(const PerfEventGroup&) = delete;
    PerfEventGroup& operator=(const PerfEventGroup&) = delete;

    // Open a set of common counters:
    // cycles, instructions, branches, branch-misses, LLC-loads, LLC-load-misses
    void open_common_counters() {
        // group leader first
        add_hw_event("cycles", PERF_COUNT_HW_CPU_CYCLES);
        add_hw_event("instructions", PERF_COUNT_HW_INSTRUCTIONS);
        add_hw_event("branches", PERF_COUNT_HW_BRANCH_INSTRUCTIONS);
        add_hw_event("branch-misses", PERF_COUNT_HW_BRANCH_MISSES);

        // LLC (Last Level Cache) loads/misses: use HW_CACHE encoding
        add_hw_cache_event("LLC-loads",
                           PERF_COUNT_HW_CACHE_LL,
                           PERF_COUNT_HW_CACHE_OP_READ,
                           PERF_COUNT_HW_CACHE_RESULT_ACCESS);

        add_hw_cache_event("LLC-load-misses",
                           PERF_COUNT_HW_CACHE_LL,
                           PERF_COUNT_HW_CACHE_OP_READ,
                           PERF_COUNT_HW_CACHE_RESULT_MISS);
    }

    void start() {
        ensure_opened();
        // reset + enable group
        if (ioctl(group_fd_, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) == -1)
            throw std::runtime_error("ioctl(PERF_EVENT_IOC_RESET) failed");
        // compiler barrier
        asm volatile("" ::: "memory");
        if (ioctl(group_fd_, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) == -1)
            throw std::runtime_error("ioctl(PERF_EVENT_IOC_ENABLE) failed");
    }

    Result stop_and_read() {
        ensure_opened();
        // barrier
        asm volatile("" ::: "memory");
        if (ioctl(group_fd_, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) == -1)
            throw std::runtime_error("ioctl(PERF_EVENT_IOC_DISABLE) failed");

        // read group format: nr, time_enabled, time_running, {value,id}...
        // We requested PERF_FORMAT_GROUP | PERF_FORMAT_ID | PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING.
        const size_t n = ids_.size();
        std::vector<uint64_t> buf(3 + 2 * n); // nr + time_enabled + time_running + pairs
        ssize_t bytes = ::read(group_fd_, buf.data(), buf.size() * sizeof(uint64_t));
        if (bytes < 0) throw std::runtime_error("read(perf fd) failed");

        uint64_t nr = buf[0];
        (void)nr; // should equal n
        uint64_t time_enabled = buf[1];
        uint64_t time_running = buf[2];

        Result r;
        // If multiplexing happened, scale values by enabled/running.
        // When enough counters are used, time_running can be < time_enabled.
        double scale = 1.0;
        if (time_running != 0 && time_running != time_enabled) {
            scale = (double)time_enabled / (double)time_running;
        }

        for (size_t i = 0; i < n; ++i) {
            uint64_t value = buf[3 + 2 * i];
            uint64_t id = buf[3 + 2 * i + 1];

            auto it = id_to_name_.find(id);
            if (it != id_to_name_.end()) {
                // scaled (rounded)
                r.values[it->second] = (uint64_t)((double)value * scale);
            }
        }
        return r;
    }

private:
    bool exclude_kernel_;
    bool exclude_hv_;
    int group_fd_ = -1;

    std::vector<int> fds_;
    std::vector<uint64_t> ids_; // perf IDs for each event
    std::unordered_map<uint64_t, std::string> id_to_name_;

    static int perf_event_open(perf_event_attr* attr, pid_t pid, int cpu, int group_fd, unsigned long flags) {
        return (int)syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
    }

    void ensure_opened() const {
        if (group_fd_ == -1) {
            throw std::runtime_error("PerfEventGroup: no counters opened. Call open_common_counters() first.");
        }
    }

    void close_all() {
        for (int fd : fds_) {
            if (fd != -1) ::close(fd);
        }
        fds_.clear();
        ids_.clear();
        id_to_name_.clear();
        group_fd_ = -1;
    }

    perf_event_attr make_base_attr() const {
        perf_event_attr attr;
        std::memset(&attr, 0, sizeof(attr));
        attr.size = sizeof(attr);
        attr.disabled = 1; // start disabled; enable as a group
        attr.exclude_kernel = exclude_kernel_ ? 1 : 0;
        attr.exclude_hv = exclude_hv_ ? 1 : 0;

        // group read format
        attr.read_format =
            PERF_FORMAT_GROUP |
            PERF_FORMAT_ID |
            PERF_FORMAT_TOTAL_TIME_ENABLED |
            PERF_FORMAT_TOTAL_TIME_RUNNING;
        return attr;
    }

    void add_hw_event(const std::string& name, uint64_t hw_config) {
        perf_event_attr attr = make_base_attr();
        attr.type = PERF_TYPE_HARDWARE;
        attr.config = hw_config;

        open_one(name, attr);
    }

    void add_hw_cache_event(const std::string& name, uint64_t cache, uint64_t op, uint64_t result) {
        perf_event_attr attr = make_base_attr();
        attr.type = PERF_TYPE_HW_CACHE;
        // config encoding for PERF_TYPE_HW_CACHE:
        // (cache) | (op << 8) | (result << 16)
        attr.config = cache | (op << 8) | (result << 16);

        open_one(name, attr);
    }

    void open_one(const std::string& name, perf_event_attr& attr) {
        // pid=0, cpu=-1 means "current thread" (good for per-thread region measurement)
        int fd = perf_event_open(&attr, /*pid=*/0, /*cpu=*/-1, /*group_fd=*/group_fd_, /*flags=*/0);
        if (fd == -1) {
            throw std::runtime_error("perf_event_open failed for event: " + name +
                                     " (permission? try: echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid)");
        }

        if (group_fd_ == -1) group_fd_ = fd;
        fds_.push_back(fd);

        uint64_t id = 0;
        if (ioctl(fd, PERF_EVENT_IOC_ID, &id) == -1) {
            throw std::runtime_error("ioctl(PERF_EVENT_IOC_ID) failed for event: " + name);
        }
        ids_.push_back(id);
        id_to_name_[id] = name;
    }
};

#else  // non-Linux fallback (no-op)

class PerfEventGroup {
public:
    struct Result { std::unordered_map<std::string, uint64_t> values; };
    PerfEventGroup(bool = true, bool = true) {}
    void open_common_counters() {}
    void start() {}
    Result stop_and_read() { return {}; }
};

#endif
