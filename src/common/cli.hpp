#pragma once
#include <string>
#include <unordered_map>
#include <unordered_set>

struct Args {
    std::unordered_map<std::string, std::string> kv;
    std::unordered_set<std::string> flags;
};

inline Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (i + 1 < argc && arg[0] == '-' && arg[1] == '-') {
            // Check if next argument is also a flag (starts with --)
            if (argv[i + 1][0] != '-' || argv[i + 1][1] != '-') {
                // This is a key-value pair
                args.kv[arg] = argv[i + 1];
                ++i; // Skip the value
            } else {
                // This is a flag (no value)
                args.flags.insert(arg);
            }
        } else if (arg[0] == '-' && arg[1] == '-') {
            // This is a flag at the end
            args.flags.insert(arg);
        }
    }
    return args;
}

inline std::string get(
    const Args& args,
    const std::string& key,
    const std::string& def = ""
) {
    auto it = args.kv.find(key);
    return it == args.kv.end() ? def : it->second;
}

inline bool has_flag(const Args& args, const std::string& flag) {
    return args.flags.find(flag) != args.flags.end();
}
