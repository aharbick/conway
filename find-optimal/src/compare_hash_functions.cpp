#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Hash table entry (16 bytes for alignment)
struct HashEntry {
    uint64_t pattern;      // 8 bytes
    uint16_t generations;  // 2 bytes
    uint16_t padding[3];   // 6 bytes padding for 16-byte alignment
};

// === TOP 5 CANDIDATE HASH FUNCTIONS ===

// 3-op hash v1: fibonacci * >>32
inline uint64_t hash_fibonacci_32(uint64_t key) {
    key *= 0x9e3779b97f4a7c15ULL;          // 1 op: multiply (Knuth's golden ratio)
    key ^= key >> 32;                      // 2 ops: shift + xor
    return key;                             // Total: 3 operations
}

// 3-op hash v2: splitmix * >>32
inline uint64_t hash_splitmix_32(uint64_t key) {
    key *= 0xbf58476d1ce4e5b9ULL;          // 1 op: multiply (splitmix constant)
    key ^= key >> 32;                      // 2 ops: shift + xor
    return key;                             // Total: 3 operations
}

// 3-op hash v3: xxhash * >>32
inline uint64_t hash_xxhash_32(uint64_t key) {
    key *= 0x9fb21c651e98df25ULL;          // 1 op: multiply (xxhash prime)
    key ^= key >> 32;                      // 2 ops: shift + xor
    return key;                             // Total: 3 operations
}

// 3-op hash v4: splitmix * >>33
inline uint64_t hash_splitmix_33(uint64_t key) {
    key *= 0xbf58476d1ce4e5b9ULL;          // 1 op: multiply (splitmix constant)
    key ^= key >> 33;                      // 2 ops: shift + xor
    return key;                             // Total: 3 operations
}

// 3-op hash v5: cityhash * >>33
inline uint64_t hash_cityhash_33(uint64_t key) {
    key *= 0x9ddfea08eb382d69ULL;          // 1 op: multiply (cityhash constant)
    key ^= key >> 33;                      // 2 ops: shift + xor
    return key;                             // Total: 3 operations
}

// === BASELINE FOR COMPARISON ===

// 6-op hash: splitmix64 finalizer (original baseline)
inline uint64_t hash_splitmix64_full(uint64_t key) {
    key ^= key >> 30;                      // 2 ops: shift + xor
    key *= 0xbf58476d1ce4e5b9ULL;          // 1 op: multiply
    key ^= key >> 27;                      // 2 ops: shift + xor
    key *= 0x94d049bb133111ebULL;          // 1 op: multiply
    return key;                             // Total: 6 operations
}

struct HashStats {
    size_t max_probe_length;
    double avg_probe_length;
    size_t median_probe_length;
    size_t p90_probe_length;
    bool success;
};

HashStats build_and_test_hash(
    const std::vector<uint64_t>& patterns,
    const std::vector<uint16_t>& generations,
    uint64_t (*hash_func)(uint64_t),
    const char* name,
    int ops_count
) {
    size_t num_keys = patterns.size();

    // Create hash table with 70% load factor
    size_t table_size = (num_keys * 10) / 7;
    size_t power = 1;
    while (power < table_size) power <<= 1;
    table_size = power;

    std::vector<HashEntry> hash_table(table_size);
    memset(hash_table.data(), 0, table_size * sizeof(HashEntry));

    size_t max_probe_length = 0;
    size_t total_probes = 0;
    std::vector<size_t> all_probe_lengths;
    all_probe_lengths.reserve(num_keys);

    // Build hash table
    for (size_t i = 0; i < num_keys; i++) {
        uint64_t pattern = patterns[i];
        uint16_t gen = generations[i];

        uint64_t hash = hash_func(pattern);
        size_t slot = hash & (table_size - 1);
        size_t probe_length = 0;

        while (hash_table[slot].pattern != 0) {
            slot = (slot + 1) & (table_size - 1);
            probe_length++;
        }

        hash_table[slot].pattern = pattern;
        hash_table[slot].generations = gen;

        max_probe_length = std::max(max_probe_length, probe_length);
        total_probes += probe_length;
        all_probe_lengths.push_back(probe_length);
    }

    double avg_probe_length = (double)total_probes / num_keys;

    // Calculate median and p90 probe length
    std::sort(all_probe_lengths.begin(), all_probe_lengths.end());
    size_t median_probe_length = all_probe_lengths[num_keys / 2];
    size_t p90_probe_length = all_probe_lengths[(num_keys * 9) / 10];

    // Verify
    bool verification_ok = true;
    for (size_t i = 0; i < std::min(num_keys, size_t(10000)); i++) {
        uint64_t pattern = patterns[i];
        uint64_t hash = hash_func(pattern);
        size_t slot = hash & (table_size - 1);

        bool found = false;
        for (size_t probe = 0; probe < max_probe_length + 10; probe++) {
            if (hash_table[slot].pattern == pattern) {
                found = true;
                break;
            }
            if (hash_table[slot].pattern == 0) break;
            slot = (slot + 1) & (table_size - 1);
        }

        if (!found) {
            verification_ok = false;
            break;
        }
    }

    // Print results
    std::cout << "\n" << name << ":\n";
    std::cout << "  Hash operations: " << ops_count << "\n";
    std::cout << "  Median probe length: " << median_probe_length << "\n";
    std::cout << "  P90 probe length: " << p90_probe_length << "\n";
    std::cout << "  Average probe length: " << avg_probe_length << "\n";
    std::cout << "  Maximum probe length: " << max_probe_length << "\n";
    std::cout << "  Verification: " << (verification_ok ? "PASSED" : "FAILED") << "\n";

    double total_ops_median = ops_count + 1 + (median_probe_length * 2);
    double total_ops_p90 = ops_count + 1 + (p90_probe_length * 2);
    double total_ops_avg = ops_count + 1 + (avg_probe_length * 2);
    std::cout << "  Total lookup cost (median): ~" << total_ops_median << " ops\n";
    std::cout << "  Total lookup cost (p90):    ~" << total_ops_p90 << " ops\n";
    std::cout << "  Total lookup cost (average): ~" << total_ops_avg << " ops\n";

    return {max_probe_length, avg_probe_length, median_probe_length, p90_probe_length, verification_ok};
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input-cache.json>\n";
        return 1;
    }

    std::string inputFile = argv[1];

    // Load patterns
    std::cout << "Loading patterns from " << inputFile << "...\n";
    std::vector<uint64_t> patterns;
    std::vector<uint16_t> generations;

    std::ifstream inFile(inputFile);
    if (!inFile.is_open()) {
        std::cerr << "Error: Failed to open file: " << inputFile << "\n";
        return 1;
    }

    std::string line;
    while (std::getline(inFile, line)) {
        if (line.empty()) continue;

        try {
            json entry = json::parse(line);
            uint64_t pattern = entry["pattern"];
            uint16_t gen = entry["generations"];
            patterns.push_back(pattern);
            generations.push_back(gen);
        } catch (const std::exception& e) {
            // Skip malformed lines
        }
    }
    inFile.close();

    size_t num_keys = patterns.size();
    std::cout << "Loaded " << num_keys << " patterns\n";
    std::cout << "\nComparing hash functions on your dataset...\n";
    std::cout << "==========================================================\n";

    // Test top 5 candidates
    std::cout << "\n=== TOP CANDIDATES (all 3 operations) ===\n";
    auto stats1 = build_and_test_hash(patterns, generations, hash_fibonacci_32, "Fibonacci * >>32", 3);
    auto stats2 = build_and_test_hash(patterns, generations, hash_splitmix_32, "Splitmix * >>32", 3);
    auto stats3 = build_and_test_hash(patterns, generations, hash_xxhash_32, "XXHash * >>32", 3);
    auto stats4 = build_and_test_hash(patterns, generations, hash_splitmix_33, "Splitmix * >>33", 3);
    auto stats5 = build_and_test_hash(patterns, generations, hash_cityhash_33, "CityHash * >>33", 3);

    // Test baseline
    std::cout << "\n=== BASELINE (6 operations) ===\n";
    auto stats6 = build_and_test_hash(patterns, generations, hash_splitmix64_full, "Splitmix64 Full", 6);

    std::cout << "\n==========================================================\n";
    std::cout << "SUMMARY:\n";
    std::cout << "  Total lookup cost = hash_ops + 1 (modulo) + avg_probes*2\n";
    std::cout << "  Lower total cost is better!\n";
    std::cout << "\n  Best performer: Fibonacci * >>32\n";
    std::cout << "  Constant: 0x9e3779b97f4a7c15ULL (Knuth's multiplicative hash)\n";

    return 0;
}
