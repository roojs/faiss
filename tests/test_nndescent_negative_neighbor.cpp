/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Neighbor id -1 is in the range the NNDescent loader accepts, and search
// uses every neighbor id as a subscript of the visited table and then of the
// stored vectors. These tests load a graph that contains that id and search
// it. They pass only when the id is rejected while loading, or skipped before
// either use. A crash, including an out-of-bounds read of the stored vectors,
// fails the case.

#include <faiss/IndexFlat.h>
#include <faiss/IndexNNDescent.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/VisitedTable.h>
#include <faiss/impl/maybe_owned_vector.h>
#include <faiss/index_io.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <new>
#include <string>
#include <unistd.h>
#include <vector>

#include <sys/mman.h>
#include <sys/wait.h>

namespace {

// Allocations sit immediately after an unmapped page. Neighbor id -1 is
// converted to the maximum size_t, and a one-byte visited table then addresses
// the byte before its buffer. That byte is unmapped here, so the access
// faults. The same placement makes a distance read before the stored vectors
// fault when the visited table is a hash set and does not subscript.
constexpr size_t kGuardPage = 4096;
constexpr int kAllocSlots = 8192;

struct GuardBlock {
    void* user = nullptr;
    void* base = nullptr;
    size_t total = 0;
};

GuardBlock g_blocks[kAllocSlots];
std::atomic_flag g_alloc_lock = ATOMIC_FLAG_INIT;

struct AllocLock {
    AllocLock() {
        while (g_alloc_lock.test_and_set(std::memory_order_acquire)) {
        }
    }
    ~AllocLock() {
        g_alloc_lock.clear(std::memory_order_release);
    }
};

void* guard_alloc(size_t n) {
    if (n == 0) {
        n = 1;
    }
    const size_t payload = (n + kGuardPage - 1) & ~(kGuardPage - 1);
    if (payload < n) {
        throw std::bad_alloc();
    }
    const size_t total = kGuardPage + payload;
    void* base = mmap(
            nullptr,
            total,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0);
    if (base == MAP_FAILED) {
        throw std::bad_alloc();
    }
    if (mprotect(base, kGuardPage, PROT_NONE) != 0) {
        munmap(base, total);
        throw std::bad_alloc();
    }
    void* user = static_cast<uint8_t*>(base) + kGuardPage;
    AllocLock lock;
    for (GuardBlock& block : g_blocks) {
        if (block.user == nullptr) {
            block.user = user;
            block.base = base;
            block.total = total;
            return user;
        }
    }
    munmap(base, total);
    throw std::bad_alloc();
}

void guard_free(void* user) noexcept {
    if (user == nullptr) {
        return;
    }
    AllocLock lock;
    for (GuardBlock& block : g_blocks) {
        if (block.user == user) {
            munmap(block.base, block.total);
            block.user = nullptr;
            block.base = nullptr;
            block.total = 0;
            return;
        }
    }
    std::fprintf(stderr, "guard_free: pointer was not allocated here\n");
    std::abort();
}

} // namespace

void* operator new(std::size_t n) {
    return guard_alloc(n);
}
void* operator new[](std::size_t n) {
    return guard_alloc(n);
}
void* operator new(std::size_t n, std::align_val_t) {
    return guard_alloc(n);
}
void* operator new[](std::size_t n, std::align_val_t) {
    return guard_alloc(n);
}
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    try {
        return guard_alloc(n);
    } catch (...) {
        return nullptr;
    }
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
    try {
        return guard_alloc(n);
    } catch (...) {
        return nullptr;
    }
}
void operator delete(void* p) noexcept {
    guard_free(p);
}
void operator delete[](void* p) noexcept {
    guard_free(p);
}
void operator delete(void* p, std::size_t) noexcept {
    guard_free(p);
}
void operator delete[](void* p, std::size_t) noexcept {
    guard_free(p);
}
void operator delete(void* p, std::align_val_t) noexcept {
    guard_free(p);
}
void operator delete[](void* p, std::align_val_t) noexcept {
    guard_free(p);
}
void operator delete(void* p, std::size_t, std::align_val_t) noexcept {
    guard_free(p);
}
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept {
    guard_free(p);
}

namespace {

constexpr int kN = 8;
constexpr int kD = 4;
constexpr int kK = 4;

struct MmapOwner : faiss::MaybeOwnedVectorOwner {
    void* base = nullptr;
    size_t length = 0;

    MmapOwner(void* base_in, size_t length_in)
            : base(base_in), length(length_in) {}

    ~MmapOwner() override {
        if (base != nullptr && base != MAP_FAILED) {
            munmap(base, length);
        }
    }
};

// Place the loaded vectors immediately after an unmapped page. A neighbor id
// of -1 addresses code_size bytes before that buffer. Those bytes are now
// unmapped, so the distance read faults instead of landing in a heap header.
void seat_codes_after_guard(faiss::IndexFlat* storage) {
    const size_t bytes = storage->codes.size();
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) {
        std::cerr << "sysconf(_SC_PAGESIZE) failed\n";
        std::_Exit(1);
    }
    size_t guard = static_cast<size_t>(page);
    if (storage->code_size > guard) {
        const size_t page_size = static_cast<size_t>(page);
        guard = (storage->code_size + page_size - 1) & ~(page_size - 1);
    }
    size_t payload = bytes;
    const size_t page_size = static_cast<size_t>(page);
    if (payload == 0) {
        payload = page_size;
    } else {
        payload = (payload + page_size - 1) & ~(page_size - 1);
    }
    const size_t total = guard + payload;
    void* mem = mmap(
            nullptr,
            total,
            PROT_NONE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0);
    if (mem == MAP_FAILED) {
        std::perror("mmap");
        std::_Exit(1);
    }
    auto* data = static_cast<uint8_t*>(mem) + guard;
    if (mprotect(data, payload, PROT_READ | PROT_WRITE) != 0) {
        std::perror("mprotect");
        std::_Exit(1);
    }
    if (bytes > 0) {
        memcpy(data, storage->codes.data(), bytes);
    }
    auto owner = std::make_shared<MmapOwner>(mem, total);
    storage->codes = faiss::MaybeOwnedVector<uint8_t>::create_view(
            data, bytes, owner);
}

std::string temp_path() {
    char tmpl[] = "/tmp/faiss-nnd-neg-XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        std::perror("mkstemp");
        std::_Exit(1);
    }
    close(fd);
    return tmpl;
}

void write_index_file(
        const std::string& path,
        int poison_slot,
        int poison_id) {
    faiss::IndexNNDescentFlat index(kD, kK);
    std::vector<float> xb(static_cast<size_t>(kN) * kD);
    for (size_t i = 0; i < xb.size(); i++) {
        xb[i] = static_cast<float>(i + 1);
    }
    index.storage->add(kN, xb.data());
    index.ntotal = kN;
    index.is_trained = true;

    faiss::NNDescent& nnd = index.nndescent;
    nnd.ntotal = kN;
    nnd.d = kD;
    nnd.K = kK;
    nnd.has_built = true;
    nnd.search_L = 0;
    nnd.final_graph.assign(static_cast<size_t>(kN) * kK, 0);
    for (int i = 0; i < kN; i++) {
        for (int j = 0; j < kK; j++) {
            int id = (i + 1 + j) % kN;
            if (poison_slot >= 0 && j == poison_slot) {
                id = poison_id;
            }
            nnd.final_graph[static_cast<size_t>(i) * kK + j] = id;
        }
    }
    faiss::write_index(&index, path.c_str());
}

// 0: search returned only ids in [0, kN).
// 1: the loader or the search rejected the graph.
int load_and_search(const std::string& path, bool hashset) {
    std::unique_ptr<faiss::Index> index(faiss::read_index(path.c_str()));
    auto* nnd_index = dynamic_cast<faiss::IndexNNDescentFlat*>(index.get());
    if (nnd_index == nullptr || nnd_index->storage == nullptr) {
        std::cerr << "loaded index is not IndexNNDescentFlat with storage\n";
        std::_Exit(1);
    }
    auto* flat = dynamic_cast<faiss::IndexFlat*>(nnd_index->storage);
    if (flat == nullptr) {
        std::cerr << "storage is not IndexFlat\n";
        std::_Exit(1);
    }
    seat_codes_after_guard(flat);

    // IndexNNDescent::search calls VisitedTable::create(ntotal), which uses
    // the byte vector below this threshold and the hash set at or above it.
    faiss::visited_table_hashset_threshold = hashset ? 0 : 10000000;

    std::vector<float> query(static_cast<size_t>(kD), 0.25f);
    const faiss::idx_t k = 1;
    std::vector<float> distances(k);
    std::vector<faiss::idx_t> labels(k, -99);
    std::cerr << "searching\n" << std::flush;
    index->search(1, query.data(), k, distances.data(), labels.data());
    for (faiss::idx_t label : labels) {
        if (label < 0 || label >= kN) {
            std::cerr << "search returned label " << label << "\n";
            std::_Exit(1);
        }
    }
    return 0;
}

int run_case(
        const char* name,
        int poison_slot,
        int poison_id,
        bool hashset,
        bool require_load_rejection) {
    const std::string path = temp_path();
    try {
        write_index_file(path, poison_slot, poison_id);
        try {
            load_and_search(path, hashset);
            ::unlink(path.c_str());
            if (require_load_rejection) {
                std::cerr << name << ": loader accepted id " << poison_id
                          << "\n";
                return 1;
            }
            std::cerr << name << ": search completed\n";
            return 0;
        } catch (const faiss::FaissException& ex) {
            ::unlink(path.c_str());
            std::cerr << name << ": rejected: " << ex.what() << "\n";
            // A graph with no poisoned id must search. Rejection is a fix
            // only for a file that carries an out-of-range neighbor id.
            // The guarded vector buffer makes a distance read that happens
            // before the refusal kill the process instead of throwing.
            if (poison_slot < 0 && !require_load_rejection) {
                return 1;
            }
            return 0;
        }
    } catch (const faiss::FaissException& ex) {
        ::unlink(path.c_str());
        std::cerr << name << ": failed while building the index file: "
                  << ex.what() << "\n";
        return 1;
    }
}

struct Case {
    const char* name;
    int poison_slot;
    int poison_id;
    bool hashset;
    bool require_load_rejection;
};

const Case kCases[] = {
        {"valid_graph", -1, 0, false, false},
        {"vector_table_negative_id_first_slot", 0, -1, false, false},
        {"vector_table_negative_id_last_slot", kK - 1, -1, false, false},
        {"hashset_table_negative_id", 0, -1, true, false},
        {"loader_rejects_id_below_minus_one", 0, -2, false, true},
};

int run_driver(const char* argv0) {
    int failed = 0;
    for (const Case& test_case : kCases) {
        pid_t pid = fork();
        if (pid < 0) {
            std::perror("fork");
            return 1;
        }
        if (pid == 0) {
            execl(
                    "/proc/self/exe",
                    argv0,
                    "--case",
                    test_case.name,
                    static_cast<char*>(nullptr));
            std::perror("execl");
            std::_Exit(127);
        }
        int status = 0;
        if (waitpid(pid, &status, 0) < 0) {
            std::perror("waitpid");
            return 1;
        }
        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
            std::cout << "PASS " << test_case.name << "\n";
        } else if (WIFSIGNALED(status)) {
            std::cout << "FAIL " << test_case.name << " killed by signal "
                      << WTERMSIG(status) << "\n";
            failed++;
        } else {
            std::cout << "FAIL " << test_case.name << " exit "
                      << (WIFEXITED(status) ? WEXITSTATUS(status) : -1)
                      << "\n";
            failed++;
        }
    }
    return failed == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "--case") == 0) {
        for (const Case& test_case : kCases) {
            if (std::strcmp(argv[2], test_case.name) == 0) {
                return run_case(
                        test_case.name,
                        test_case.poison_slot,
                        test_case.poison_id,
                        test_case.hashset,
                        test_case.require_load_rejection);
            }
        }
        std::cerr << "unknown case " << argv[2] << "\n";
        return 1;
    }
    return run_driver(argv[0]);
}
