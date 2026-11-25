/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Spec for the inverted-list deserialization loop limit.
//
// The default of deserialization_loop_limit is 0, and the check is skipped
// when the limit is 0. Array, panorama, and block inverted-list readers then
// resize per-list tables to the file-supplied count before any list body is
// read. The same default lets the legacy IVF id table, the skip-IVF size
// table, the NSG graph, and IndexBinaryMultiHash::maps resize from a count in
// the file before the payload that count describes.
//
// These tests fail while that behavior is present. A fixed loader rejects
// kAttackCount at the default limit, with a deserialization_loop_limit error,
// before those tables exist. The child address cap is only VmSize + 128MiB, so
// a resize of kAttackCount still fails the test when the check is missing or
// runs after the resize. Callers that need a larger index raise the limit with
// set_deserialization_loop_limit.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <faiss/IndexBinary.h>
#include <faiss/IndexBinaryIVF.h>
#include <faiss/IndexIVF.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>
#include <faiss/invlists/InvertedLists.h>

namespace {

// Hundreds of millions of lists. Per-list tables are one container entry each
// (24 bytes for std::vector, ~80 bytes for MaybeOwnedVector) plus an 8-byte
// size, so this count is several gigabytes and cannot fit in the child cap.
constexpr size_t kAttackCount = 200000000;
constexpr size_t kChildSlackBytes = 128ull << 20;

// Captured during static initialization, before any test changes the limit.
const size_t kStartupLoopLimit = faiss::get_deserialization_loop_limit();

struct RestoreLoopLimit {
    size_t previous = faiss::get_deserialization_loop_limit();
    ~RestoreLoopLimit() {
        faiss::set_deserialization_loop_limit(previous);
    }
};

template <typename T>
void push_val(std::vector<uint8_t>& buf, T val) {
    const auto* p = reinterpret_cast<const uint8_t*>(&val);
    buf.insert(buf.end(), p, p + sizeof(T));
}

void push_fourcc(std::vector<uint8_t>& buf, const char s[4]) {
    const auto* x = reinterpret_cast<const unsigned char*>(s);
    uint32_t h = x[0] | (x[1] << 8) | (x[2] << 16) | (x[3] << 24);
    push_val<uint32_t>(buf, h);
}

void push_index_header(
        std::vector<uint8_t>& buf,
        int d,
        int64_t ntotal,
        bool is_trained = true,
        int metric_type = 1 /* L2 */) {
    push_val<int>(buf, d);
    push_val<int64_t>(buf, ntotal);
    int64_t dummy = 1 << 20;
    push_val<int64_t>(buf, dummy);
    push_val<int64_t>(buf, dummy);
    push_val<uint8_t>(buf, is_trained ? 1 : 0);
    push_val<int>(buf, metric_type);
}

void push_binary_header(
        std::vector<uint8_t>& buf,
        int d,
        int64_t ntotal,
        bool is_trained = true,
        int metric_type = 1 /* L2 */) {
    push_val<int>(buf, d);
    push_val<int>(buf, d / 8); // code_size
    push_val<int64_t>(buf, ntotal);
    push_val<uint8_t>(buf, is_trained ? 1 : 0);
    push_val<int>(buf, metric_type);
}

void push_empty_vector(std::vector<uint8_t>& buf) {
    push_val<size_t>(buf, 0);
}

void push_empty_flat_quantizer(std::vector<uint8_t>& buf, int d) {
    push_fourcc(buf, "IxF2");
    push_index_header(buf, d, /*ntotal=*/0);
    push_empty_vector(buf); // xb, stored as size/4
}

void push_empty_binary_flat(std::vector<uint8_t>& buf, int d) {
    push_fourcc(buf, "IBxF");
    push_binary_header(buf, d, /*ntotal=*/0);
    push_empty_vector(buf);
}

void push_direct_map_none(std::vector<uint8_t>& buf) {
    push_val<uint8_t>(buf, 0); // DirectMap::NoMap
    push_empty_vector(buf);
}

// Complete sparse inverted lists with every list empty. A few dozen bytes
// regardless of nlist: the per-list tables are not in the file.
void push_ilar(std::vector<uint8_t>& buf, size_t nlist, size_t code_size) {
    push_fourcc(buf, "ilar");
    push_val<size_t>(buf, nlist);
    push_val<size_t>(buf, code_size);
    push_fourcc(buf, "sprs");
    push_empty_vector(buf);
}

void push_ilpn(std::vector<uint8_t>& buf, size_t nlist) {
    push_fourcc(buf, "ilpn");
    push_val<size_t>(buf, nlist);
    push_val<size_t>(buf, 4); // code_size, multiple of sizeof(float)
    push_val<size_t>(buf, 1); // n_levels
    push_fourcc(buf, "sprs");
    push_empty_vector(buf);
}

void push_ilp2(std::vector<uint8_t>& buf, size_t nlist) {
    push_fourcc(buf, "ilp2");
    push_val<size_t>(buf, nlist);
    push_val<size_t>(buf, 4);
    push_val<size_t>(buf, 1); // n_levels
    push_val<size_t>(buf, 1); // batch_size
    push_fourcc(buf, "sprs");
    push_empty_vector(buf);
}

// Block lists have no sparse size section. n_per_block and block_size are
// positive so the only large allocation is the per-list tables. List bodies
// are omitted: they are read after ids.resize / codes.resize.
void push_ilbl_header(std::vector<uint8_t>& buf, size_t nlist) {
    push_fourcc(buf, "ilbl");
    push_val<size_t>(buf, nlist);
    push_val<size_t>(buf, 1); // code_size
    push_val<size_t>(buf, 1); // n_per_block
    push_val<size_t>(buf, 1); // block_size
}

void push_empty_ilbl_lists(std::vector<uint8_t>& buf, size_t nlist) {
    for (size_t i = 0; i < nlist; i++) {
        push_empty_vector(buf); // ids
        push_empty_vector(buf); // codes
    }
}

void push_ivf_prefix(
        std::vector<uint8_t>& buf,
        const char fourcc[4],
        size_t nlist,
        int d) {
    push_fourcc(buf, fourcc);
    push_index_header(buf, d, /*ntotal=*/0);
    push_val<size_t>(buf, nlist);
    push_val<size_t>(buf, 1); // nprobe
    push_empty_flat_quantizer(buf, d);
}

void push_iwfl(
        std::vector<uint8_t>& buf,
        size_t header_nlist,
        size_t ilar_nlist,
        int d) {
    push_ivf_prefix(buf, "IwFl", header_nlist, d);
    push_direct_map_none(buf);
    push_ilar(buf, ilar_nlist, static_cast<size_t>(d) * sizeof(float));
}

void push_binary_ivf(
        std::vector<uint8_t>& buf,
        size_t header_nlist,
        size_t ilar_nlist,
        int d) {
    push_fourcc(buf, "IBwF");
    push_binary_header(buf, d, /*ntotal=*/0);
    push_val<size_t>(buf, header_nlist);
    push_val<size_t>(buf, 1); // nprobe
    push_empty_binary_flat(buf, d);
    push_direct_map_none(buf);
    push_ilar(buf, ilar_nlist, static_cast<size_t>(d / 8));
}

// Legacy IvFl resizes the id table from the header nlist before reading ids.
// A complete small file has one empty id vector, an empty direct map, and one
// empty code vector. The attack file stops after the quantizer.
void push_legacy_ivfl_prefix(std::vector<uint8_t>& buf, size_t nlist, int d) {
    push_ivf_prefix(buf, "IvFl", nlist, d);
}

void push_legacy_ivfl_tail(std::vector<uint8_t>& buf) {
    push_empty_vector(buf); // one list of ids
    push_direct_map_none(buf);
    push_empty_vector(buf); // one list of codes, as floats
}

void push_nsg(std::vector<uint8_t>& buf, int ntotal, bool with_body) {
    push_fourcc(buf, "INSf");
    push_index_header(buf, /*d=*/4, with_body ? ntotal : 0);
    push_val<int>(buf, 1);          // GK
    push_val<uint8_t>(buf, 0);      // build_type
    push_val<int>(buf, 1);          // nndescent_S
    push_val<int>(buf, 1);          // nndescent_R
    push_val<int>(buf, 1);          // nndescent_L
    push_val<int>(buf, 1);          // nndescent_iter
    push_val<int>(buf, ntotal);     // nsg.ntotal
    push_val<int>(buf, 1);          // R
    push_val<int>(buf, 1);          // L
    push_val<int>(buf, 1);          // C
    push_val<int>(buf, 1);          // search_L
    push_val<int>(buf, 0);          // enterpoint
    push_val<uint8_t>(buf, 1);      // is_built
    if (!with_body) {
        return;
    }
    push_val<int>(buf, -1); // empty neighbor list
    push_fourcc(buf, "IxF2");
    push_index_header(buf, 4, ntotal);
    // IndexFlat stores xb with READXBVECTOR: the size field counts groups of
    // 4 bytes, and the payload is ntotal * d floats.
    push_val<size_t>(buf, static_cast<size_t>(ntotal) * 4);
    for (int i = 0; i < ntotal * 4; i++) {
        push_val<float>(buf, 0.0f);
    }
}

void push_binary_multi_hash(
        std::vector<uint8_t>& buf,
        int d,
        int nhash,
        bool with_map) {
    push_fourcc(buf, "IBHm");
    push_binary_header(buf, d, /*ntotal=*/0);
    push_empty_binary_flat(buf, d);
    push_val<int>(buf, 1); // b
    push_val<int>(buf, nhash);
    push_val<int>(buf, 0); // nflip
    if (!with_map) {
        return;
    }
    push_val<int>(buf, 1); // id_bits
    push_val<size_t>(buf, 0);
    push_empty_vector(buf);
}

enum class LoadKind { InvertedLists, Index, IndexBinary, TouchMemory };

size_t current_vmsize_bytes() {
    std::ifstream in("/proc/self/status");
    std::string line;
    while (std::getline(in, line)) {
        if (line.compare(0, 7, "VmSize:") != 0) {
            continue;
        }
        size_t kb = 0;
        if (std::sscanf(line.c_str(), "VmSize: %zu", &kb) == 1) {
            return kb * 1024;
        }
    }
    return 0;
}

// Returns:
//   0  FaissException naming deserialization_loop_limit
//   2  std::bad_alloc or std::length_error (the count was used to allocate)
//   3  loader returned an object
//   5  address cap could not be installed
//   6  FaissException that is not the loop limit
//   7  modest allocation under the cap succeeded (TouchMemory)
//   100+signal  child killed
int load_under_address_cap(
        const std::vector<uint8_t>& payload,
        int io_flags,
        LoadKind kind) {
    pid_t pid = fork();
    if (pid < 0) {
        return 5;
    }
    if (pid == 0) {
        size_t vm = current_vmsize_bytes();
        if (vm == 0) {
            _exit(5);
        }
        struct rlimit rl;
        if (getrlimit(RLIMIT_AS, &rl) != 0) {
            _exit(5);
        }
        rl.rlim_cur = vm + kChildSlackBytes;
        if (rl.rlim_max != RLIM_INFINITY && rl.rlim_cur > rl.rlim_max) {
            _exit(5);
        }
        if (setrlimit(RLIMIT_AS, &rl) != 0) {
            _exit(5);
        }
        struct rlimit core;
        core.rlim_cur = 0;
        core.rlim_max = 0;
        setrlimit(RLIMIT_CORE, &core);
        try {
            if (kind == LoadKind::TouchMemory) {
                std::vector<uint8_t> block(32ull << 20, 1);
                if (block.size() != (32ull << 20) || block.back() != 1) {
                    _exit(2);
                }
                _exit(7);
            }
            faiss::VectorIOReader reader;
            reader.data = payload;
            if (kind == LoadKind::InvertedLists) {
                std::unique_ptr<faiss::InvertedLists> ils(
                        faiss::read_InvertedLists(&reader, io_flags));
                _exit(ils ? 3 : 6);
            } else if (kind == LoadKind::Index) {
                std::unique_ptr<faiss::Index> idx(
                        faiss::read_index(&reader, io_flags));
                _exit(idx ? 3 : 6);
            } else {
                std::unique_ptr<faiss::IndexBinary> idx(
                        faiss::read_index_binary(&reader, io_flags));
                _exit(idx ? 3 : 6);
            }
        } catch (const std::bad_alloc&) {
            _exit(2);
        } catch (const std::length_error&) {
            _exit(2);
        } catch (const faiss::FaissException& ex) {
            const char* what = ex.what();
            if (what != nullptr &&
                std::strstr(what, "deserialization_loop_limit") != nullptr) {
                _exit(0);
            }
            std::fprintf(
                    stderr,
                    "rejected without deserialization_loop_limit: %s\n",
                    what != nullptr ? what : "");
            std::fflush(stderr);
            _exit(6);
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "unexpected exception: %s\n", ex.what());
            std::fflush(stderr);
            _exit(6);
        } catch (...) {
            _exit(6);
        }
        _exit(6);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        return 5;
    }
    if (WIFSIGNALED(status)) {
        return 100 + WTERMSIG(status);
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    return 5;
}

void expect_loop_limit_rejection(
        const std::vector<uint8_t>& payload,
        int io_flags,
        LoadKind kind) {
    faiss::set_deserialization_loop_limit(kStartupLoopLimit);
    int code = load_under_address_cap(payload, io_flags, kind);
    EXPECT_EQ(code, 0)
            << "default loop limit must reject this file before the "
               "file-supplied count is used to resize. "
            << "0=loop-limit FaissException, 2=allocation attempted, "
            << "3=loaded, 6=other rejection, 100+signal=killed, got " << code;
}

template <typename T>
std::unique_ptr<T> read_up(const std::vector<uint8_t>& payload, int io_flags);

template <>
std::unique_ptr<faiss::InvertedLists> read_up(
        const std::vector<uint8_t>& payload,
        int io_flags) {
    faiss::VectorIOReader reader;
    reader.data = payload;
    return std::unique_ptr<faiss::InvertedLists>(
            faiss::read_InvertedLists(&reader, io_flags));
}

template <>
std::unique_ptr<faiss::Index> read_up(
        const std::vector<uint8_t>& payload,
        int io_flags) {
    faiss::VectorIOReader reader;
    reader.data = payload;
    return std::unique_ptr<faiss::Index>(faiss::read_index(&reader, io_flags));
}

template <>
std::unique_ptr<faiss::IndexBinary> read_up(
        const std::vector<uint8_t>& payload,
        int io_flags) {
    faiss::VectorIOReader reader;
    reader.data = payload;
    return std::unique_ptr<faiss::IndexBinary>(
            faiss::read_index_binary(&reader, io_flags));
}

} // namespace

TEST(DeserializationLoopLimitHarness, CapAllowsModestAllocation) {
    int code = load_under_address_cap({}, 0, LoadKind::TouchMemory);
    ASSERT_EQ(code, 7) << "address cap rejected a 32MiB allocation, got "
                       << code;
}

TEST(DeserializationLoopLimit, StartupLimitIsWhatTheAttackUses) {
    // The attack tests restore this value. It is the process default.
    EXPECT_EQ(kStartupLoopLimit, faiss::get_deserialization_loop_limit());
}

TEST(DeserializationLoopLimit, SmallIlarLoads) {
    RestoreLoopLimit restore;
    faiss::set_deserialization_loop_limit(kStartupLoopLimit);
    std::vector<uint8_t> payload;
    push_ilar(payload, 4, 1);
    auto ils = read_up<faiss::InvertedLists>(payload, 0);
    ASSERT_NE(ils, nullptr);
    EXPECT_EQ(ils->nlist, 4u);
    EXPECT_EQ(ils->list_size(0), 0u);
}

TEST(DeserializationLoopLimit, HugeIlarRejectedBeforePerListTables) {
    std::vector<uint8_t> payload;
    push_ilar(payload, kAttackCount, 1);
    EXPECT_LT(payload.size(), 64u);
    expect_loop_limit_rejection(payload, 0, LoadKind::InvertedLists);
}

TEST(DeserializationLoopLimit, SmallIlblLoads) {
    RestoreLoopLimit restore;
    faiss::set_deserialization_loop_limit(kStartupLoopLimit);
    std::vector<uint8_t> payload;
    push_ilbl_header(payload, 2);
    push_empty_ilbl_lists(payload, 2);
    auto ils = read_up<faiss::InvertedLists>(payload, 0);
    ASSERT_NE(ils, nullptr);
    EXPECT_EQ(ils->nlist, 2u);
    EXPECT_EQ(ils->list_size(1), 0u);
}

TEST(DeserializationLoopLimit, HugeIlblRejectedBeforePerListTables) {
    std::vector<uint8_t> payload;
    push_ilbl_header(payload, kAttackCount);
    EXPECT_LT(payload.size(), 64u);
    expect_loop_limit_rejection(payload, 0, LoadKind::InvertedLists);
}

TEST(DeserializationLoopLimit, SmallIlpnLoads) {
    RestoreLoopLimit restore;
    faiss::set_deserialization_loop_limit(kStartupLoopLimit);
    std::vector<uint8_t> payload;
    push_ilpn(payload, 3);
    auto ils = read_up<faiss::InvertedLists>(payload, 0);
    ASSERT_NE(ils, nullptr);
    EXPECT_EQ(ils->nlist, 3u);
}

TEST(DeserializationLoopLimit, HugeIlpnRejectedBeforePerListTables) {
    std::vector<uint8_t> payload;
    push_ilpn(payload, kAttackCount);
    EXPECT_LT(payload.size(), 64u);
    expect_loop_limit_rejection(payload, 0, LoadKind::InvertedLists);
}

TEST(DeserializationLoopLimit, SmallIlp2Loads) {
    RestoreLoopLimit restore;
    faiss::set_deserialization_loop_limit(kStartupLoopLimit);
    std::vector<uint8_t> payload;
    push_ilp2(payload, 3);
    auto ils = read_up<faiss::InvertedLists>(payload, 0);
    ASSERT_NE(ils, nullptr);
    EXPECT_EQ(ils->nlist, 3u);
}

TEST(DeserializationLoopLimit, HugeIlp2RejectedBeforePerListTables) {
    std::vector<uint8_t> payload;
    push_ilp2(payload, kAttackCount);
    EXPECT_LT(payload.size(), 64u);
    expect_loop_limit_rejection(payload, 0, LoadKind::InvertedLists);
}

TEST(DeserializationLoopLimit, SmallIwFlLoads) {
    RestoreLoopLimit restore;
    faiss::set_deserialization_loop_limit(kStartupLoopLimit);
    std::vector<uint8_t> payload;
    push_iwfl(payload, 4, 4, 4);
    auto idx = read_up<faiss::Index>(payload, 0);
    auto* ivf = dynamic_cast<faiss::IndexIVF*>(idx.get());
    ASSERT_NE(ivf, nullptr);
    EXPECT_EQ(ivf->nlist, 4u);
    ASSERT_NE(ivf->invlists, nullptr);
    EXPECT_EQ(ivf->invlists->nlist, 4u);
}

TEST(DeserializationLoopLimit, HugeIwFlRejectedBeforePerListTables) {
    std::vector<uint8_t> payload;
    push_iwfl(payload, kAttackCount, kAttackCount, 4);
    EXPECT_LT(payload.size(), 160u);
    expect_loop_limit_rejection(payload, 0, LoadKind::Index);
}

// Header nlist is small. The ilar section carries the attack count. A check
// of only the header field still resizes the per-list tables.
TEST(DeserializationLoopLimit, IwFlMismatchedIlarNlistRejectedBeforeResize) {
    std::vector<uint8_t> payload;
    push_iwfl(payload, /*header_nlist=*/1, kAttackCount, 4);
    EXPECT_LT(payload.size(), 160u);
    expect_loop_limit_rejection(payload, 0, LoadKind::Index);
}

TEST(DeserializationLoopLimit, SmallBinaryIvfLoads) {
    RestoreLoopLimit restore;
    faiss::set_deserialization_loop_limit(kStartupLoopLimit);
    std::vector<uint8_t> payload;
    push_binary_ivf(payload, 2, 2, 8);
    auto idx = read_up<faiss::IndexBinary>(payload, 0);
    auto* ivf = dynamic_cast<faiss::IndexBinaryIVF*>(idx.get());
    ASSERT_NE(ivf, nullptr);
    EXPECT_EQ(ivf->nlist, 2u);
    ASSERT_NE(ivf->invlists, nullptr);
    EXPECT_EQ(ivf->invlists->nlist, 2u);
}

TEST(DeserializationLoopLimit, HugeBinaryIvfRejectedBeforePerListTables) {
    std::vector<uint8_t> payload;
    push_binary_ivf(payload, kAttackCount, kAttackCount, 8);
    EXPECT_LT(payload.size(), 128u);
    expect_loop_limit_rejection(payload, 0, LoadKind::IndexBinary);
}

TEST(DeserializationLoopLimit, BinaryIvfMismatchedIlarNlistRejectedBeforeResize) {
    std::vector<uint8_t> payload;
    push_binary_ivf(payload, /*header_nlist=*/1, kAttackCount, 8);
    EXPECT_LT(payload.size(), 128u);
    expect_loop_limit_rejection(payload, 0, LoadKind::IndexBinary);
}

TEST(DeserializationLoopLimit, SmallLegacyIvFlLoads) {
    RestoreLoopLimit restore;
    faiss::set_deserialization_loop_limit(kStartupLoopLimit);
    std::vector<uint8_t> payload;
    push_legacy_ivfl_prefix(payload, 1, 4);
    push_legacy_ivfl_tail(payload);
    auto idx = read_up<faiss::Index>(payload, 0);
    auto* ivf = dynamic_cast<faiss::IndexIVF*>(idx.get());
    ASSERT_NE(ivf, nullptr);
    EXPECT_EQ(ivf->nlist, 1u);
    ASSERT_NE(ivf->invlists, nullptr);
    EXPECT_EQ(ivf->invlists->list_size(0), 0u);
}

TEST(DeserializationLoopLimit, HugeLegacyIvFlRejectedBeforeIdTable) {
    std::vector<uint8_t> payload;
    push_legacy_ivfl_prefix(payload, kAttackCount, 4);
    EXPECT_LT(payload.size(), 128u);
    expect_loop_limit_rejection(payload, 0, LoadKind::Index);
}

TEST(DeserializationLoopLimit, SkipIvfDataSmallReachesHookAfterSizeTable) {
    RestoreLoopLimit restore;
    faiss::set_deserialization_loop_limit(kStartupLoopLimit);
    std::vector<uint8_t> payload;
    push_ilar(payload, 4, 1);
    try {
        auto ils = read_up<faiss::InvertedLists>(
                payload, faiss::IO_FLAG_SKIP_IVF_DATA);
        FAIL() << "skip path returned an inverted list";
        static_cast<void>(ils);
    } catch (const faiss::FaissException& ex) {
        // sizes(nlist) and the sparse size section were consumed. The hook
        // id in the low flag bits is not registered.
        EXPECT_NE(
                std::string(ex.what()).find("could not load"),
                std::string::npos)
                << ex.what();
    }
}

TEST(DeserializationLoopLimit, SkipIvfDataHugeNlistRejectedBeforeSizeTable) {
    std::vector<uint8_t> payload;
    push_ilar(payload, kAttackCount, 1);
    EXPECT_LT(payload.size(), 64u);
    expect_loop_limit_rejection(
            payload, faiss::IO_FLAG_SKIP_IVF_DATA, LoadKind::InvertedLists);
}

TEST(DeserializationLoopLimit, SmallNsgLoads) {
    RestoreLoopLimit restore;
    faiss::set_deserialization_loop_limit(kStartupLoopLimit);
    std::vector<uint8_t> payload;
    push_nsg(payload, 1, true);
    auto idx = read_up<faiss::Index>(payload, 0);
    ASSERT_NE(idx, nullptr);
    EXPECT_EQ(idx->ntotal, 1);
}

TEST(DeserializationLoopLimit, HugeNsgRejectedBeforeGraphAllocation) {
    std::vector<uint8_t> payload;
    push_nsg(payload, static_cast<int>(kAttackCount), false);
    EXPECT_LT(payload.size(), 128u);
    expect_loop_limit_rejection(payload, 0, LoadKind::Index);
}

TEST(DeserializationLoopLimit, SmallBinaryMultiHashLoads) {
    RestoreLoopLimit restore;
    faiss::set_deserialization_loop_limit(kStartupLoopLimit);
    std::vector<uint8_t> payload;
    push_binary_multi_hash(payload, /*d=*/8, /*nhash=*/1, true);
    auto idx = read_up<faiss::IndexBinary>(payload, 0);
    ASSERT_NE(idx, nullptr);
    EXPECT_EQ(idx->d, 8);
}

TEST(DeserializationLoopLimit, HugeBinaryMultiHashRejectedBeforeMapTable) {
    // nhash * b <= d, with b = 1. d must be a multiple of 8.
    ASSERT_EQ(kAttackCount % 8, 0u);
    std::vector<uint8_t> payload;
    push_binary_multi_hash(
            payload,
            static_cast<int>(kAttackCount),
            static_cast<int>(kAttackCount),
            false);
    EXPECT_LT(payload.size(), 128u);
    expect_loop_limit_rejection(payload, 0, LoadKind::IndexBinary);
}

TEST(DeserializationLoopLimit, ExplicitLimitAllowsThatCount) {
    RestoreLoopLimit restore;
    faiss::set_deserialization_loop_limit(8);
    std::vector<uint8_t> payload;
    push_ilar(payload, 8, 1);
    auto ils = read_up<faiss::InvertedLists>(payload, 0);
    ASSERT_NE(ils, nullptr);
    EXPECT_EQ(ils->nlist, 8u);
}

TEST(DeserializationLoopLimit, ExplicitLimitRejectsOnePastWithoutResize) {
    RestoreLoopLimit restore;
    faiss::set_deserialization_loop_limit(8);
    std::vector<uint8_t> payload;
    push_ilar(payload, 9, 1);
    // Nine empty lists would fit in the child cap, so a check that resizes
    // first and then throws would still return 0 only if it did not resize
    // to a larger second count. The file has a single nlist. Rejection must
    // be the loop limit, and the child must not die allocating.
    int code = load_under_address_cap(payload, 0, LoadKind::InvertedLists);
    EXPECT_EQ(code, 0) << "got " << code;
}
