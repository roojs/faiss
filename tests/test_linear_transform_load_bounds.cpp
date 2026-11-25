/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The linear-transform reader checks the matrix and bias lengths, and calls
// set_is_orthonormal, before it reads d_in and d_out. Those fields are still
// the constructor zeros, so any matrix passes and the orthonormal flag is
// forced on. The dimensions stored later are not checked against the matrix.
// reverse_transform then multiplies that matrix at the stored dimensions.
//
// These tests require read_index / read_VectorTransform to reject the payload
// before any reverse multiply. They fail while that gap is present.
//
// A fix that only repeats the early check (the one that runs while both
// dimensions are still zero) does not satisfy them: the multiply is still
// performed, and the guard-page tests die on the read past the allocation.
// The rejection has to happen on the dimensions the multiply uses, and it
// has to happen before that multiply.
//
// Operations that reach LinearTransform::reverse_transform from a transform
// loaded by read_VectorTransform:
//   IndexPreTransform::reconstruct
//   IndexPreTransform::reconstruct_n
//   IndexPreTransform::sa_decode
//   Index::reconstruct_batch and Index::compute_residual (both call
//     reconstruct, covered by the reconstruct test)
//   IndexLSH::sa_decode, and IndexLSH::reconstruct via IndexFlatCodes
//   VectorTransform::reverse_transform on a transform returned by
//     read_VectorTransform
//
// IndexPreTransform::search and search_and_reconstruct call apply_chain
// first. LinearTransform::apply_noalloc rejects a matrix whose length is not
// d_in * d_out before it multiplies, so those two operations do not read
// past the allocation. They are not covered here.
//
// Clustering and SuperKMeans call reverse_transform on matrices they build
// themselves, not on bytes from this reader.
//
// PCAMatrix, ITQMatrix, RandomRotationMatrix, and a generic LinearTransform
// (including OPQMatrix, which is written as "LTra") all share the same
// length check. The tests below drive "LTra" and "rrot".

#include <gtest/gtest.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexLSH.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <new>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr float kSentinel = 42.f;
constexpr int kDin = 4;
constexpr int kDout = 2;

enum class WatchMode { Off = 0, Sentinel = 1, Guard = 2 };

struct WatchedBlock {
    void* user = nullptr;
    void* map = nullptr;
    size_t bytes = 0;
    bool live = false;
};

constexpr int kMaxWatched = 64;
WatchedBlock g_blocks[kMaxWatched];
std::atomic<int> g_mode{static_cast<int>(WatchMode::Off)};
std::atomic<int> g_watched_count{0};
std::atomic_flag g_spin = ATOMIC_FLAG_INIT;

struct SpinLock {
    SpinLock() {
        while (g_spin.test_and_set(std::memory_order_acquire)) {
        }
    }
    ~SpinLock() {
        g_spin.clear(std::memory_order_release);
    }
};

void fill_sentinel(void* p, size_t bytes) {
    auto* f = static_cast<float*>(p);
    size_t n = bytes / sizeof(float);
    for (size_t i = 0; i < n; i++) {
        f[i] = kSentinel;
    }
}

void* watch_alloc(size_t n) {
    if (g_mode.load(std::memory_order_relaxed) ==
                static_cast<int>(WatchMode::Off) ||
        n != sizeof(float)) {
        return nullptr;
    }
    long page = sysconf(_SC_PAGESIZE);
    if (page < static_cast<long>(n)) {
        return nullptr;
    }
    SpinLock lock;
    int slot = -1;
    for (int i = 0; i < kMaxWatched; i++) {
        if (!g_blocks[i].live) {
            slot = i;
            break;
        }
    }
    if (slot < 0) {
        return nullptr;
    }
    int mode = g_mode.load(std::memory_order_relaxed);
    void* map = MAP_FAILED;
    size_t bytes = 0;
    void* user = nullptr;
    if (mode == static_cast<int>(WatchMode::Guard)) {
        bytes = static_cast<size_t>(page) * 2;
        map = mmap(
                nullptr,
                bytes,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0);
        if (map == MAP_FAILED) {
            return nullptr;
        }
        if (mprotect(
                    static_cast<char*>(map) + page,
                    static_cast<size_t>(page),
                    PROT_NONE) != 0) {
            munmap(map, bytes);
            return nullptr;
        }
        // The single float sits against the protected page, so the next
        // float the multiply reads is inaccessible.
        user = static_cast<char*>(map) + page - static_cast<long>(n);
    } else {
        bytes = static_cast<size_t>(page) * 32;
        map = mmap(
                nullptr,
                bytes,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0);
        if (map == MAP_FAILED) {
            return nullptr;
        }
        fill_sentinel(map, bytes);
        user = static_cast<char*>(map) + page;
    }
    g_blocks[slot] = WatchedBlock{user, map, bytes, true};
    g_watched_count.fetch_add(1, std::memory_order_relaxed);
    return user;
}

bool watch_free(void* p) {
    if (p == nullptr) {
        return false;
    }
    SpinLock lock;
    for (int i = 0; i < kMaxWatched; i++) {
        if (g_blocks[i].live && g_blocks[i].user == p) {
            munmap(g_blocks[i].map, g_blocks[i].bytes);
            g_blocks[i].live = false;
            return true;
        }
    }
    return false;
}

struct WatchScope {
    explicit WatchScope(WatchMode mode) {
        g_watched_count.store(0, std::memory_order_relaxed);
        g_mode.store(static_cast<int>(mode), std::memory_order_release);
    }
    ~WatchScope() {
        g_mode.store(
                static_cast<int>(WatchMode::Off), std::memory_order_release);
    }
    WatchScope(const WatchScope&) = delete;
    WatchScope& operator=(const WatchScope&) = delete;
};

std::vector<uint8_t> serialize_index(const faiss::Index* index) {
    faiss::VectorIOWriter writer;
    faiss::write_index(index, &writer);
    return writer.data;
}

size_t find_fourcc(const std::vector<uint8_t>& bytes, const char fcc[4]) {
    uint32_t h = faiss::fourcc(fcc);
    const auto* pat = reinterpret_cast<const uint8_t*>(&h);
    for (size_t i = 0; i + 4 <= bytes.size(); i++) {
        if (std::memcmp(bytes.data() + i, pat, 4) == 0) {
            return i;
        }
    }
    return static_cast<size_t>(-1);
}

// Replace the matrix that follows a linear-transform fourcc. The on-disk
// layout is the one write_VectorTransform uses: fourcc, have_bias, A, b,
// then d_in, d_out, is_trained.
void require(bool ok, const char* what) {
    if (!ok) {
        throw std::runtime_error(what);
    }
}

void rewrite_matrix(
        std::vector<uint8_t>& bytes,
        const char fcc[4],
        size_t old_n,
        size_t new_n,
        float first) {
    size_t off = find_fourcc(bytes, fcc);
    require(off != static_cast<size_t>(-1), "missing linear-transform fourcc");
    size_t size_pos = off + 4 + sizeof(bool);
    require(size_pos + sizeof(size_t) <= bytes.size(), "truncated matrix header");
    size_t n = 0;
    std::memcpy(&n, bytes.data() + size_pos, sizeof(n));
    require(n == old_n, "matrix length in the serialized index is not the fixture length");
    std::memcpy(bytes.data() + size_pos, &new_n, sizeof(new_n));
    size_t data = size_pos + sizeof(size_t);
    require(data + old_n * sizeof(float) <= bytes.size(), "truncated matrix");
    if (new_n > 0) {
        std::memcpy(bytes.data() + data, &first, sizeof(first));
    }
    bytes.erase(
            bytes.begin() + static_cast<std::ptrdiff_t>(data + new_n * sizeof(float)),
            bytes.begin() +
                    static_cast<std::ptrdiff_t>(data + old_n * sizeof(float)));
}

void rewrite_bias(
        std::vector<uint8_t>& bytes,
        const char fcc[4],
        size_t new_n,
        float first) {
    size_t off = find_fourcc(bytes, fcc);
    require(off != static_cast<size_t>(-1), "missing linear-transform fourcc");
    bytes[off + 4] = 1; // have_bias
    size_t a_size_pos = off + 4 + sizeof(bool);
    size_t a_n = 0;
    std::memcpy(&a_n, bytes.data() + a_size_pos, sizeof(a_n));
    size_t b_size_pos = a_size_pos + sizeof(size_t) + a_n * sizeof(float);
    require(b_size_pos + sizeof(size_t) <= bytes.size(), "truncated bias header");
    size_t b_n = 0;
    std::memcpy(&b_n, bytes.data() + b_size_pos, sizeof(b_n));
    require(b_n == 0u, "fixture bias is not empty");
    std::memcpy(bytes.data() + b_size_pos, &new_n, sizeof(new_n));
    if (new_n > 0) {
        std::vector<uint8_t> elem(sizeof(float));
        std::memcpy(elem.data(), &first, sizeof(first));
        bytes.insert(
                bytes.begin() +
                        static_cast<std::ptrdiff_t>(b_size_pos + sizeof(size_t)),
                elem.begin(),
                elem.end());
    }
}

std::unique_ptr<faiss::Index> make_pretransform(
        int d_in,
        int d_out,
        std::vector<float> matrix,
        std::vector<float> stored) {
    auto flat = std::make_unique<faiss::IndexFlatL2>(d_out);
    flat->add(1, stored.data());
    auto lt = std::make_unique<faiss::LinearTransform>(d_in, d_out, false);
    lt->A = std::move(matrix);
    lt->is_trained = true;
    // The serialized form does not store is_orthonormal. The reader computes
    // it, which is the behavior under test.
    auto pt = std::make_unique<faiss::IndexPreTransform>(lt.release(), flat.release());
    pt->own_fields = true;
    return pt;
}

// Two orthonormal columns in R^4, stored column-major with lda = d_in.
// Forward keeps the first two coordinates. Reverse of (1, 1) is (1, 1, 0, 0).
std::vector<float> orthonormal_4x2() {
    return {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f};
}

std::vector<uint8_t> pretransform_bytes(int d_in, int d_out, std::vector<float> matrix) {
    std::vector<float> stored(static_cast<size_t>(d_out), 1.f);
    auto index = make_pretransform(d_in, d_out, std::move(matrix), stored);
    return serialize_index(index.get());
}

std::vector<uint8_t> short_pretransform_bytes(int d_in, int d_out) {
    std::vector<float> matrix(
            static_cast<size_t>(d_in) * static_cast<size_t>(d_out), 0.f);
    auto bytes = pretransform_bytes(d_in, d_out, matrix);
    rewrite_matrix(bytes, "LTra", matrix.size(), 1, 0.f);
    return bytes;
}

bool message_rejects_matrix(const std::string& msg) {
    return msg.find("A.size") != std::string::npos ||
            msg.find("matrix") != std::string::npos;
}

bool message_rejects_bias(const std::string& msg) {
    return msg.find("b.size") != std::string::npos ||
            msg.find("bias") != std::string::npos;
}

std::string floats_of(const float* v, int n) {
    std::string s = "[";
    for (int i = 0; i < n; i++) {
        if (i) {
            s += ", ";
        }
        s += std::to_string(v[i]);
    }
    s += "]";
    return s;
}

// x = A * y, with A column-major, A[0] = stored and every other entry the
// sentinel that sits past the one-float allocation.
void leaked_matrix_product(
        int d_in,
        int d_out,
        const float* y,
        float stored,
        float* x) {
    for (int i = 0; i < d_in; i++) {
        float sum = 0.f;
        for (int p = 0; p < d_out; p++) {
            int idx = i + p * d_in;
            float a = idx == 0 ? stored : kSentinel;
            sum += a * y[p];
        }
        x[i] = sum;
    }
}

void expect_matrix_rejection(const std::string& msg) {
    EXPECT_TRUE(message_rejects_matrix(msg))
            << "rejection must name the matrix length, got: " << msg;
}

void expect_bias_rejection(const std::string& msg) {
    EXPECT_TRUE(message_rejects_bias(msg))
            << "rejection must name the bias length, got: " << msg;
}

// Run body in a child. Exit 0 means the loader threw the expected exception.
// A signal means a read past the allocation (or a null dereference) happened
// before the loader refused. Exit 2 means the multiply returned a vector.
void expect_child_rejects(
        const std::function<void()>& body,
        const std::function<bool(const std::string&)>& accept) {
    int fds[2];
    ASSERT_EQ(pipe(fds), 0);
    pid_t pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        close(fds[0]);
        int code = 4;
        try {
            body();
            code = 2;
        } catch (const faiss::FaissException& e) {
            std::string msg = e.what();
            if (accept(msg)) {
                code = 0;
            } else {
                (void)!write(fds[1], msg.data(), msg.size());
                code = 3;
            }
        } catch (const std::exception& e) {
            std::string msg = e.what();
            (void)!write(fds[1], msg.data(), msg.size());
            code = 4;
        }
        close(fds[1]);
        _exit(code);
    }
    close(fds[1]);
    std::string child_msg;
    char buf[1024];
    ssize_t nread = 0;
    while ((nread = read(fds[0], buf, sizeof(buf))) > 0) {
        child_msg.append(buf, buf + nread);
    }
    close(fds[0]);
    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    if (WIFSIGNALED(status)) {
        FAIL() << "read past the transform allocation (signal "
               << WTERMSIG(status)
               << ") before the loader rejected the index";
    }
    ASSERT_TRUE(WIFEXITED(status));
    int code = WEXITSTATUS(status);
    if (code == 0) {
        return;
    }
    if (code == 2) {
        FAIL() << "short transform was accepted and the reverse multiply returned";
    }
    FAIL() << "loader did not reject the transform (child exit " << code
           << "): " << child_msg;
}

} // namespace

void* operator new(std::size_t n) {
    if (void* p = watch_alloc(n)) {
        return p;
    }
    void* p = std::malloc(n);
    if (!p) {
        throw std::bad_alloc();
    }
    return p;
}

void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    try {
        return operator new(n);
    } catch (...) {
        return nullptr;
    }
}

void* operator new[](std::size_t n) {
    return operator new(n);
}

void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
    return operator new(n, std::nothrow);
}

void operator delete(void* p) noexcept {
    if (!watch_free(p)) {
        std::free(p);
    }
}

void operator delete(void* p, std::size_t) noexcept {
    operator delete(p);
}

void operator delete[](void* p) noexcept {
    operator delete(p);
}

void operator delete[](void* p, std::size_t) noexcept {
    operator delete(p);
}

void operator delete(void* p, const std::nothrow_t&) noexcept {
    operator delete(p);
}

void operator delete[](void* p, const std::nothrow_t&) noexcept {
    operator delete(p);
}

namespace {

void fail_if_disclosed(
        const float* got,
        const float* leaked,
        int n,
        const faiss::LinearTransform* lt) {
    bool same = true;
    for (int i = 0; i < n; i++) {
        if (got[i] != leaked[i]) {
            same = false;
            break;
        }
    }
    std::string detail = "read_index accepted a linear transform with A.size=" +
            std::to_string(lt ? lt->A.size() : 0) + " d_in=" +
            std::to_string(lt ? lt->d_in : -1) + " d_out=" +
            std::to_string(lt ? lt->d_out : -1) + " is_orthonormal=" +
            std::to_string(lt ? lt->is_orthonormal : 0) + " reconstructed=" +
            floats_of(got, n) + " neighbor_product=" + floats_of(leaked, n);
    if (same) {
        FAIL() << "neighbor floats were folded into the returned vector; "
               << detail;
    }
    FAIL() << detail;
}

faiss::LinearTransform* linear_of(faiss::Index& index) {
    auto* pt = dynamic_cast<faiss::IndexPreTransform*>(&index);
    EXPECT_NE(pt, nullptr);
    if (!pt || pt->chain.empty()) {
        return nullptr;
    }
    return dynamic_cast<faiss::LinearTransform*>(pt->chain[0]);
}

} // namespace

TEST(LinearTransformLoadBounds, ValidPreTransformRoundTrip) {
    auto bytes = pretransform_bytes(kDin, kDout, orthonormal_4x2());
    faiss::VectorIOReader reader;
    reader.data = bytes;
    auto index = faiss::read_index_up(&reader);
    ASSERT_NE(index, nullptr);
    ASSERT_EQ(index->d, kDin);
    std::vector<float> recons(kDin, -1.f);
    index->reconstruct(0, recons.data());
    EXPECT_FLOAT_EQ(recons[0], 1.f);
    EXPECT_FLOAT_EQ(recons[1], 1.f);
    EXPECT_FLOAT_EQ(recons[2], 0.f);
    EXPECT_FLOAT_EQ(recons[3], 0.f);
}

TEST(LinearTransformLoadBounds, ShortMatrixReconstructDisclosesNeighborFloats) {
    auto bytes = short_pretransform_bytes(kDin, kDout);
    WatchScope watch(WatchMode::Sentinel);
    faiss::VectorIOReader reader;
    reader.data = bytes;
    try {
        auto index = faiss::read_index_up(&reader);
        ASSERT_NE(index, nullptr);
        auto* lt = linear_of(*index);
        ASSERT_NE(lt, nullptr);
        float y[kDout] = {1.f, 1.f};
        float leaked[kDin];
        leaked_matrix_product(kDin, kDout, y, 0.f, leaked);
        std::vector<float> recons(kDin, -1.f);
        index->reconstruct(0, recons.data());
        fail_if_disclosed(recons.data(), leaked, kDin, lt);
    } catch (const faiss::FaissException& e) {
        expect_matrix_rejection(e.what());
    }
}

TEST(LinearTransformLoadBounds, ShortMatrixReconstructNDisclosesNeighborFloats) {
    auto bytes = short_pretransform_bytes(kDin, kDout);
    WatchScope watch(WatchMode::Sentinel);
    faiss::VectorIOReader reader;
    reader.data = bytes;
    try {
        auto index = faiss::read_index_up(&reader);
        float y[kDout] = {1.f, 1.f};
        float leaked[kDin];
        leaked_matrix_product(kDin, kDout, y, 0.f, leaked);
        std::vector<float> recons(kDin, -1.f);
        index->reconstruct_n(0, 1, recons.data());
        fail_if_disclosed(recons.data(), leaked, kDin, linear_of(*index));
    } catch (const faiss::FaissException& e) {
        expect_matrix_rejection(e.what());
    }
}

TEST(LinearTransformLoadBounds, ShortMatrixSaDecodeDisclosesNeighborFloats) {
    auto bytes = short_pretransform_bytes(kDin, kDout);
    WatchScope watch(WatchMode::Sentinel);
    faiss::VectorIOReader reader;
    reader.data = bytes;
    try {
        auto index = faiss::read_index_up(&reader);
        float stored[kDout] = {1.f, 1.f};
        float leaked[kDin];
        leaked_matrix_product(kDin, kDout, stored, 0.f, leaked);
        std::vector<float> decoded(kDin, -1.f);
        index->sa_decode(
                1, reinterpret_cast<const uint8_t*>(stored), decoded.data());
        fail_if_disclosed(decoded.data(), leaked, kDin, linear_of(*index));
    } catch (const faiss::FaissException& e) {
        expect_matrix_rejection(e.what());
    }
}

TEST(LinearTransformLoadBounds, ShortMatrixReverseTransformDisclosesNeighborFloats) {
    std::vector<float> matrix(
            static_cast<size_t>(kDin) * static_cast<size_t>(kDout), 0.f);
    faiss::LinearTransform lt(kDin, kDout, false);
    lt.A = matrix;
    lt.is_trained = true;
    faiss::VectorIOWriter writer;
    faiss::write_VectorTransform(&lt, &writer);
    auto bytes = writer.data;
    rewrite_matrix(bytes, "LTra", matrix.size(), 1, 0.f);

    WatchScope watch(WatchMode::Sentinel);
    faiss::VectorIOReader reader;
    reader.data = bytes;
    try {
        auto loaded = faiss::read_VectorTransform_up(&reader);
        auto* lin = dynamic_cast<faiss::LinearTransform*>(loaded.get());
        ASSERT_NE(lin, nullptr);
        float y[kDout] = {1.f, 1.f};
        float leaked[kDin];
        leaked_matrix_product(kDin, kDout, y, 0.f, leaked);
        float x[kDin];
        loaded->reverse_transform(1, y, x);
        fail_if_disclosed(x, leaked, kDin, lin);
    } catch (const faiss::FaissException& e) {
        expect_matrix_rejection(e.what());
    }
}

TEST(LinearTransformLoadBounds, ShortMatrixWideOutputDisclosesNeighborFloats) {
    // d_out > d_in. set_is_orthonormal would refuse this shape, but the
    // loader runs that check while both dimensions are still zero and leaves
    // the flag set.
    constexpr int d_in = 2;
    constexpr int d_out = 4;
    auto bytes = short_pretransform_bytes(d_in, d_out);
    WatchScope watch(WatchMode::Sentinel);
    faiss::VectorIOReader reader;
    reader.data = bytes;
    try {
        auto index = faiss::read_index_up(&reader);
        float y[d_out] = {1.f, 1.f, 1.f, 1.f};
        float leaked[d_in];
        leaked_matrix_product(d_in, d_out, y, 0.f, leaked);
        std::vector<float> recons(d_in, -1.f);
        index->reconstruct(0, recons.data());
        fail_if_disclosed(recons.data(), leaked, d_in, linear_of(*index));
    } catch (const faiss::FaissException& e) {
        expect_matrix_rejection(e.what());
    }
}

TEST(LinearTransformLoadBounds, ShortMatrixIndexLshDecodeDisclosesNeighborFloats) {
    faiss::IndexLSH lsh(kDin, kDout, true, false);
    ASSERT_EQ(lsh.rrot.A.size(), static_cast<size_t>(kDin * kDout));
    lsh.ntotal = 1;
    lsh.codes.resize(lsh.code_size);
    // Both stored bits set, so bitvecs2fvecs yields (+1, +1).
    lsh.codes[0] = 0x03;
    auto bytes = serialize_index(&lsh);
    rewrite_matrix(bytes, "rrot", lsh.rrot.A.size(), 1, 0.f);

    WatchScope watch(WatchMode::Sentinel);
    faiss::VectorIOReader reader;
    reader.data = bytes;
    try {
        auto index = faiss::read_index_up(&reader);
        auto* loaded = dynamic_cast<faiss::IndexLSH*>(index.get());
        ASSERT_NE(loaded, nullptr);
        uint8_t code = 0x03;
        float y[kDout] = {1.f, 1.f};
        float leaked[kDin];
        leaked_matrix_product(kDin, kDout, y, 0.f, leaked);
        std::vector<float> decoded(kDin, -1.f);
        index->sa_decode(1, &code, decoded.data());
        fail_if_disclosed(decoded.data(), leaked, kDin, &loaded->rrot);
    } catch (const faiss::FaissException& e) {
        expect_matrix_rejection(e.what());
    }
}

TEST(LinearTransformLoadBounds, ShortMatrixIndexLshReconstructDisclosesNeighborFloats) {
    faiss::IndexLSH lsh(kDin, kDout, true, false);
    lsh.ntotal = 1;
    lsh.codes.resize(lsh.code_size);
    lsh.codes[0] = 0x03;
    auto bytes = serialize_index(&lsh);
    rewrite_matrix(bytes, "rrot", static_cast<size_t>(kDin * kDout), 1, 0.f);

    WatchScope watch(WatchMode::Sentinel);
    faiss::VectorIOReader reader;
    reader.data = bytes;
    try {
        auto index = faiss::read_index_up(&reader);
        auto* loaded = dynamic_cast<faiss::IndexLSH*>(index.get());
        ASSERT_NE(loaded, nullptr);
        float y[kDout] = {1.f, 1.f};
        float leaked[kDin];
        leaked_matrix_product(kDin, kDout, y, 0.f, leaked);
        std::vector<float> recons(kDin, -1.f);
        index->reconstruct(0, recons.data());
        fail_if_disclosed(recons.data(), leaked, kDin, &loaded->rrot);
    } catch (const faiss::FaissException& e) {
        expect_matrix_rejection(e.what());
    }
}

TEST(LinearTransformLoadBounds, ShortBiasReconstructDisclosesNeighborFloats) {
    auto bytes = pretransform_bytes(kDin, kDout, orthonormal_4x2());
    rewrite_bias(bytes, "LTra", 1, 0.f);
    WatchScope watch(WatchMode::Sentinel);
    faiss::VectorIOReader reader;
    reader.data = bytes;
    try {
        auto index = faiss::read_index_up(&reader);
        std::vector<float> recons(kDin, -1.f);
        index->reconstruct(0, recons.data());
        // y = (1, 1), b[0] = 0, b[1] is the neighbor sentinel.
        // bias-corrected y is (1, 1 - sentinel). A maps that to
        // (1, 1 - sentinel, 0, 0).
        float leaked[kDin] = {1.f, 1.f - kSentinel, 0.f, 0.f};
        fail_if_disclosed(recons.data(), leaked, kDin, linear_of(*index));
    } catch (const faiss::FaissException& e) {
        expect_bias_rejection(e.what());
    }
}

TEST(LinearTransformLoadBounds, ShortMatrixReconstructGuardPage) {
    auto bytes = short_pretransform_bytes(kDin, kDout);
    expect_child_rejects(
            [&] {
                WatchScope watch(WatchMode::Guard);
                faiss::VectorIOReader reader;
                reader.data = bytes;
                auto index = faiss::read_index_up(&reader);
                std::vector<float> recons(kDin, -1.f);
                index->reconstruct(0, recons.data());
            },
            message_rejects_matrix);
}

TEST(LinearTransformLoadBounds, ShortMatrixSaDecodeGuardPage) {
    auto bytes = short_pretransform_bytes(kDin, kDout);
    expect_child_rejects(
            [&] {
                WatchScope watch(WatchMode::Guard);
                faiss::VectorIOReader reader;
                reader.data = bytes;
                auto index = faiss::read_index_up(&reader);
                float stored[kDout] = {1.f, 1.f};
                std::vector<float> decoded(kDin, -1.f);
                index->sa_decode(
                        1,
                        reinterpret_cast<const uint8_t*>(stored),
                        decoded.data());
            },
            message_rejects_matrix);
}

TEST(LinearTransformLoadBounds, ShortMatrixReverseTransformGuardPage) {
    std::vector<float> matrix(
            static_cast<size_t>(kDin) * static_cast<size_t>(kDout), 0.f);
    faiss::LinearTransform lt(kDin, kDout, false);
    lt.A = matrix;
    lt.is_trained = true;
    faiss::VectorIOWriter writer;
    faiss::write_VectorTransform(&lt, &writer);
    auto bytes = writer.data;
    rewrite_matrix(bytes, "LTra", matrix.size(), 1, 0.f);
    expect_child_rejects(
            [&] {
                WatchScope watch(WatchMode::Guard);
                faiss::VectorIOReader reader;
                reader.data = bytes;
                auto loaded = faiss::read_VectorTransform_up(&reader);
                float y[kDout] = {1.f, 1.f};
                float x[kDin];
                loaded->reverse_transform(1, y, x);
            },
            message_rejects_matrix);
}

TEST(LinearTransformLoadBounds, ShortBiasReconstructGuardPage) {
    auto bytes = pretransform_bytes(kDin, kDout, orthonormal_4x2());
    rewrite_bias(bytes, "LTra", 1, 0.f);
    expect_child_rejects(
            [&] {
                WatchScope watch(WatchMode::Guard);
                faiss::VectorIOReader reader;
                reader.data = bytes;
                auto index = faiss::read_index_up(&reader);
                std::vector<float> recons(kDin, -1.f);
                index->reconstruct(0, recons.data());
            },
            message_rejects_bias);
}

TEST(LinearTransformLoadBounds, EmptyBiasReconstructGuardPage) {
    auto bytes = pretransform_bytes(kDin, kDout, orthonormal_4x2());
    rewrite_bias(bytes, "LTra", 0, 0.f);
    expect_child_rejects(
            [&] {
                WatchScope watch(WatchMode::Guard);
                faiss::VectorIOReader reader;
                reader.data = bytes;
                auto index = faiss::read_index_up(&reader);
                std::vector<float> recons(kDin, -1.f);
                index->reconstruct(0, recons.data());
            },
            message_rejects_bias);
}

TEST(LinearTransformLoadBounds, EmptyMatrixReconstructDoesNotRun) {
    std::vector<float> matrix(
            static_cast<size_t>(kDin) * static_cast<size_t>(kDout), 0.f);
    auto bytes = pretransform_bytes(kDin, kDout, matrix);
    rewrite_matrix(bytes, "LTra", matrix.size(), 0, 0.f);
    expect_child_rejects(
            [&] {
                faiss::VectorIOReader reader;
                reader.data = bytes;
                auto index = faiss::read_index_up(&reader);
                std::vector<float> recons(kDin, -1.f);
                index->reconstruct(0, recons.data());
            },
            message_rejects_matrix);
}
