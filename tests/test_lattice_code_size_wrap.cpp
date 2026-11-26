/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Spec for the lattice code-size wrap.
//
// IndexLattice stores (lattice_nbit + scale_nbit) * nsq in a signed 32-bit
// integer. A scale width of 1431655764 with 2 lattice bits and 3 subspaces
// wraps that product to 2, so code_size becomes 1. The loader then only
// rejects a code_size larger than the raw vector, and sa_encode / sa_decode
// still walk the unwrapped bit width. These tests fail while that happens:
// the guarded calls fault (or AddressSanitizer reports a heap overflow), and
// an accepted index that finishes the call is a failure too. They pass once
// read_index and index_factory reject the width and the bit reader/writer
// stay inside the caller buffer even when asserts are compiled out.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include <faiss/IndexLattice.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/io.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>
#include <faiss/utils/hamming.h>

namespace {

constexpr int kDim = 6;
constexpr int kNsq = 3;
constexpr int kR2 = 1;
constexpr int kWrappedScaleNbit = 1431655764;
constexpr int kValidScaleNbit = 4;

template <typename T>
void push_val(std::vector<uint8_t>& buf, T val) {
    const auto* p = reinterpret_cast<const uint8_t*>(&val);
    buf.insert(buf.end(), p, p + sizeof(T));
}

template <typename T>
void push_vector(std::vector<uint8_t>& buf, const std::vector<T>& vec) {
    push_val<size_t>(buf, vec.size());
    if (!vec.empty()) {
        const auto* p = reinterpret_cast<const uint8_t*>(vec.data());
        buf.insert(buf.end(), p, p + vec.size() * sizeof(T));
    }
}

void push_fourcc(std::vector<uint8_t>& buf, const char s[4]) {
    const auto* x = reinterpret_cast<const unsigned char*>(s);
    uint32_t h = x[0] | (x[1] << 8) | (x[2] << 16) | (x[3] << 24);
    push_val<uint32_t>(buf, h);
}

// Byte layout of write_index for IndexLattice (IxLa) at this commit:
// fourcc, d, nsq, scale_nbit, r2, then write_index_header, then trained.
std::vector<uint8_t> make_ixla(int scale_nbit, bool is_trained) {
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxLa");
    push_val<int>(buf, kDim);
    push_val<int>(buf, kNsq);
    push_val<int>(buf, scale_nbit);
    push_val<int>(buf, kR2);

    push_val<int>(buf, kDim);
    push_val<int64_t>(buf, 0); // ntotal
    push_val<int64_t>(buf, int64_t{1} << 20);
    push_val<int64_t>(buf, int64_t{1} << 20);
    push_val<uint8_t>(buf, is_trained ? 1 : 0);
    push_val<int>(buf, static_cast<int>(faiss::METRIC_L2));

    // mins[nsq] then maxs[nsq]. Equal endpoints would divide by zero.
    std::vector<float> trained(kNsq * 2, 0.f);
    for (int i = 0; i < kNsq; i++) {
        trained[kNsq + i] = 1.f;
    }
    push_vector<float>(buf, trained);
    return buf;
}

std::vector<float> zero_query() {
    return std::vector<float>(kDim, 0.f);
}

// Caller-owned code buffer followed immediately by an inaccessible region
// large enough to cover a cursor advanced by scale_nbit bits.
struct GuardCode {
    uint8_t* map = nullptr;
    size_t span = 0;
    uint8_t* code = nullptr;

    GuardCode(size_t code_size, size_t far_byte) {
        long page_l = sysconf(_SC_PAGESIZE);
        if (page_l <= 0) {
            throw std::runtime_error("sysconf(_SC_PAGESIZE) failed");
        }
        size_t page = static_cast<size_t>(page_l);
        size_t writable =
                code_size == 0 ? 0 : (code_size + page - 1) & ~(page - 1);
        size_t cover = far_byte + page;
        span = writable + cover;
        span = (span + page - 1) & ~(page - 1);
        map = static_cast<uint8_t*>(mmap(
                nullptr,
                span,
                PROT_NONE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0));
        if (map == MAP_FAILED) {
            throw std::runtime_error("mmap of lattice code guard failed");
        }
        if (writable > 0) {
            if (mprotect(map, writable, PROT_READ | PROT_WRITE) != 0) {
                throw std::runtime_error("mprotect of lattice code buffer failed");
            }
            code = map + writable - code_size;
            memset(code, 0x5a, code_size);
        } else {
            code = map;
        }
    }

    ~GuardCode() {
        if (map && map != MAP_FAILED) {
            munmap(map, span);
        }
    }

    GuardCode(const GuardCode&) = delete;
    GuardCode& operator=(const GuardCode&) = delete;
};

size_t far_byte_for(const faiss::IndexLattice& index) {
    if (index.scale_nbit <= 0) {
        return static_cast<size_t>(sysconf(_SC_PAGESIZE));
    }
    return static_cast<size_t>(index.scale_nbit) / 8 + 64;
}

// true when the payload is rejected. Any other outcome leaves `out` set.
bool load_ixla(int scale_nbit, std::unique_ptr<faiss::Index>& out) {
    faiss::VectorIOReader reader;
    reader.data = make_ixla(scale_nbit, true);
    try {
        out = faiss::read_index_up(&reader);
    } catch (const faiss::FaissException&) {
        out.reset();
        return true;
    }
    return false;
}

faiss::IndexLattice* as_lattice(faiss::Index& index) {
    auto* lattice = dynamic_cast<faiss::IndexLattice*>(&index);
    EXPECT_NE(lattice, nullptr);
    return lattice;
}

void note_loaded(const char* op, const faiss::IndexLattice& index) {
    fprintf(stderr,
            "%s: loaded IxLa d=%d nsq=%d scale_nbit=%d lattice_nbit=%d "
            "code_size=%zu\n",
            op,
            index.d,
            index.nsq,
            index.scale_nbit,
            index.lattice_nbit,
            index.code_size);
}

// Rejecting the payload is the safe result. If it was accepted, `fn` must
// refuse without touching memory outside the code buffer. Returning normally
// still fails: a wrapped code size must not be usable.
template <class Fn>
void expect_rejected(const char* op, Fn&& fn) {
    std::unique_ptr<faiss::Index> index;
    if (load_ixla(kWrappedScaleNbit, index)) {
        return;
    }
    ASSERT_NE(index, nullptr);
    faiss::IndexLattice* lattice = as_lattice(*index);
    ASSERT_NE(lattice, nullptr);
    note_loaded(op, *lattice);
    try {
        fn(*index, *lattice);
    } catch (const faiss::FaissException& ex) {
        fprintf(stderr, "%s: refused after load (%s)\n", op, ex.what());
        return;
    }
    FAIL() << op
           << " accepted a lattice index whose scale width wraps the "
              "stored code size";
}

void seed_trained(faiss::IndexLattice& index) {
    index.is_trained = true;
    index.trained.assign(kNsq * 2, 0.f);
    for (int i = 0; i < kNsq; i++) {
        index.trained[kNsq + i] = 1.f;
    }
}

} // namespace

TEST(IndexLatticeCodeSizeWrap, ValidIndexRoundTrip) {
    std::unique_ptr<faiss::Index> index;
    ASSERT_FALSE(load_ixla(kValidScaleNbit, index));
    ASSERT_NE(index, nullptr);
    faiss::IndexLattice* lattice = as_lattice(*index);
    ASSERT_NE(lattice, nullptr);
    const int64_t total_nbit =
            (static_cast<int64_t>(lattice->lattice_nbit) + kValidScaleNbit) *
            kNsq;
    const size_t expect = static_cast<size_t>((total_nbit + 7) / 8);
    EXPECT_EQ(lattice->code_size, expect);
    EXPECT_GT(lattice->code_size, 1u);
    EXPECT_LE(lattice->code_size, static_cast<size_t>(kDim) * sizeof(float));

    std::vector<float> x = zero_query();
    std::vector<uint8_t> codes(lattice->code_size, 0x5a);
    ASSERT_NO_THROW(index->sa_encode(1, x.data(), codes.data()));
    std::vector<float> decoded(kDim, 0.f);
    ASSERT_NO_THROW(index->sa_decode(1, codes.data(), decoded.data()));
    ASSERT_NO_THROW(index->add(1, x.data()));
    EXPECT_EQ(index->ntotal, 1);

    std::vector<float> recon(kDim, 0.f);
    ASSERT_NO_THROW(index->reconstruct(0, recon.data()));
    float dist = 0.f;
    faiss::idx_t label = -1;
    ASSERT_NO_THROW(index->search(1, x.data(), 1, &dist, &label));
}

TEST(IndexLatticeCodeSizeWrap, ReadIndexSaEncodeStaysInCodeBuffer) {
    expect_rejected("sa_encode", [](faiss::Index& index, faiss::IndexLattice& lattice) {
        GuardCode guard(index.sa_code_size(), far_byte_for(lattice));
        std::vector<float> x = zero_query();
        index.sa_encode(1, x.data(), guard.code);
    });
}

TEST(IndexLatticeCodeSizeWrap, ReadIndexAddStaysInCodeBuffer) {
    expect_rejected("add", [](faiss::Index& index, faiss::IndexLattice&) {
        std::vector<float> x = zero_query();
        index.add(1, x.data());
    });
}

TEST(IndexLatticeCodeSizeWrap, ReadIndexAddExStaysInCodeBuffer) {
    expect_rejected("add_ex", [](faiss::Index& index, faiss::IndexLattice&) {
        std::vector<float> x = zero_query();
        index.add_ex(1, x.data(), faiss::NumericType::Float32);
    });
}

TEST(IndexLatticeCodeSizeWrap, ReadIndexSaDecodeStaysInCodeBuffer) {
    expect_rejected("sa_decode", [](faiss::Index& index, faiss::IndexLattice& lattice) {
        GuardCode guard(index.sa_code_size(), far_byte_for(lattice));
        std::vector<float> decoded(kDim, 0.f);
        index.sa_decode(1, guard.code, decoded.data());
    });
}

TEST(IndexLatticeCodeSizeWrap, ReadIndexReconstructStaysInCodeBuffer) {
    expect_rejected("reconstruct", [](faiss::Index& index, faiss::IndexLattice&) {
        std::vector<uint8_t> code(index.sa_code_size(), 0);
        index.add_sa_codes(1, code.data(), nullptr);
        std::vector<float> recon(kDim, 0.f);
        index.reconstruct(0, recon.data());
    });
}

TEST(IndexLatticeCodeSizeWrap, ReadIndexReconstructNStaysInCodeBuffer) {
    expect_rejected(
            "reconstruct_n", [](faiss::Index& index, faiss::IndexLattice&) {
                std::vector<uint8_t> code(index.sa_code_size(), 0);
                index.add_sa_codes(1, code.data(), nullptr);
                std::vector<float> recon(kDim, 0.f);
                index.reconstruct_n(0, 1, recon.data());
            });
}

TEST(IndexLatticeCodeSizeWrap, ReadIndexSearchStaysInCodeBuffer) {
    expect_rejected("search", [](faiss::Index& index, faiss::IndexLattice&) {
        std::vector<uint8_t> code(index.sa_code_size(), 0);
        index.add_sa_codes(1, code.data(), nullptr);
        std::vector<float> x = zero_query();
        float dist = 0.f;
        faiss::idx_t label = -1;
        index.search(1, x.data(), 1, &dist, &label);
    });
}

TEST(IndexLatticeCodeSizeWrap, ReadIndexSearchExStaysInCodeBuffer) {
    expect_rejected("search_ex", [](faiss::Index& index, faiss::IndexLattice&) {
        std::vector<uint8_t> code(index.sa_code_size(), 0);
        index.add_sa_codes(1, code.data(), nullptr);
        std::vector<float> x = zero_query();
        float dist = 0.f;
        faiss::idx_t label = -1;
        index.search_ex(
                1, x.data(), faiss::NumericType::Float32, 1, &dist, &label);
    });
}

TEST(IndexLatticeCodeSizeWrap, ReadIndexSearch1StaysInCodeBuffer) {
    expect_rejected("search1", [](faiss::Index& index, faiss::IndexLattice&) {
        std::vector<uint8_t> code(index.sa_code_size(), 0);
        index.add_sa_codes(1, code.data(), nullptr);
        std::vector<float> x = zero_query();
        float dist = 1.0e30f;
        faiss::idx_t label = -1;
        faiss::HeapResultHandler<faiss::CMax<float, faiss::idx_t>> handler(
                1, &dist, &label);
        index.search1(x.data(), handler);
    });
}

TEST(IndexLatticeCodeSizeWrap, ReadIndexRangeSearchStaysInCodeBuffer) {
    expect_rejected(
            "range_search", [](faiss::Index& index, faiss::IndexLattice&) {
                std::vector<uint8_t> code(index.sa_code_size(), 0);
                index.add_sa_codes(1, code.data(), nullptr);
                std::vector<float> x = zero_query();
                faiss::RangeSearchResult result(1);
                index.range_search(1, x.data(), 1.0e30f, &result);
            });
}

TEST(IndexLatticeCodeSizeWrap, FactoryStringSaEncodeStaysInCodeBuffer) {
    faiss::Index* raw = nullptr;
    try {
        raw = faiss::index_factory(kDim, "ZnLattice3x1_1431655764");
    } catch (const faiss::FaissException&) {
        return;
    }
    std::unique_ptr<faiss::Index> index(raw);
    ASSERT_NE(index, nullptr);
    faiss::IndexLattice* lattice = as_lattice(*index);
    ASSERT_NE(lattice, nullptr);
    note_loaded("index_factory", *lattice);
    seed_trained(*lattice);
    try {
        GuardCode guard(index->sa_code_size(), far_byte_for(*lattice));
        std::vector<float> x = zero_query();
        index->sa_encode(1, x.data(), guard.code);
    } catch (const faiss::FaissException&) {
        return;
    }
    FAIL() << "index_factory built a lattice whose scale width wraps the "
              "stored code size";
}

TEST(IndexLatticeCodeSizeWrap, BitWriterDoesNotAdvancePastBuffer) {
    // Same cursor sequence as sa_encode's first subspace when the scale
    // sample is 0 and the lattice code is 2 bits: the scale write advances
    // the cursor by the full width, and the next write stores at that cursor.
    const size_t code_size = 1;
    const size_t far = static_cast<size_t>(kWrappedScaleNbit) / 8 + 64;
    GuardCode guard(code_size, far);
    try {
        faiss::BitstringWriter writer(guard.code, code_size);
        writer.write(0, kWrappedScaleNbit);
        writer.write(1, 2);
    } catch (const faiss::FaissException&) {
        return;
    }
}

TEST(IndexLatticeCodeSizeWrap, BitReaderDoesNotAdvancePastBuffer) {
    const size_t code_size = 1;
    const size_t far = static_cast<size_t>(kWrappedScaleNbit) / 8 + 64;
    GuardCode guard(code_size, far);
    try {
        faiss::BitstringReader reader(guard.code, code_size);
        // The decoded value has to be consumed. A discarded read is a dead
        // load, and the compiler deletes it, which would hide the overrun.
        volatile uint64_t decoded = reader.read(kWrappedScaleNbit);
        (void)decoded;
    } catch (const faiss::FaissException&) {
        return;
    }
}
