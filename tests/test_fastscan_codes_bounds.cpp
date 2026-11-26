/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Flat fast-scan loaders (IndexPQFastScan, IndexSQFastScan,
// IndexAdditiveQuantizerFastScan and its four concrete fourccs, and
// IndexRaBitQFastScan) store ntotal2 and the packed code buffer independently.
// Search then walks ntotal2 codes. These tests require read_index to reject an
// index whose packed codes do not cover every block search will load.
//
// They fail while that check is missing: read_index returns the index, and
// search reads past the buffer. They pass once the index is rejected.
//
// IVF fast-scan indexes are not covered here. Their codes live in
// BlockInvertedLists, which already rejects a short list at load
// (faiss/invlists/BlockInvertedLists.cpp).

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <faiss/IndexAdditiveQuantizerFastScan.h>
#include <faiss/IndexPQFastScan.h>
#include <faiss/IndexRaBitQFastScan.h>
#include <faiss/IndexSQFastScan.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>

namespace {

constexpr int kDim = 4;
constexpr int kSubquantizers = 2;
constexpr int kNbits = 4;
constexpr int kTrain = 256;

std::vector<float> random_vectors(int n, int d, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> x(static_cast<size_t>(n) * d);
    for (float& v : x) {
        v = dist(rng);
    }
    return x;
}

template <typename T>
void append_pod(std::vector<uint8_t>& out, const T& value) {
    const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(T));
}

template <typename T>
void write_pod(std::vector<uint8_t>& data, size_t offset, const T& value) {
    ASSERT_LE(offset + sizeof(T), data.size());
    std::memcpy(data.data() + offset, &value, sizeof(T));
}

size_t find_last(
        const std::vector<uint8_t>& data,
        const std::vector<uint8_t>& pattern) {
    if (pattern.empty() || data.size() < pattern.size()) {
        return static_cast<size_t>(-1);
    }
    for (size_t i = data.size() - pattern.size();; --i) {
        if (std::memcmp(data.data() + i, pattern.data(), pattern.size()) ==
            0) {
            return i;
        }
        if (i == 0) {
            break;
        }
    }
    return static_cast<size_t>(-1);
}

// Bytes search loads for one SIMD block. Matches CodePackerPQ4::block_size
// for the PQ4 layout used by PQ, SQ, and additive-quantizer fast scan.
size_t pq4_block_stride(size_t m, int bbs) {
    return ((m + 1) / 2) * static_cast<size_t>(bbs);
}

struct LocatedCodes {
    size_t implem_off = static_cast<size_t>(-1);
    size_t ntotal2_off = static_cast<size_t>(-1);
    size_t codes_size_off = static_cast<size_t>(-1);
    size_t suffix = 0;
};

// PQ, additive-quantizer, and RaBitQ fast-scan files end with the codes
// vector. SQ appends rerank_factor and an empty low-bit vector.
LocatedCodes locate_codes(
        const std::vector<uint8_t>& data,
        const std::vector<uint8_t>& pattern,
        size_t codes_size,
        size_t suffix) {
    LocatedCodes loc;
    size_t at = find_last(data, pattern);
    EXPECT_NE(at, static_cast<size_t>(-1)) << "packed-code tail not found";
    if (at == static_cast<size_t>(-1)) {
        return loc;
    }
    // pattern ends with the size_t codes length.
    loc.codes_size_off = at + pattern.size() - sizeof(size_t);
    loc.suffix = suffix;
    const size_t end = loc.codes_size_off + sizeof(size_t) + codes_size + suffix;
    EXPECT_EQ(end, data.size()) << "packed-code tail is not where expected";
    return loc;
}

LocatedCodes locate_pq_like(
        const std::vector<uint8_t>& data,
        int implem,
        int bbs,
        int qbs,
        size_t ntotal2,
        size_t m2,
        size_t codes_size) {
    std::vector<uint8_t> pattern;
    append_pod<int>(pattern, implem);
    append_pod<int>(pattern, bbs);
    append_pod<int>(pattern, qbs);
    append_pod<size_t>(pattern, ntotal2);
    append_pod<size_t>(pattern, m2);
    append_pod<size_t>(pattern, codes_size);
    LocatedCodes loc = locate_codes(data, pattern, codes_size, 0);
    loc.implem_off = loc.codes_size_off - (sizeof(size_t) * 2 + sizeof(int) * 3);
    loc.ntotal2_off = loc.implem_off + sizeof(int) * 3;
    return loc;
}

LocatedCodes locate_sq(
        const std::vector<uint8_t>& data,
        const faiss::IndexSQFastScan& index) {
    std::vector<uint8_t> pattern;
    append_pod<int>(pattern, index.implem);
    append_pod<int>(pattern, index.bbs);
    append_pod<int>(pattern, index.qbs);
    append_pod<size_t>(pattern, index.ntotal2);
    append_pod<size_t>(pattern, index.M2);
    append_pod<size_t>(pattern, index.codes.size());
    append_pod<float>(pattern, index.rerank_factor);
    append_pod<size_t>(pattern, index.lo_codes.size());
    // The size field sits before the payload, which is followed by the
    // rerank factor and the low-bit vector already included above. Those
    // trailing fields are the suffix relative to the codes payload, but they
    // are also inside `pattern`. Search for the header without the suffix
    // and check the suffix separately.
    pattern.resize(pattern.size() - sizeof(float) - sizeof(size_t));
    const size_t suffix = sizeof(float) + sizeof(size_t) + index.lo_codes.size();
    LocatedCodes loc = locate_codes(data, pattern, index.codes.size(), suffix);
    loc.implem_off = loc.codes_size_off - (sizeof(size_t) * 2 + sizeof(int) * 3);
    loc.ntotal2_off = loc.implem_off + sizeof(int) * 3;
    if (loc.codes_size_off != static_cast<size_t>(-1) &&
        loc.codes_size_off + sizeof(size_t) + index.codes.size() + suffix <=
                data.size()) {
        float rerank = 0;
        size_t lo = 1;
        std::memcpy(
                &rerank,
                data.data() + loc.codes_size_off + sizeof(size_t) +
                        index.codes.size(),
                sizeof(rerank));
        std::memcpy(
                &lo,
                data.data() + loc.codes_size_off + sizeof(size_t) +
                        index.codes.size() + sizeof(float),
                sizeof(lo));
        EXPECT_EQ(rerank, index.rerank_factor);
        EXPECT_EQ(lo, index.lo_codes.size());
    }
    return loc;
}

LocatedCodes locate_aq(
        const std::vector<uint8_t>& data,
        const faiss::IndexAdditiveQuantizerFastScan& index) {
    std::vector<uint8_t> pattern;
    append_pod<int>(pattern, index.implem);
    append_pod<int>(pattern, index.bbs);
    append_pod<int>(pattern, index.qbs);
    append_pod<size_t>(pattern, index.M);
    append_pod<size_t>(pattern, index.nbits);
    append_pod<size_t>(pattern, index.ksub);
    append_pod<size_t>(pattern, index.code_size);
    append_pod<size_t>(pattern, index.ntotal2);
    append_pod<size_t>(pattern, index.M2);
    append_pod<bool>(pattern, index.rescale_norm);
    append_pod<int>(pattern, index.norm_scale);
    append_pod<size_t>(pattern, index.max_train_points);
    append_pod<size_t>(pattern, index.codes.size());
    LocatedCodes loc = locate_codes(data, pattern, index.codes.size(), 0);
    loc.implem_off = loc.codes_size_off -
            (sizeof(size_t) * 7 + sizeof(int) * 3 + sizeof(bool) + sizeof(int));
    // ntotal2 follows implem, bbs, qbs, M, nbits, ksub, code_size.
    loc.ntotal2_off = loc.implem_off + sizeof(int) * 3 + sizeof(size_t) * 4;
    return loc;
}

LocatedCodes locate_rabitq(
        const std::vector<uint8_t>& data,
        const faiss::IndexRaBitQFastScan& index) {
    std::vector<uint8_t> pattern;
    append_pod<uint8_t>(pattern, index.qb);
    append_pod<int>(pattern, index.bbs);
    append_pod<size_t>(pattern, index.ntotal2);
    append_pod<size_t>(pattern, index.M2);
    append_pod<size_t>(pattern, index.code_size);
    append_pod<size_t>(pattern, index.codes.size());
    LocatedCodes loc = locate_codes(data, pattern, index.codes.size(), 0);
    loc.ntotal2_off = loc.codes_size_off - sizeof(size_t) * 3;
    loc.implem_off = static_cast<size_t>(-1);
    return loc;
}

void resize_codes(
        std::vector<uint8_t>& data,
        const LocatedCodes& loc,
        size_t old_size,
        size_t new_size) {
    ASSERT_NE(loc.codes_size_off, static_cast<size_t>(-1));
    const size_t payload = loc.codes_size_off + sizeof(size_t);
    ASSERT_LE(payload + old_size + loc.suffix, data.size());
    std::vector<uint8_t> suffix(
            data.begin() + static_cast<std::ptrdiff_t>(payload + old_size),
            data.begin() +
                    static_cast<std::ptrdiff_t>(payload + old_size + loc.suffix));
    std::vector<uint8_t> rewritten(
            data.begin(), data.begin() + static_cast<std::ptrdiff_t>(payload));
    rewritten.resize(payload + new_size, 0);
    const size_t keep = std::min(old_size, new_size);
    if (keep > 0) {
        std::memcpy(rewritten.data() + payload, data.data() + payload, keep);
    }
    rewritten.insert(rewritten.end(), suffix.begin(), suffix.end());
    data.swap(rewritten);
    write_pod<size_t>(data, loc.codes_size_off, new_size);
}

std::vector<uint8_t> serialize(const faiss::Index& index) {
    faiss::VectorIOWriter writer;
    faiss::write_index(&index, &writer);
    return writer.data;
}

// A well-formed twin of the same bytes must search. The malformed bytes must
// be rejected by read_index. If load succeeds, search is invoked on that
// object so a check that does not cover the buffer search reads still fails.
void expect_load_rejected(std::vector<uint8_t> bytes) {
    faiss::VectorIOReader reader;
    reader.data = std::move(bytes);
    std::unique_ptr<faiss::Index> index;
    try {
        index = faiss::read_index_up(&reader);
    } catch (const faiss::FaissException&) {
        return;
    }
    ASSERT_NE(index, nullptr);
    std::vector<float> query(static_cast<size_t>(index->d), 0.25f);
    std::vector<float> distances(1);
    std::vector<faiss::idx_t> labels(1, -1);
    index->search(1, query.data(), 1, distances.data(), labels.data());
    FAIL() << "read_index accepted fast-scan codes that do not cover ntotal2, "
              "and search completed";
}

void expect_round_trip_searches(const faiss::Index& index) {
    faiss::VectorIOReader reader;
    reader.data = serialize(index);
    std::unique_ptr<faiss::Index> loaded;
    ASSERT_NO_THROW(loaded = faiss::read_index_up(&reader));
    ASSERT_NE(loaded, nullptr);
    std::vector<float> query(static_cast<size_t>(index.d), 0.25f);
    std::vector<float> distances(1);
    std::vector<faiss::idx_t> labels(1, -1);
    ASSERT_NO_THROW(
            loaded->search(1, query.data(), 1, distances.data(), labels.data()));
}

void train_and_add(faiss::Index& index, uint32_t seed) {
    auto xb = random_vectors(kTrain, index.d, seed);
    index.train(kTrain, xb.data());
    index.add(1, xb.data());
    ASSERT_EQ(index.ntotal, 1);
}

faiss::IndexPQFastScan make_pq(int bbs) {
    faiss::IndexPQFastScan index(kDim, kSubquantizers, kNbits, faiss::METRIC_L2, bbs);
    train_and_add(index, 1);
    return index;
}

faiss::IndexSQFastScan make_sq() {
    faiss::IndexSQFastScan index(
            kDim, faiss::ScalarQuantizer::QT_4bit, faiss::METRIC_L2, 32);
    train_and_add(index, 2);
    return index;
}

faiss::IndexResidualQuantizerFastScan make_rq() {
    faiss::IndexResidualQuantizerFastScan index(
            kDim,
            kSubquantizers,
            kNbits,
            faiss::METRIC_INNER_PRODUCT,
            faiss::AdditiveQuantizer::ST_LUT_nonorm,
            32);
    train_and_add(index, 3);
    return index;
}

faiss::IndexLocalSearchQuantizerFastScan make_lsq() {
    faiss::IndexLocalSearchQuantizerFastScan index(
            kDim,
            kSubquantizers,
            kNbits,
            faiss::METRIC_INNER_PRODUCT,
            faiss::AdditiveQuantizer::ST_LUT_nonorm,
            32);
    index.lsq.train_iters = 1;
    index.lsq.encode_ils_iters = 1;
    index.lsq.train_ils_iters = 1;
    index.lsq.icm_iters = 1;
    train_and_add(index, 4);
    return index;
}

faiss::IndexProductResidualQuantizerFastScan make_prq() {
    faiss::IndexProductResidualQuantizerFastScan index(
            kDim,
            2,
            1,
            kNbits,
            faiss::METRIC_INNER_PRODUCT,
            faiss::AdditiveQuantizer::ST_LUT_nonorm,
            32);
    train_and_add(index, 5);
    return index;
}

faiss::IndexProductLocalSearchQuantizerFastScan make_plsq() {
    faiss::IndexProductLocalSearchQuantizerFastScan index(
            kDim,
            2,
            1,
            kNbits,
            faiss::METRIC_INNER_PRODUCT,
            faiss::AdditiveQuantizer::ST_LUT_nonorm,
            32);
    for (faiss::AdditiveQuantizer* sub : index.plsq.quantizers) {
        auto* lsq = dynamic_cast<faiss::LocalSearchQuantizer*>(sub);
        if (lsq == nullptr) {
            throw std::runtime_error("product LSQ subquantizer has unexpected type");
        }
        lsq->train_iters = 1;
        lsq->encode_ils_iters = 1;
        lsq->train_ils_iters = 1;
        lsq->icm_iters = 1;
    }
    train_and_add(index, 6);
    return index;
}

faiss::IndexRaBitQFastScan make_rabitq() {
    faiss::IndexRaBitQFastScan index(kDim, faiss::METRIC_L2, 32);
    train_and_add(index, 7);
    return index;
}

} // namespace

TEST(FastScanCodesBounds, PqRoundTripSearches) {
    auto index = make_pq(32);
    ASSERT_EQ(index.ntotal2, 32u);
    ASSERT_EQ(index.codes.size(), pq4_block_stride(index.M, index.bbs));
    expect_round_trip_searches(index);
}

// Attack path: block size 32, 4-bit PQ, ntotal2 == 32, empty packed codes.
// Default implem 0 selects the qbs search loop.
TEST(FastScanCodesBounds, PqEmptyCodesDefaultSearchRejected) {
    auto index = make_pq(32);
    auto bytes = serialize(index);
    auto loc = locate_pq_like(
            bytes, index.implem, index.bbs, index.qbs, index.ntotal2, index.M2, index.codes.size());
    ASSERT_NE(loc.codes_size_off, static_cast<size_t>(-1));
    resize_codes(bytes, loc, index.codes.size(), 0);
    expect_load_rejected(std::move(bytes));
}

// One byte short of a single block. The rounded count stays a multiple of bbs.
TEST(FastScanCodesBounds, PqShortCodesDefaultSearchRejected) {
    auto index = make_pq(32);
    const size_t stride = pq4_block_stride(index.M, index.bbs);
    ASSERT_GT(stride, 1u);
    auto bytes = serialize(index);
    auto loc = locate_pq_like(
            bytes, index.implem, index.bbs, index.qbs, index.ntotal2, index.M2, index.codes.size());
    ASSERT_EQ(index.codes.size(), stride);
    resize_codes(bytes, loc, index.codes.size(), stride - 1);
    expect_load_rejected(std::move(bytes));
}

// implem 14 is the non-qbs accumulate loop. It has the same unchecked read.
TEST(FastScanCodesBounds, PqEmptyCodesImplem14Rejected) {
    auto index = make_pq(32);
    auto bytes = serialize(index);
    auto loc = locate_pq_like(
            bytes, index.implem, index.bbs, index.qbs, index.ntotal2, index.M2, index.codes.size());
    write_pod<int>(bytes, loc.implem_off, 14);
    resize_codes(bytes, loc, index.codes.size(), 0);
    expect_load_rejected(std::move(bytes));
}

// bbs > 32 also selects the non-qbs loop when implem is left at 0.
TEST(FastScanCodesBounds, PqEmptyCodesBbs64Rejected) {
    auto index = make_pq(64);
    ASSERT_EQ(index.bbs, 64);
    ASSERT_EQ(index.ntotal2, 64u);
    auto bytes = serialize(index);
    auto loc = locate_pq_like(
            bytes, index.implem, index.bbs, index.qbs, index.ntotal2, index.M2, index.codes.size());
    resize_codes(bytes, loc, index.codes.size(), 0);
    expect_load_rejected(std::move(bytes));
}

// The qbs loop steps 32 vectors at a time and does not require ntotal2 % bbs
// == 0 before loading the next block.
TEST(FastScanCodesBounds, PqUnalignedRoundedCountRejected) {
    auto index = make_pq(32);
    auto bytes = serialize(index);
    auto loc = locate_pq_like(
            bytes, index.implem, index.bbs, index.qbs, index.ntotal2, index.M2, index.codes.size());
    write_pod<size_t>(bytes, loc.ntotal2_off, static_cast<size_t>(index.bbs) + 1);
    expect_load_rejected(std::move(bytes));
}

// ntotal2 stays a multiple of bbs, but the buffer is shorter than the blocks
// that multiple implies. The allocation is exactly 256 bytes; search reads
// further than that.
TEST(FastScanCodesBounds, PqRoundedCountExceedsCodeBufferRejected) {
    auto index = make_pq(32);
    const size_t stride = pq4_block_stride(index.M, index.bbs);
    ASSERT_EQ(stride, 32u);
    constexpr size_t kKept = 256;
    constexpr size_t kBlocks = kKept / 32 + 1;
    auto bytes = serialize(index);
    auto loc = locate_pq_like(
            bytes, index.implem, index.bbs, index.qbs, index.ntotal2, index.M2, index.codes.size());
    write_pod<size_t>(bytes, loc.ntotal2_off, kBlocks * 32);
    resize_codes(bytes, loc, index.codes.size(), kKept);
    expect_load_rejected(std::move(bytes));
}

TEST(FastScanCodesBounds, SqEmptyCodesRejected) {
    auto index = make_sq();
    auto bytes = serialize(index);
    auto loc = locate_sq(bytes, index);
    ASSERT_NE(loc.codes_size_off, static_cast<size_t>(-1));
    resize_codes(bytes, loc, index.codes.size(), 0);
    expect_load_rejected(std::move(bytes));
}

TEST(FastScanCodesBounds, SqShortCodesRejected) {
    auto index = make_sq();
    const size_t stride = pq4_block_stride(index.M, index.bbs);
    ASSERT_EQ(index.codes.size(), stride);
    auto bytes = serialize(index);
    auto loc = locate_sq(bytes, index);
    resize_codes(bytes, loc, index.codes.size(), stride - 1);
    expect_load_rejected(std::move(bytes));
}

TEST(FastScanCodesBounds, RqEmptyCodesRejected) {
    auto index = make_rq();
    auto bytes = serialize(index);
    auto loc = locate_aq(bytes, index);
    ASSERT_NE(loc.codes_size_off, static_cast<size_t>(-1));
    resize_codes(bytes, loc, index.codes.size(), 0);
    expect_load_rejected(std::move(bytes));
}

TEST(FastScanCodesBounds, LsqEmptyCodesRejected) {
    auto index = make_lsq();
    auto bytes = serialize(index);
    auto loc = locate_aq(bytes, index);
    ASSERT_NE(loc.codes_size_off, static_cast<size_t>(-1));
    resize_codes(bytes, loc, index.codes.size(), 0);
    expect_load_rejected(std::move(bytes));
}

TEST(FastScanCodesBounds, PrqEmptyCodesRejected) {
    auto index = make_prq();
    auto bytes = serialize(index);
    auto loc = locate_aq(bytes, index);
    ASSERT_NE(loc.codes_size_off, static_cast<size_t>(-1));
    resize_codes(bytes, loc, index.codes.size(), 0);
    expect_load_rejected(std::move(bytes));
}

TEST(FastScanCodesBounds, PlsqEmptyCodesRejected) {
    auto index = make_plsq();
    auto bytes = serialize(index);
    auto loc = locate_aq(bytes, index);
    ASSERT_NE(loc.codes_size_off, static_cast<size_t>(-1));
    resize_codes(bytes, loc, index.codes.size(), 0);
    expect_load_rejected(std::move(bytes));
}

TEST(FastScanCodesBounds, RabitqEmptyCodesRejected) {
    auto index = make_rabitq();
    ASSERT_GT(index.codes.size(), 0u);
    auto bytes = serialize(index);
    auto loc = locate_rabitq(bytes, index);
    ASSERT_NE(loc.codes_size_off, static_cast<size_t>(-1));
    resize_codes(bytes, loc, index.codes.size(), 0);
    expect_load_rejected(std::move(bytes));
}

TEST(FastScanCodesBounds, RabitqShortCodesRejected) {
    auto index = make_rabitq();
    ASSERT_GT(index.codes.size(), 1u);
    auto bytes = serialize(index);
    auto loc = locate_rabitq(bytes, index);
    resize_codes(bytes, loc, index.codes.size(), index.codes.size() - 1);
    expect_load_rejected(std::move(bytes));
}
