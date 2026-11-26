/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A trained RaBitQ index stores its centroid as a vector whose length is
// written independently of the index dimension. These tests build a real
// index, shorten that vector in the serialized bytes, and require read_index
// to reject the result before search, range search, or encode/decode can use
// it. They fail while a short non-empty centroid is still accepted.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <faiss/IndexRaBitQ.h>
#include <faiss/IndexRaBitQFastScan.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>

namespace {

constexpr int kDim = 8;
constexpr int kN = 4;

enum class Op { Search, RangeSearch, SaEncode, SaDecode };

struct CentroidLoc {
    std::string fourcc;
    size_t count_offset = 0;
    size_t count = 0;
    size_t qb_offset = 0;
};

class ByteCursor {
   public:
    explicit ByteCursor(const std::vector<uint8_t>& bytes) : bytes_(bytes) {}

    template <typename T>
    T take() {
        EXPECT_LE(i_ + sizeof(T), bytes_.size());
        T value{};
        if (i_ + sizeof(T) <= bytes_.size()) {
            memcpy(&value, bytes_.data() + i_, sizeof(T));
        }
        i_ += sizeof(T);
        return value;
    }

    void skip(size_t n) {
        EXPECT_LE(i_ + n, bytes_.size());
        i_ += n;
    }

    size_t offset() const {
        return i_;
    }

   private:
    const std::vector<uint8_t>& bytes_;
    size_t i_ = 0;
};

std::string fourcc_string(uint32_t h) {
    char s[5] = {
            static_cast<char>(h & 0xff),
            static_cast<char>((h >> 8) & 0xff),
            static_cast<char>((h >> 16) & 0xff),
            static_cast<char>((h >> 24) & 0xff),
            0};
    return std::string(s);
}

CentroidLoc locate_centroid(const std::vector<uint8_t>& bytes) {
    ByteCursor cur(bytes);
    const std::string tag = fourcc_string(cur.take<uint32_t>());
    cur.take<int>();             // d
    cur.take<int64_t>();         // ntotal
    cur.take<int64_t>();         // dummy
    cur.take<int64_t>();         // dummy
    cur.take<uint8_t>();         // is_trained
    const int metric = cur.take<int>();
    if (metric > 1) {
        cur.take<float>();
    }
    cur.take<size_t>(); // RaBitQuantizer::d
    cur.take<size_t>(); // RaBitQuantizer::code_size
    cur.take<int>();    // RaBitQuantizer::metric_type
    const bool fastscan = tag == "Irfn" || tag == "Irfs";
    const bool multi_bit = tag == "Ixrr" || fastscan;
    if (multi_bit) {
        cur.take<size_t>(); // nb_bits
    }
    if (!fastscan) {
        const size_t ncodes = cur.take<size_t>();
        cur.skip(ncodes);
    }
    CentroidLoc loc;
    loc.fourcc = tag;
    loc.count_offset = cur.offset();
    loc.count = cur.take<size_t>();
    loc.qb_offset = cur.offset() + loc.count * sizeof(float);
    return loc;
}

void set_centroid_length(std::vector<uint8_t>& bytes, size_t new_count) {
    const CentroidLoc loc = locate_centroid(bytes);
    ASSERT_GT(loc.count, 0u);
    const size_t payload = loc.count_offset + sizeof(size_t);
    if (new_count < loc.count) {
        const size_t drop_at = payload + new_count * sizeof(float);
        const size_t drop_end = payload + loc.count * sizeof(float);
        bytes.erase(bytes.begin() + drop_at, bytes.begin() + drop_end);
    } else if (new_count > loc.count) {
        bytes.insert(
                bytes.begin() + payload + loc.count * sizeof(float),
                (new_count - loc.count) * sizeof(float),
                0);
    }
    memcpy(bytes.data() + loc.count_offset, &new_count, sizeof(size_t));
    const CentroidLoc after = locate_centroid(bytes);
    ASSERT_EQ(after.count, new_count);
}

void set_qb(std::vector<uint8_t>& bytes, uint8_t qb) {
    const CentroidLoc loc = locate_centroid(bytes);
    ASSERT_LT(loc.qb_offset, bytes.size());
    bytes[loc.qb_offset] = qb;
}

// is_trained is the single byte after d and three idx_t header fields.
void set_trained(std::vector<uint8_t>& bytes, bool trained) {
    const size_t off = sizeof(uint32_t) + sizeof(int) + 3 * sizeof(int64_t);
    ASSERT_LT(off, bytes.size());
    bytes[off] = trained ? 1 : 0;
}

std::vector<uint8_t> to_legacy_fastscan(std::vector<uint8_t> bytes) {
    const CentroidLoc loc = locate_centroid(bytes);
    EXPECT_EQ(loc.fourcc, "Irfn");
    bytes[3] = 's';
    const size_t insert_at = loc.qb_offset + 1;
    const size_t zero = 0;
    const auto* raw = reinterpret_cast<const uint8_t*>(&zero);
    bytes.insert(bytes.begin() + insert_at, raw, raw + sizeof(zero));
    return bytes;
}

std::vector<float> sample_vectors(int n, int d) {
    std::vector<float> x(static_cast<size_t>(n) * d);
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = static_cast<float>((i % 11) + 1) * 0.05f;
    }
    return x;
}

std::vector<uint8_t> write_flat(int nb_bits, int n = kN) {
    faiss::IndexRaBitQ index(kDim, faiss::METRIC_L2, nb_bits);
    const std::vector<float> xb = sample_vectors(n, kDim);
    index.train(n, xb.data());
    if (n > 0) {
        index.add(n, xb.data());
    }
    faiss::VectorIOWriter writer;
    faiss::write_index(&index, &writer);
    return writer.data;
}

std::vector<uint8_t> write_fastscan(int nb_bits, int n = kN) {
    faiss::IndexRaBitQFastScan index(kDim, faiss::METRIC_L2, 32, nb_bits);
    const std::vector<float> xb = sample_vectors(n, kDim);
    index.train(n, xb.data());
    if (n > 0) {
        index.add(n, xb.data());
    }
    faiss::VectorIOWriter writer;
    faiss::write_index(&index, &writer);
    return writer.data;
}

std::unique_ptr<faiss::Index> load_index(const std::vector<uint8_t>& bytes) {
    faiss::VectorIOReader reader;
    reader.data = bytes;
    return std::unique_ptr<faiss::Index>(faiss::read_index(&reader));
}

void exercise(faiss::Index& index, Op op) {
    const std::vector<float> query = sample_vectors(1, index.d);
    switch (op) {
        case Op::Search: {
            std::vector<float> distances(1);
            std::vector<faiss::idx_t> labels(1, -1);
            index.search(1, query.data(), 1, distances.data(), labels.data());
            break;
        }
        case Op::RangeSearch: {
            faiss::RangeSearchResult result(1);
            index.range_search(1, query.data(), 1.0e6f, &result);
            break;
        }
        case Op::SaEncode: {
            std::vector<uint8_t> codes(index.sa_code_size());
            index.sa_encode(1, query.data(), codes.data());
            break;
        }
        case Op::SaDecode: {
            std::vector<uint8_t> codes(index.sa_code_size(), 0);
            std::vector<float> decoded(index.d);
            index.sa_decode(1, codes.data(), decoded.data());
            break;
        }
    }
}

// The fixed loader throws from read_index. Any path that returns an index
// still has to run the real operation, which reads one centroid float per
// dimension, and then the test fails.
void expect_short_centroid_rejected(
        std::vector<uint8_t> bytes,
        size_t centroid_len,
        Op op) {
    set_centroid_length(bytes, centroid_len);
    faiss::VectorIOReader reader;
    reader.data = std::move(bytes);
    std::unique_ptr<faiss::Index> index;
    try {
        index.reset(faiss::read_index(&reader));
    } catch (const faiss::FaissException& ex) {
        const std::string message = ex.what();
        EXPECT_NE(message.find("center"), std::string::npos) << message;
        return;
    }
    ASSERT_NE(index, nullptr);
    EXPECT_EQ(index->d, kDim);
    try {
        exercise(*index, op);
    } catch (const std::exception& ex) {
        FAIL() << "read_index accepted a centroid of length " << centroid_len
               << " for dimension " << kDim << "; operation threw: " << ex.what();
    }
    FAIL() << "read_index accepted a centroid of length " << centroid_len
           << " for dimension " << kDim << " and the operation completed";
}

} // namespace

TEST(RaBitQShortCentroid, WellFormedFlatAndFastScanSearch) {
    for (int nb_bits : {1, 4}) {
        const std::vector<uint8_t> flat = write_flat(nb_bits);
        const CentroidLoc flat_loc = locate_centroid(flat);
        EXPECT_EQ(flat_loc.fourcc, nb_bits == 1 ? "Ixrq" : "Ixrr");
        EXPECT_EQ(flat_loc.count, static_cast<size_t>(kDim));
        std::unique_ptr<faiss::Index> flat_index = load_index(flat);
        ASSERT_NE(flat_index, nullptr);
        EXPECT_TRUE(flat_index->is_trained);
        exercise(*flat_index, Op::Search);
        exercise(*flat_index, Op::RangeSearch);
        exercise(*flat_index, Op::SaEncode);
        exercise(*flat_index, Op::SaDecode);

        const std::vector<uint8_t> fast = write_fastscan(nb_bits);
        const CentroidLoc fast_loc = locate_centroid(fast);
        EXPECT_EQ(fast_loc.fourcc, "Irfn");
        EXPECT_EQ(fast_loc.count, static_cast<size_t>(kDim));
        std::unique_ptr<faiss::Index> fast_index = load_index(fast);
        ASSERT_NE(fast_index, nullptr);
        EXPECT_TRUE(fast_index->is_trained);
        exercise(*fast_index, Op::Search);
        exercise(*fast_index, Op::SaEncode);
        exercise(*fast_index, Op::SaDecode);
    }

    const std::vector<uint8_t> legacy = to_legacy_fastscan(write_fastscan(1, 0));
    EXPECT_EQ(locate_centroid(legacy).fourcc, "Irfs");
    std::unique_ptr<faiss::Index> legacy_index = load_index(legacy);
    ASSERT_NE(legacy_index, nullptr);
    exercise(*legacy_index, Op::Search);
}

TEST(RaBitQShortCentroid, FlatOneBitSearchRejectsOneFloatCenter) {
    expect_short_centroid_rejected(write_flat(1), 1, Op::Search);
}

TEST(RaBitQShortCentroid, FlatOneBitSearchRejectsShortCenter) {
    expect_short_centroid_rejected(write_flat(1), kDim - 1, Op::Search);
}

TEST(RaBitQShortCentroid, FlatOneBitSearchRejectsLongCenter) {
    expect_short_centroid_rejected(write_flat(1), kDim + 1, Op::Search);
}

TEST(RaBitQShortCentroid, FlatOneBitSearchRejectsEmptyCenter) {
    expect_short_centroid_rejected(write_flat(1), 0, Op::Search);
}

TEST(RaBitQShortCentroid, FlatOneBitQueryBitsZeroStillRejects) {
    std::vector<uint8_t> bytes = write_flat(1);
    set_qb(bytes, 0);
    expect_short_centroid_rejected(std::move(bytes), 1, Op::Search);
}

TEST(RaBitQShortCentroid, FlatOneBitRangeSearchRejects) {
    expect_short_centroid_rejected(write_flat(1), 1, Op::RangeSearch);
}

TEST(RaBitQShortCentroid, FlatOneBitSaEncodeRejects) {
    expect_short_centroid_rejected(write_flat(1), 1, Op::SaEncode);
}

TEST(RaBitQShortCentroid, FlatOneBitSaDecodeRejects) {
    expect_short_centroid_rejected(write_flat(1), 1, Op::SaDecode);
}

TEST(RaBitQShortCentroid, FlatMultiBitSearchRejects) {
    expect_short_centroid_rejected(write_flat(4), 1, Op::Search);
}

TEST(RaBitQShortCentroid, FastScanSearchRejects) {
    expect_short_centroid_rejected(write_fastscan(1), 1, Op::Search);
}

TEST(RaBitQShortCentroid, FastScanMultiBitSearchRejects) {
    expect_short_centroid_rejected(write_fastscan(4), 1, Op::Search);
}

TEST(RaBitQShortCentroid, FastScanSaEncodeRejects) {
    expect_short_centroid_rejected(write_fastscan(1), 1, Op::SaEncode);
}

TEST(RaBitQShortCentroid, FastScanSaDecodeRejects) {
    expect_short_centroid_rejected(write_fastscan(1), 1, Op::SaDecode);
}

TEST(RaBitQShortCentroid, LegacyFastScanSearchRejects) {
    // ntotal is zero so the legacy flat-storage vector can stay empty and
    // still pass the migration size check. The centroid read is in query setup.
    expect_short_centroid_rejected(
            to_legacy_fastscan(write_fastscan(1, 0)), 1, Op::Search);
}

TEST(RaBitQShortCentroid, UntrainedShortCenterLoadsAndSearchRefuses) {
    std::vector<uint8_t> bytes = write_flat(1);
    set_centroid_length(bytes, 1);
    set_trained(bytes, false);
    std::unique_ptr<faiss::Index> index = load_index(bytes);
    ASSERT_NE(index, nullptr);
    EXPECT_FALSE(index->is_trained);
    EXPECT_EQ(index->d, kDim);
    try {
        exercise(*index, Op::Search);
        FAIL() << "search of an untrained index returned";
    } catch (const faiss::FaissException&) {
    }
}
