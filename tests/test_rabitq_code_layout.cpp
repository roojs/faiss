/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Spec for the flat RaBitQ (Ixrq / Ixrr) code-layout checks. These tests
// fail while read_index keeps a file-supplied code size or bit width and a
// code buffer that is shorter than one encoded vector. Search, range
// search, reconstruct, and add then read or write past that buffer. They
// pass once read_index rejects the payload before returning it.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <omp.h>

#include <faiss/IndexRaBitQ.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/RaBitQuantizer.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>

namespace {

constexpr int kDim = 16;

enum class Use {
    Search,
    RangeSearch,
    Reconstruct,
    Add,
};

struct ParsedFlat {
    uint32_t fourcc_h = 0;
    int d = 0;
    int64_t ntotal = 0;
    size_t rabitq_d = 0;
    size_t code_size = 0;
    size_t code_size_off = 0;
    bool has_nb_bits = false;
    size_t nb_bits = 0;
    size_t nb_bits_off = 0;
    size_t codes_len_off = 0;
    size_t codes_len = 0;
};

void read_pod(const std::vector<uint8_t>& buf, size_t& off, void* dst, size_t n) {
    ASSERT_LE(off + n, buf.size()) << "truncated RaBitQ payload at offset " << off;
    memcpy(dst, buf.data() + off, n);
    off += n;
}

template <typename T>
T read_pod(const std::vector<uint8_t>& buf, size_t& off) {
    T value{};
    read_pod(buf, off, &value, sizeof(T));
    return value;
}

// Walk a flat Ixrq or Ixrr payload written by write_index. Fails the test
// if the bytes are not that layout, so a later rejection cannot be credited
// to a truncated or mis-framed file.
ParsedFlat parse_flat(const std::vector<uint8_t>& buf) {
    ParsedFlat parsed;
    size_t off = 0;
    parsed.fourcc_h = read_pod<uint32_t>(buf, off);
    const uint32_t ixrq = faiss::fourcc("Ixrq");
    const uint32_t ixrr = faiss::fourcc("Ixrr");
    EXPECT_TRUE(parsed.fourcc_h == ixrq || parsed.fourcc_h == ixrr);
    parsed.d = read_pod<int>(buf, off);
    parsed.ntotal = read_pod<int64_t>(buf, off);
    (void)read_pod<int64_t>(buf, off);
    (void)read_pod<int64_t>(buf, off);
    (void)read_pod<uint8_t>(buf, off);
    (void)read_pod<int>(buf, off);
    parsed.rabitq_d = read_pod<size_t>(buf, off);
    parsed.code_size_off = off;
    parsed.code_size = read_pod<size_t>(buf, off);
    (void)read_pod<int>(buf, off);
    parsed.has_nb_bits = parsed.fourcc_h == ixrr;
    if (parsed.has_nb_bits) {
        parsed.nb_bits_off = off;
        parsed.nb_bits = read_pod<size_t>(buf, off);
    }
    parsed.codes_len_off = off;
    parsed.codes_len = read_pod<size_t>(buf, off);
    EXPECT_LE(off + parsed.codes_len, buf.size());
    off += parsed.codes_len;
    const size_t center_len = read_pod<size_t>(buf, off);
    EXPECT_EQ(center_len, static_cast<size_t>(parsed.d));
    EXPECT_LE(off + center_len * sizeof(float) + 1, buf.size());
    off += center_len * sizeof(float);
    off += 1; // qb
    EXPECT_EQ(off, buf.size());
    return parsed;
}

std::vector<uint8_t> serialize_flat(int ntotal, uint8_t nb_bits, uint8_t qb) {
    faiss::IndexRaBitQ index(kDim, faiss::METRIC_L2, nb_bits);
    index.qb = qb;
    std::vector<float> xb(static_cast<size_t>(kDim) * ntotal, 0.25f);
    for (int i = 0; i < ntotal; i++) {
        xb[static_cast<size_t>(i) * kDim] = 0.5f + static_cast<float>(i);
    }
    if (ntotal == 0) {
        std::vector<float> train(kDim, 0.25f);
        index.train(1, train.data());
    } else {
        index.train(ntotal, xb.data());
        index.add(ntotal, xb.data());
    }
    faiss::VectorIOWriter writer;
    faiss::write_index(&index, &writer);
    return writer.data;
}

void write_at(std::vector<uint8_t>& buf, size_t off, const void* src, size_t n) {
    ASSERT_LE(off + n, buf.size());
    memcpy(buf.data() + off, src, n);
}

// Replace the stored code size and the codes vector, keeping the header,
// bit width, centroid, and query-quantization byte in place.
std::vector<uint8_t> with_codes(
        const std::vector<uint8_t>& original,
        size_t new_code_size,
        const std::vector<uint8_t>& codes) {
    const ParsedFlat parsed = parse_flat(original);
    std::vector<uint8_t> out;
    out.insert(out.end(), original.begin(), original.begin() + parsed.code_size_off);
    const size_t code_size_le = new_code_size;
    out.insert(
            out.end(),
            reinterpret_cast<const uint8_t*>(&code_size_le),
            reinterpret_cast<const uint8_t*>(&code_size_le) + sizeof(code_size_le));
    out.insert(
            out.end(),
            original.begin() + parsed.code_size_off + sizeof(size_t),
            original.begin() + parsed.codes_len_off);
    const size_t codes_len = codes.size();
    out.insert(
            out.end(),
            reinterpret_cast<const uint8_t*>(&codes_len),
            reinterpret_cast<const uint8_t*>(&codes_len) + sizeof(codes_len));
    out.insert(out.end(), codes.begin(), codes.end());
    const size_t tail = parsed.codes_len_off + sizeof(size_t) + parsed.codes_len;
    out.insert(out.end(), original.begin() + tail, original.end());
    const ParsedFlat check = parse_flat(out);
    EXPECT_EQ(check.d, kDim);
    EXPECT_EQ(check.rabitq_d, static_cast<size_t>(kDim));
    EXPECT_EQ(check.code_size, new_code_size);
    EXPECT_EQ(check.codes_len, codes.size());
    EXPECT_EQ(check.ntotal, parsed.ntotal);
    return out;
}

std::vector<uint8_t> as_ixrr(const std::vector<uint8_t>& ixrq, size_t nb_bits) {
    const ParsedFlat parsed = parse_flat(ixrq);
    EXPECT_EQ(parsed.fourcc_h, faiss::fourcc("Ixrq"));
    std::vector<uint8_t> out(
            ixrq.begin(), ixrq.begin() + static_cast<std::ptrdiff_t>(parsed.codes_len_off));
    out.insert(
            out.end(),
            reinterpret_cast<const uint8_t*>(&nb_bits),
            reinterpret_cast<const uint8_t*>(&nb_bits) + sizeof(nb_bits));
    out.insert(out.end(), ixrq.begin() + static_cast<std::ptrdiff_t>(parsed.codes_len_off), ixrq.end());
    const uint32_t h = faiss::fourcc("Ixrr");
    write_at(out, 0, &h, sizeof(h));
    const ParsedFlat check = parse_flat(out);
    EXPECT_EQ(check.nb_bits, nb_bits);
    EXPECT_TRUE(check.has_nb_bits);
    return out;
}

void use_index(faiss::Index& index, Use use) {
    std::vector<float> query(static_cast<size_t>(index.d), 0.3f);
    if (use == Use::Search) {
        std::vector<float> distances(1);
        std::vector<faiss::idx_t> labels(1);
        index.search(1, query.data(), 1, distances.data(), labels.data());
        return;
    }
    if (use == Use::RangeSearch) {
        faiss::RangeSearchResult result(1);
        index.range_search(1, query.data(), 1.0e30f, &result);
        return;
    }
    if (use == Use::Reconstruct) {
        std::vector<float> reconstructed(static_cast<size_t>(index.d));
        index.reconstruct(0, reconstructed.data());
        return;
    }
    std::vector<float> row(static_cast<size_t>(index.d), 0.4f);
    index.add(1, row.data());
}

// Success is only a refusal from read_index itself. An index that comes
// back is used immediately, so a check that does not apply to the object
// that was returned still fails this test.
void expect_rejected(const std::vector<uint8_t>& payload, Use use) {
    faiss::VectorIOReader reader;
    reader.data = payload;
    std::unique_ptr<faiss::Index> index;
    try {
        index = faiss::read_index_up(&reader);
    } catch (const faiss::FaissException& ex) {
        EXPECT_EQ(std::string(ex.what()).find("read error"), std::string::npos)
                << ex.what();
        return;
    }
    ASSERT_NE(index, nullptr);
    use_index(*index, use);
    FAIL() << "read_index returned a RaBitQ index whose code layout does not "
              "match the dimension, bit width, and vector count";
}

void expect_round_trip(const std::vector<uint8_t>& payload, size_t nb_bits) {
    const ParsedFlat parsed = parse_flat(payload);
    faiss::RaBitQuantizer quantizer(kDim, faiss::METRIC_L2, nb_bits);
    EXPECT_EQ(parsed.code_size, quantizer.code_size);
    EXPECT_EQ(parsed.codes_len, static_cast<size_t>(parsed.ntotal) * quantizer.code_size);
    faiss::VectorIOReader reader;
    reader.data = payload;
    std::unique_ptr<faiss::Index> index;
    ASSERT_NO_THROW(index = faiss::read_index_up(&reader));
    ASSERT_NE(index, nullptr);
    EXPECT_EQ(reader.rp, reader.data.size());
    ASSERT_NO_THROW(use_index(*index, Use::Search));
    ASSERT_NO_THROW(use_index(*index, Use::RangeSearch));
    if (parsed.ntotal > 0) {
        ASSERT_NO_THROW(use_index(*index, Use::Reconstruct));
    }
}

size_t sign_bytes() {
    return (static_cast<size_t>(kDim) + 7) / 8;
}

class RaBitQCodeLayout : public ::testing::Test {
   protected:
    void SetUp() override {
        omp_set_num_threads(1);
    }
};

} // namespace

TEST_F(RaBitQCodeLayout, ValidIxrqRoundTrip) {
    expect_round_trip(serialize_flat(1, /*nb_bits=*/1, /*qb=*/0), 1);
}

TEST_F(RaBitQCodeLayout, ValidIxrrMultiBitRoundTrip) {
    expect_round_trip(serialize_flat(1, /*nb_bits=*/2, /*qb=*/0), 2);
}

TEST_F(RaBitQCodeLayout, IxrqCodeSizeOneSearch) {
    const auto payload = with_codes(serialize_flat(1, 1, 0), 1, {0x00});
    expect_rejected(payload, Use::Search);
}

TEST_F(RaBitQCodeLayout, IxrqCodeSizeOneQuantizedQuerySearch) {
    // qb=4 is the distance computer used when the file keeps the default.
    const auto payload = with_codes(serialize_flat(1, 1, 4), 1, {0x00});
    expect_rejected(payload, Use::Search);
}

TEST_F(RaBitQCodeLayout, IxrqSignBytesOnlyReadsFactorsPastBuffer) {
    // The codes vector is exactly the sign-bit prefix and matches
    // ntotal * stored code_size. The sign-bit factors live at offset
    // (d + 7) / 8, past that allocation.
    const std::vector<uint8_t> codes(sign_bytes(), 0);
    const auto payload = with_codes(serialize_flat(1, 1, 0), sign_bytes(), codes);
    expect_rejected(payload, Use::Search);
}

TEST_F(RaBitQCodeLayout, IxrqSignBytesOnlyRangeSearch) {
    const std::vector<uint8_t> codes(sign_bytes(), 0);
    const auto payload = with_codes(serialize_flat(1, 1, 0), sign_bytes(), codes);
    expect_rejected(payload, Use::RangeSearch);
}

TEST_F(RaBitQCodeLayout, IxrqSignBytesOnlyReconstruct) {
    const std::vector<uint8_t> codes(sign_bytes(), 0);
    const auto payload = with_codes(serialize_flat(1, 1, 0), sign_bytes(), codes);
    expect_rejected(payload, Use::Reconstruct);
}

TEST_F(RaBitQCodeLayout, IxrqShortCodeBuffer) {
    // Stored code size matches the 1-bit layout, but the buffer holds one
    // vector while ntotal is two, so the second code pointer is past the end.
    faiss::RaBitQuantizer quantizer(kDim, faiss::METRIC_L2, 1);
    std::vector<uint8_t> one_code(quantizer.code_size, 0);
    const auto payload = with_codes(serialize_flat(2, 1, 0), quantizer.code_size, one_code);
    expect_rejected(payload, Use::Search);
}

TEST_F(RaBitQCodeLayout, IxrqEmptyIndexShortCodeSizeAdd) {
    // ntotal * code_size equals the empty buffer, so a length check against
    // the stored code size accepts it. add() then writes a full code into
    // that short stride.
    const auto payload = with_codes(serialize_flat(0, 1, 0), 1, {});
    expect_rejected(payload, Use::Add);
}

TEST_F(RaBitQCodeLayout, IxrrBitWidthZeroSearch) {
    auto payload = as_ixrr(serialize_flat(1, 1, 0), /*nb_bits=*/0);
    payload = with_codes(payload, 1, {0x00});
    expect_rejected(payload, Use::Search);
}

TEST_F(RaBitQCodeLayout, IxrrBitWidthTenSearch) {
    auto payload = as_ixrr(serialize_flat(1, 1, 0), /*nb_bits=*/10);
    payload = with_codes(payload, 1, {0x00});
    expect_rejected(payload, Use::Search);
}

TEST_F(RaBitQCodeLayout, IxrrBitWidthTenConsistentBuffer) {
    // Bit width 10 is outside 1..9 even when the buffer matches the same
    // byte formula the distance computer uses for that width.
    const size_t ex_bits = 9;
    const size_t code_size = (static_cast<size_t>(kDim) + 7) / 8 +
            sizeof(faiss::rabitq_utils::SignBitFactorsWithError) +
            (static_cast<size_t>(kDim) * ex_bits + 7) / 8 +
            sizeof(faiss::rabitq_utils::ExtraBitsFactors);
    auto payload = as_ixrr(serialize_flat(1, 1, 0), /*nb_bits=*/10);
    payload = with_codes(payload, code_size, std::vector<uint8_t>(code_size, 0));
    expect_rejected(payload, Use::Search);
}

TEST_F(RaBitQCodeLayout, IxrrMultiBitShortCodeSize) {
    faiss::RaBitQuantizer one_bit(kDim, faiss::METRIC_L2, 1);
    auto payload = as_ixrr(serialize_flat(1, 1, 0), /*nb_bits=*/2);
    payload = with_codes(
            payload, one_bit.code_size, std::vector<uint8_t>(one_bit.code_size, 0));
    expect_rejected(payload, Use::Search);
}

TEST_F(RaBitQCodeLayout, IxrrMultiBitShortBuffer) {
    faiss::RaBitQuantizer two_bit(kDim, faiss::METRIC_L2, 2);
    const auto original = serialize_flat(1, 2, 0);
    const ParsedFlat parsed = parse_flat(original);
    EXPECT_EQ(parsed.fourcc_h, faiss::fourcc("Ixrr"));
    EXPECT_EQ(parsed.nb_bits, static_cast<size_t>(2));
    EXPECT_EQ(parsed.code_size, two_bit.code_size);
    const auto payload = with_codes(original, two_bit.code_size, {0x00});
    expect_rejected(payload, Use::Search);
}
