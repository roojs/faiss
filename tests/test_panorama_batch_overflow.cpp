/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Spec for the ilp2 batch-size overflow in read_index.
//
// These tests fail while the overflow is present: read_index accepts a
// crafted IwP2 file, search then throws on the huge batch allocation, and
// reconstruct faults by copying out of the empty code buffer. They pass
// once read_index rejects that file with FaissException before returning
// an index. A normal IwP2 round-trip must keep working.

#include <gtest/gtest.h>

#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlatPanorama.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>

namespace {

constexpr int kDim = 4;

void append_bytes(std::vector<uint8_t>& buf, const void* p, size_t n) {
    const auto* b = static_cast<const uint8_t*>(p);
    buf.insert(buf.end(), b, b + n);
}

template <typename T>
void append_val(std::vector<uint8_t>& buf, T val) {
    append_bytes(buf, &val, sizeof(T));
}

void append_fourcc(std::vector<uint8_t>& buf, const char s[4]) {
    const auto* x = reinterpret_cast<const unsigned char*>(s);
    uint32_t h = x[0] | (x[1] << 8) | (x[2] << 16) | (x[3] << 24);
    append_val<uint32_t>(buf, h);
}

// write_index_header layout: d, ntotal, two dummy int64s, bool, metric.
void append_index_header(
        std::vector<uint8_t>& buf,
        int d,
        int64_t ntotal,
        bool is_trained,
        int metric_type) {
    append_val<int>(buf, d);
    append_val<int64_t>(buf, ntotal);
    int64_t dummy = 1 << 20;
    append_val<int64_t>(buf, dummy);
    append_val<int64_t>(buf, dummy);
    append_val<uint8_t>(buf, is_trained ? 1 : 0);
    append_val<int>(buf, metric_type);
}

void append_index_flat_l2(
        std::vector<uint8_t>& buf,
        int d,
        const std::vector<float>& vectors) {
    append_fourcc(buf, "IxF2");
    const int64_t ntotal = static_cast<int64_t>(vectors.size() / d);
    append_index_header(buf, d, ntotal, true, faiss::METRIC_L2);
    // WRITEXBVECTOR stores the number of floats, then the raw bytes.
    append_val<size_t>(buf, vectors.size());
    append_bytes(buf, vectors.data(), vectors.size() * sizeof(float));
}

// IwP2 whose ilp2 section has one list of `n_ids` ids, an empty code
// buffer, and an empty cumulative-sum buffer. `list_batch` is the batch
// size stored in the list section; `index_batch` is the field on the
// index header. Those are independent on disk.
std::vector<uint8_t> make_iwp2_payload(
        size_t n_ids,
        size_t list_batch,
        size_t index_batch) {
    std::vector<uint8_t> buf;
    append_fourcc(buf, "IwP2");
    append_index_header(
            buf, kDim, static_cast<int64_t>(n_ids), true, faiss::METRIC_L2);
    append_val<size_t>(buf, 1); // nlist
    append_val<size_t>(buf, 1); // nprobe
    std::vector<float> centroid(kDim, 0.f);
    append_index_flat_l2(buf, kDim, centroid);
    append_val<uint8_t>(buf, 0); // DirectMap::NoMap
    append_val<size_t>(buf, 0);  // empty direct-map array
    append_val<size_t>(buf, 1);  // n_levels
    append_val<size_t>(buf, index_batch);

    append_fourcc(buf, "ilp2");
    append_val<size_t>(buf, 1); // nlist
    append_val<size_t>(buf, static_cast<size_t>(kDim) * sizeof(float));
    append_val<size_t>(buf, 1); // n_levels
    append_val<size_t>(buf, list_batch);
    append_fourcc(buf, "full");
    append_val<size_t>(buf, 1); // one list length
    append_val<size_t>(buf, n_ids);
    // codes and cum_sums are omitted: after the wrap their on-disk
    // length is 0. The id vector is raw idx_t values, not length-prefixed.
    for (size_t i = 0; i < n_ids; i++) {
        append_val<faiss::idx_t>(buf, static_cast<faiss::idx_t>(i));
    }
    return buf;
}

enum class Op { Search, RangeSearch, Reconstruct, SearchAndReconstruct, Add };

const char* op_name(Op op) {
    switch (op) {
        case Op::Search:
            return "search";
        case Op::RangeSearch:
            return "range_search";
        case Op::Reconstruct:
            return "reconstruct_n";
        case Op::SearchAndReconstruct:
            return "search_and_reconstruct";
        case Op::Add:
            return "add";
    }
    return "unknown";
}

// Child status:
//   0  read_index rejected the payload with FaissException
//   2  read_index returned an index and the operation finished
//   3  read_index returned an index and the operation threw
//      bad_alloc or length_error (huge batch allocation)
//   4  read_index returned an index and the operation threw otherwise
//   5  read_index threw something other than FaissException
//   6  read_index returned null
// A signal (reconstruct's out-of-bounds copy) is reported by wait status.
int exercise(const std::vector<uint8_t>& payload, Op op) {
    std::unique_ptr<faiss::Index> idx;
    try {
        faiss::VectorIOReader reader;
        reader.data = payload;
        idx.reset(faiss::read_index(&reader));
    } catch (const faiss::FaissException& e) {
        std::fprintf(stderr, "read_index rejected: %s\n", e.what());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "read_index threw: %s\n", e.what());
        return 5;
    }
    if (!idx) {
        std::fprintf(stderr, "read_index returned null\n");
        return 6;
    }

    try {
        std::vector<float> query(static_cast<size_t>(idx->d), 0.f);
        if (op == Op::Search) {
            std::vector<float> dist(1);
            std::vector<faiss::idx_t> labels(1);
            idx->search(1, query.data(), 1, dist.data(), labels.data());
        } else if (op == Op::RangeSearch) {
            faiss::RangeSearchResult result(1);
            idx->range_search(1, query.data(), 1.0e10f, &result);
        } else if (op == Op::Reconstruct) {
            std::vector<float> recons(
                    static_cast<size_t>(idx->ntotal) *
                    static_cast<size_t>(idx->d));
            idx->reconstruct_n(0, idx->ntotal, recons.data());
        } else if (op == Op::SearchAndReconstruct) {
            auto* ivf = dynamic_cast<faiss::IndexIVF*>(idx.get());
            if (!ivf) {
                std::fprintf(stderr, "loaded index is not IndexIVF\n");
                return 4;
            }
            std::vector<float> dist(1);
            std::vector<faiss::idx_t> labels(1);
            std::vector<float> recons(static_cast<size_t>(idx->d));
            ivf->search_and_reconstruct(
                    1,
                    query.data(),
                    1,
                    dist.data(),
                    labels.data(),
                    recons.data());
        } else {
            idx->add(1, query.data());
        }
    } catch (const std::bad_alloc&) {
        std::fprintf(stderr, "%s threw std::bad_alloc\n", op_name(op));
        return 3;
    } catch (const std::length_error& e) {
        std::fprintf(
                stderr,
                "%s threw std::length_error: %s\n",
                op_name(op),
                e.what());
        return 3;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s threw: %s\n", op_name(op), e.what());
        return 4;
    }
    std::fprintf(
            stderr,
            "read_index accepted the crafted index and %s returned\n",
            op_name(op));
    return 2;
}

std::string status_text(int status) {
    if (WIFEXITED(status)) {
        return "exit " + std::to_string(WEXITSTATUS(status));
    }
    if (WIFSIGNALED(status)) {
        return std::string("signal ") + strsignal(WTERMSIG(status));
    }
    return "status " + std::to_string(status);
}

// The payload must be refused. A crash or an allocation exception in the
// child means the loader still handed back the crafted index.
void expect_rejected(const std::vector<uint8_t>& payload, Op op) {
    const pid_t pid = fork();
    ASSERT_NE(pid, -1) << std::strerror(errno);
    if (pid == 0) {
        const int code = exercise(payload, op);
        std::fflush(stderr);
        _exit(code);
    }
    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0)
            << op_name(op) << " on crafted IwP2: " << status_text(status)
            << " (0 means read_index rejected the file)";
}

} // namespace

TEST(PanoramaBatchOverflow, SearchRejectsSizeMaxBatch) {
    const size_t batch = std::numeric_limits<size_t>::max();
    expect_rejected(make_iwp2_payload(2, batch, batch), Op::Search);
}

TEST(PanoramaBatchOverflow, ReconstructRejectsSizeMaxBatch) {
    const size_t batch = std::numeric_limits<size_t>::max();
    expect_rejected(make_iwp2_payload(2, batch, batch), Op::Reconstruct);
}

TEST(PanoramaBatchOverflow, RangeSearchRejectsSizeMaxBatch) {
    const size_t batch = std::numeric_limits<size_t>::max();
    expect_rejected(make_iwp2_payload(2, batch, batch), Op::RangeSearch);
}

TEST(PanoramaBatchOverflow, SearchAndReconstructRejectsSizeMaxBatch) {
    const size_t batch = std::numeric_limits<size_t>::max();
    expect_rejected(
            make_iwp2_payload(2, batch, batch), Op::SearchAndReconstruct);
}

// n + batch - 1 also wraps for batch sizes just below size_t's maximum
// when the list holds more than one id. Rejecting only size_t's maximum
// leaves this payload in place.
TEST(PanoramaBatchOverflow, SearchRejectsBatchJustBelowSizeMax) {
    const size_t batch = std::numeric_limits<size_t>::max() - 1;
    expect_rejected(make_iwp2_payload(3, batch, batch), Op::Search);
}

TEST(PanoramaBatchOverflow, ReconstructRejectsBatchJustBelowSizeMax) {
    const size_t batch = std::numeric_limits<size_t>::max() - 1;
    expect_rejected(make_iwp2_payload(3, batch, batch), Op::Reconstruct);
}

// The scanner and reconstruct read the list section's batch size, not the
// batch size field on the IwP2 header. A check of only the header field
// still accepts this file.
TEST(PanoramaBatchOverflow, SearchRejectsHugeListBatchWithSmallHeaderBatch) {
    const size_t list_batch = std::numeric_limits<size_t>::max();
    expect_rejected(make_iwp2_payload(2, list_batch, 32), Op::Search);
}

TEST(PanoramaBatchOverflow, ReconstructRejectsHugeListBatchWithSmallHeaderBatch) {
    const size_t list_batch = std::numeric_limits<size_t>::max();
    expect_rejected(make_iwp2_payload(2, list_batch, 32), Op::Reconstruct);
}

// add() rounds the list with the same unchecked (n + batch - 1) expression
// the loader uses. Once the crafted file has been accepted, inserting one
// vector copies into the empty code buffer.
TEST(PanoramaBatchOverflow, AddRejectsSizeMaxBatch) {
    const size_t batch = std::numeric_limits<size_t>::max();
    expect_rejected(make_iwp2_payload(2, batch, batch), Op::Add);
}

TEST(PanoramaBatchOverflow, LegitimateIndexRoundTrips) {
    constexpr int d = kDim;
    auto* quantizer = new faiss::IndexFlatL2(d);
    std::vector<float> centroid(d, 0.f);
    quantizer->add(1, centroid.data());
    faiss::IndexIVFFlatPanorama index(
            quantizer, d, 1, 1, faiss::METRIC_L2, true, 32);
    ASSERT_TRUE(index.is_trained);

    std::vector<float> xb(static_cast<size_t>(d) * 4);
    for (size_t i = 0; i < xb.size(); i++) {
        xb[i] = static_cast<float>(i);
    }
    index.add(4, xb.data());

    faiss::VectorIOWriter writer;
    faiss::write_index(&index, &writer);
    faiss::VectorIOReader reader;
    reader.data = writer.data;
    std::unique_ptr<faiss::Index> loaded(faiss::read_index(&reader));
    ASSERT_NE(loaded, nullptr);

    std::vector<float> dist(1);
    std::vector<faiss::idx_t> labels(1, -1);
    loaded->search(1, centroid.data(), 1, dist.data(), labels.data());
    EXPECT_GE(labels[0], 0);
    EXPECT_LT(labels[0], 4);

    std::vector<float> recons(static_cast<size_t>(loaded->ntotal) * d);
    loaded->reconstruct_n(0, loaded->ntotal, recons.data());
    EXPECT_EQ(recons.size(), xb.size());
}
