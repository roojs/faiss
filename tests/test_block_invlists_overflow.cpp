/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A block inverted list whose n_per_block is size_t max wraps
// (n_ids + n_per_block - 1) to 0, so an empty code array matches the
// expected size while the id list is non-empty. These tests require
// read_index / read_InvertedLists to reject that payload. Until that
// happens, search of the loaded index reads the null code pointer and
// the test process faults.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>
#include <faiss/invlists/BlockInvertedLists.h>
#include <faiss/utils/Heap.h>

namespace {

constexpr int kDim = 4;

std::vector<float> toy_vectors(size_t n, int d) {
    std::vector<float> x(n * static_cast<size_t>(d));
    for (size_t i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            x[i * static_cast<size_t>(d) + j] =
                    0.1f * static_cast<float>(i + 1) + static_cast<float>(j);
        }
    }
    return x;
}

// n_per_block at the top of size_t, two ids, and an empty code array.
// block_size is at least 1. The on-disk layout is the one BlockInvertedLists
// writes and the default read path consumes.
std::vector<uint8_t> wrapping_block_list(size_t nlist, size_t block_size) {
    faiss::BlockInvertedLists lists;
    lists.nlist = nlist;
    lists.code_size = faiss::InvertedLists::INVALID_CODE_SIZE;
    lists.n_per_block = std::numeric_limits<size_t>::max();
    lists.block_size = block_size;
    lists.ids.assign(nlist, {});
    lists.codes.assign(nlist, {});
    lists.ids[0].push_back(0);
    lists.ids[0].push_back(1);

    faiss::VectorIOWriter writer;
    faiss::write_InvertedLists(&lists, &writer);
    return writer.data;
}

// Replace the inverted-list suffix of a real index with `replacement`.
// IndexIVFFlat and IndexIVFPQFastScan both write that section last.
std::vector<uint8_t> splice_invlists(
        const faiss::Index& index,
        const faiss::InvertedLists* invlists,
        const std::vector<uint8_t>& replacement) {
    faiss::VectorIOWriter full;
    faiss::write_index(&index, &full);
    faiss::VectorIOWriter lists;
    faiss::write_InvertedLists(invlists, &lists);
    if (full.data.size() < lists.data.size() ||
        !std::equal(
                lists.data.begin(),
                lists.data.end(),
                full.data.begin() +
                        static_cast<std::ptrdiff_t>(
                                full.data.size() - lists.data.size()))) {
        throw std::runtime_error(
                "inverted lists are not a suffix of the serialized index");
    }
    const size_t off = full.data.size() - lists.data.size();
    full.data.resize(off);
    full.data.insert(full.data.end(), replacement.begin(), replacement.end());
    return full.data;
}

// Fixed behavior is a FaissException from the reader. If the reader accepts
// the bytes, `query` runs against that index and must not return: the scan
// reads codes from a null block pointer.
template <class Query>
void require_reader_rejection(std::vector<uint8_t> bytes, Query&& query) {
    faiss::VectorIOReader reader;
    reader.data = std::move(bytes);
    std::unique_ptr<faiss::Index> index;
    try {
        index.reset(faiss::read_index(&reader));
    } catch (const faiss::FaissException&) {
        return;
    }
    ASSERT_NE(index, nullptr);
    query(*index);
    FAIL() << "read_index accepted a block inverted list whose code buffer "
              "does not cover its ids";
}

void train_ivf_flat(faiss::IndexIVFFlat& index) {
    index.cp.niter = 1;
    index.nprobe = 1;
    auto xb = toy_vectors(16, kDim);
    index.train(static_cast<faiss::idx_t>(16), xb.data());
    index.add(static_cast<faiss::idx_t>(16), xb.data());
}

} // namespace

TEST(BlockInvlistsOverflow, ConsistentBlockListStillLoads) {
    // A list whose code bytes match the rounded block count is unchanged.
    faiss::IndexFlatL2 quantizer(8);
    faiss::IndexIVFPQFastScan index(
            &quantizer, 8, /*nlist=*/1, /*M=*/2, /*nbits=*/4);
    index.cp.niter = 1;
    index.pq.cp.niter = 1;
    index.nprobe = 1;
    auto xb = toy_vectors(64, 8);
    index.train(static_cast<faiss::idx_t>(xb.size() / 8), xb.data());
    index.add(static_cast<faiss::idx_t>(xb.size() / 8), xb.data());

    faiss::VectorIOWriter writer;
    faiss::write_index(&index, &writer);
    faiss::VectorIOReader reader;
    reader.data = writer.data;
    std::unique_ptr<faiss::Index> loaded(faiss::read_index(&reader));
    ASSERT_NE(loaded, nullptr);

    std::vector<float> q(8, 0.2f);
    std::vector<float> dist(1);
    std::vector<faiss::idx_t> ids(1, -1);
    loaded->search(1, q.data(), 1, dist.data(), ids.data());
    EXPECT_GE(ids[0], 0);
}

TEST(BlockInvlistsOverflow, ShortCodeBufferIsAlreadyRejected) {
    // Without a wrap, two ids and an empty code array fail the size check.
    faiss::BlockInvertedLists lists;
    lists.nlist = 1;
    lists.code_size = faiss::InvertedLists::INVALID_CODE_SIZE;
    lists.n_per_block = 32;
    lists.block_size = 32;
    lists.ids.resize(1);
    lists.codes.resize(1);
    lists.ids[0] = {0, 1};

    faiss::VectorIOWriter writer;
    faiss::write_InvertedLists(&lists, &writer);
    faiss::VectorIOReader reader;
    reader.data = writer.data;
    EXPECT_THROW(
            {
                std::unique_ptr<faiss::InvertedLists> loaded(
                        faiss::read_InvertedLists(&reader));
            },
            faiss::FaissException);
}

TEST(BlockInvlistsOverflow, LoaderRejectsWrappedBlockCount) {
    auto bytes = wrapping_block_list(/*nlist=*/1, /*block_size=*/1);
    faiss::VectorIOReader reader;
    reader.data = std::move(bytes);
    try {
        std::unique_ptr<faiss::InvertedLists> loaded(
                faiss::read_InvertedLists(&reader));
        ADD_FAILURE()
                << "BlockInvertedLists reader accepted n_per_block=SIZE_MAX, "
                   "two ids, and an empty code array";
        (void)loaded;
    } catch (const faiss::FaissException&) {
    }
}

TEST(BlockInvlistsOverflow, IvfFlatSearchRejectsWrappedBlockCount) {
    faiss::IndexFlatL2 quantizer(kDim);
    faiss::IndexIVFFlat index(&quantizer, kDim, /*nlist=*/1);
    train_ivf_flat(index);
    auto bytes = splice_invlists(
            index, index.invlists, wrapping_block_list(index.nlist, 1));

    require_reader_rejection(std::move(bytes), [](faiss::Index& loaded) {
        std::vector<float> q(kDim, 0.25f);
        float dist = 0;
        faiss::idx_t id = -1;
        loaded.search(1, q.data(), 1, &dist, &id);
    });
}

TEST(BlockInvlistsOverflow, IvfFlatRangeSearchRejectsWrappedBlockCount) {
    faiss::IndexFlatL2 quantizer(kDim);
    faiss::IndexIVFFlat index(&quantizer, kDim, /*nlist=*/1);
    train_ivf_flat(index);
    auto bytes = splice_invlists(
            index, index.invlists, wrapping_block_list(index.nlist, 1));

    require_reader_rejection(std::move(bytes), [](faiss::Index& loaded) {
        std::vector<float> q(kDim, 0.25f);
        faiss::RangeSearchResult result(1);
        loaded.range_search(1, q.data(), 1.0e10f, &result);
    });
}

TEST(BlockInvlistsOverflow, IvfFlatSearch1RejectsWrappedBlockCount) {
    faiss::IndexFlatL2 quantizer(kDim);
    faiss::IndexIVFFlat index(&quantizer, kDim, /*nlist=*/1);
    train_ivf_flat(index);
    auto bytes = splice_invlists(
            index, index.invlists, wrapping_block_list(index.nlist, 1));

    require_reader_rejection(std::move(bytes), [](faiss::Index& loaded) {
        auto* ivf = dynamic_cast<faiss::IndexIVF*>(&loaded);
        ASSERT_NE(ivf, nullptr);
        std::vector<float> q(kDim, 0.25f);
        float dist = 1.0e30f;
        faiss::idx_t id = -1;
        faiss::heap_heapify<faiss::CMax<float, faiss::idx_t>>(1, &dist, &id);
        faiss::HeapResultHandler<faiss::CMax<float, faiss::idx_t>, false>
                handler(1, &dist, &id);
        ivf->search1(q.data(), handler, nullptr);
    });
}

TEST(BlockInvlistsOverflow, FastScanSearchRejectsWrappedBlockCount) {
    faiss::IndexFlatL2 quantizer(8);
    faiss::IndexIVFPQFastScan index(
            &quantizer, 8, /*nlist=*/1, /*M=*/2, /*nbits=*/4);
    index.cp.niter = 1;
    index.pq.cp.niter = 1;
    index.nprobe = 1;
    // Default implem 0 selects the qbs scanner (implem 12) when bbs == 32.
    auto xb = toy_vectors(64, 8);
    index.train(static_cast<faiss::idx_t>(xb.size() / 8), xb.data());
    index.add(static_cast<faiss::idx_t>(xb.size() / 8), xb.data());

    auto bytes = splice_invlists(
            index,
            index.invlists,
            wrapping_block_list(index.nlist, /*block_size=*/32));

    require_reader_rejection(std::move(bytes), [](faiss::Index& loaded) {
        std::vector<float> q(8, 0.2f);
        float dist = 0;
        faiss::idx_t id = -1;
        loaded.search(1, q.data(), 1, &dist, &id);
    });
}

TEST(BlockInvlistsOverflow, FastScanImplem10RejectsWrappedBlockCount) {
    faiss::IndexFlatL2 quantizer(8);
    faiss::IndexIVFPQFastScan index(
            &quantizer, 8, /*nlist=*/1, /*M=*/2, /*nbits=*/4);
    index.cp.niter = 1;
    index.pq.cp.niter = 1;
    index.nprobe = 1;
    index.implem = 10;
    auto xb = toy_vectors(64, 8);
    index.train(static_cast<faiss::idx_t>(xb.size() / 8), xb.data());
    index.add(static_cast<faiss::idx_t>(xb.size() / 8), xb.data());

    auto bytes = splice_invlists(
            index,
            index.invlists,
            wrapping_block_list(index.nlist, /*block_size=*/32));

    require_reader_rejection(std::move(bytes), [](faiss::Index& loaded) {
        std::vector<float> q(8, 0.2f);
        float dist = 0;
        faiss::idx_t id = -1;
        loaded.search(1, q.data(), 1, &dist, &id);
    });
}
