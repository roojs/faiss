/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A RemapDimensionsTransform stored in an index file is loaded through the
// default constructor. read_VectorTransform checks only that the map is at
// least d_out long, then search and reconstruct use each entry as a
// subscript. These tests require the loader and every operation that indexes
// the map to reject an entry other than -1 or [0, d_in) before they read or
// write past the caller buffer.
//
// Not invoked directly: IndexPreTransform::train never applies a remap (an
// untrained remap is trained with the empty VectorTransform::train and the
// loop stops before apply). Index::assign calls search. The C API
// faiss_read_index* wrappers call faiss::read_index.
// IndexIVFIndependentQuantizer::train calls VectorTransform::apply, which is
// covered, after clustering.

#include <gtest/gtest.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFIndependentQuantizer.h>
#include <faiss/IndexIVFSpectralHash.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>

namespace {

constexpr int kDim = 4;
constexpr int kLargeEntry = 1 << 30;

template <typename T>
struct Watched {
    std::vector<T> values;
    T fill;

    Watched(size_t n, T fill_in) : values(n, fill_in), fill(fill_in) {}

    T* data() {
        return values.data();
    }

    bool unchanged() const {
        for (T value : values) {
            if (value != fill) {
                return false;
            }
        }
        return true;
    }
};

// nfloats valid floats placed so element nfloats is the first byte of a
// PROT_NONE page. A subscript of nfloats or a few past it faults.
class GuardedFloats {
   public:
    explicit GuardedFloats(int nfloats) : nfloats_(nfloats) {
        page_ = sysconf(_SC_PAGESIZE);
        if (page_ < nfloats * static_cast<long>(sizeof(float))) {
            throw std::runtime_error("guard page smaller than vector");
        }
        mem_ = mmap(
                nullptr,
                static_cast<size_t>(page_) * 2,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0);
        if (mem_ == MAP_FAILED) {
            throw std::runtime_error("mmap failed");
        }
        if (mprotect(
                    static_cast<char*>(mem_) + page_,
                    static_cast<size_t>(page_),
                    PROT_NONE) != 0) {
            throw std::runtime_error("mprotect failed");
        }
        data_ = reinterpret_cast<float*>(
                static_cast<char*>(mem_) + page_ -
                nfloats * static_cast<long>(sizeof(float)));
        if (reinterpret_cast<char*>(data_ + nfloats_) !=
            static_cast<char*>(mem_) + page_) {
            throw std::runtime_error("guard page is not flush with the vector");
        }
        for (int i = 0; i < nfloats_; i++) {
            data_[i] = 0.25f * static_cast<float>(i + 1);
        }
        snapshot_.assign(data_, data_ + nfloats_);
    }

    GuardedFloats(const GuardedFloats&) = delete;
    GuardedFloats& operator=(const GuardedFloats&) = delete;

    ~GuardedFloats() {
        if (mem_ != nullptr && mem_ != MAP_FAILED) {
            munmap(mem_, static_cast<size_t>(page_) * 2);
        }
    }

    float* data() {
        return data_;
    }

    bool unchanged() const {
        for (int i = 0; i < nfloats_; i++) {
            if (data_[i] != snapshot_[i]) {
                return false;
            }
        }
        return true;
    }

   private:
    int nfloats_ = 0;
    long page_ = 0;
    void* mem_ = nullptr;
    float* data_ = nullptr;
    std::vector<float> snapshot_;
};

template <typename Fn>
std::string run_child(Fn&& fn) {
    const pid_t pid = fork();
    if (pid < 0) {
        return "fork failed";
    }
    if (pid == 0) {
        try {
            fn();
            _exit(1);
        } catch (...) {
            _exit(2);
        }
    }
    int status = 0;
    pid_t waited = -1;
    do {
        waited = waitpid(pid, &status, 0);
    } while (waited < 0 && errno == EINTR);
    if (waited < 0) {
        return "waitpid failed";
    }
    if (WIFSIGNALED(status)) {
        return "out-of-bounds access (signal " +
                std::to_string(WTERMSIG(status)) + ")";
    }
    if (!WIFEXITED(status)) {
        return "child did not exit";
    }
    switch (WEXITSTATUS(status)) {
        case 0:
            return "";
        case 1:
            return "operation completed without rejecting the out-of-range map entry";
        case 2:
            return "unexpected exception";
        case 3:
            return "caller-visible buffer or index was mutated before refusal";
        case 4:
            return "valid remap behavior changed";
        case 5:
            return "test precondition failed";
        default:
            return "unexpected child status " +
                    std::to_string(WEXITSTATUS(status));
    }
}

template <typename Fn>
void expect_child_refuses(Fn&& fn) {
    const std::string failure = run_child(std::forward<Fn>(fn));
    EXPECT_TRUE(failure.empty()) << failure;
}

bool contains_i32(const std::vector<uint8_t>& bytes, int needle) {
    if (bytes.size() < sizeof(int)) {
        return false;
    }
    for (size_t i = 0; i + sizeof(int) <= bytes.size(); i++) {
        int value = 0;
        memcpy(&value, bytes.data() + i, sizeof(value));
        if (value == needle) {
            return true;
        }
    }
    return false;
}

struct BuiltPreTransform {
    faiss::IndexFlatL2 flat;
    int entries[kDim];
    faiss::RemapDimensionsTransform remap;
    faiss::IndexPreTransform pre;

    BuiltPreTransform()
            : flat(kDim),
              entries{0, 1, 2, 3},
              remap(kDim, kDim, entries),
              pre(&remap, &flat) {
        std::vector<float> row(kDim);
        for (int i = 0; i < kDim; i++) {
            row[i] = static_cast<float>(i + 1);
        }
        pre.add(1, row.data());
    }
};

std::vector<uint8_t> poisoned_pretransform_bytes(int hostile) {
    BuiltPreTransform built;
    built.remap.map[kDim - 1] = hostile;
    faiss::VectorIOWriter writer;
    faiss::write_index(&built.pre, &writer);
    if (!contains_i32(writer.data, hostile)) {
        _exit(5);
    }
    return writer.data;
}

void require_hostile_pretransform(faiss::Index* idx, int hostile) {
    auto* pre = dynamic_cast<faiss::IndexPreTransform*>(idx);
    if (pre == nullptr || pre->chain.size() != 1) {
        _exit(5);
    }
    auto* remap =
            dynamic_cast<faiss::RemapDimensionsTransform*>(pre->chain[0]);
    if (remap == nullptr) {
        _exit(5);
    }
    for (int entry : remap->map) {
        if (entry == hostile) {
            return;
        }
    }
    _exit(5);
}

std::unique_ptr<faiss::Index> load_or_refuse(
        const std::vector<uint8_t>& bytes) {
    faiss::VectorIOReader reader;
    reader.data = bytes;
    try {
        return std::unique_ptr<faiss::Index>(faiss::read_index(&reader));
    } catch (const faiss::FaissException&) {
        _exit(0);
    }
}

void refuse_on_faiss(bool unchanged) {
    // Reached only when the operation threw faiss::FaissException.
    _exit(unchanged ? 0 : 3);
}

} // namespace

TEST(RemapDimensionsBounds, ValidSentinelRoundTripsThroughLoad) {
    const std::string failure = run_child([] {
        constexpr int kMap[kDim] = {2, -1, 0, 1};
        faiss::IndexFlatL2 flat(kDim);
        faiss::RemapDimensionsTransform remap(kDim, kDim, kMap);
        faiss::IndexPreTransform pre(&remap, &flat);
        const float row[kDim] = {10.f, 20.f, 30.f, 40.f};
        pre.add(1, row);

        faiss::VectorIOWriter writer;
        faiss::write_index(&pre, &writer);
        faiss::VectorIOReader reader;
        reader.data = writer.data;
        std::unique_ptr<faiss::Index> loaded(faiss::read_index(&reader));
        float reconstructed[kDim] = {};
        loaded->reconstruct(0, reconstructed);
        const float expect[kDim] = {10.f, 20.f, 30.f, 0.f};
        for (int i = 0; i < kDim; i++) {
            if (reconstructed[i] != expect[i]) {
                _exit(4);
            }
        }
        float distance = -1.f;
        faiss::idx_t label = -1;
        loaded->search(1, row, 1, &distance, &label);
        if (label != 0 || distance != 0.f) {
            _exit(4);
        }
        _exit(0);
    });
    EXPECT_TRUE(failure.empty()) << failure;
}

TEST(RemapDimensionsBounds, ConstructorRejectsEntryEqualToDimension) {
    const std::string failure = run_child([] {
        const int bad[1] = {kDim};
        try {
            faiss::RemapDimensionsTransform remap(kDim, 1, bad);
            (void)remap;
        } catch (const faiss::FaissException&) {
            _exit(0);
        }
    });
    EXPECT_TRUE(failure.empty()) << failure;
}

TEST(RemapDimensionsBounds, LoadRejectsLargeMapEntry) {
    expect_child_refuses([] {
        const std::vector<uint8_t> bytes =
                poisoned_pretransform_bytes(kLargeEntry);
        faiss::VectorIOReader reader;
        reader.data = bytes;
        try {
            std::unique_ptr<faiss::Index> idx(faiss::read_index(&reader));
        } catch (const faiss::FaissException&) {
            _exit(0);
        }
    });
}

TEST(RemapDimensionsBounds, LoadRejectsEntryEqualToInputDimension) {
    expect_child_refuses([] {
        const std::vector<uint8_t> bytes = poisoned_pretransform_bytes(kDim);
        faiss::VectorIOReader reader;
        reader.data = bytes;
        try {
            std::unique_ptr<faiss::Index> idx(faiss::read_index(&reader));
        } catch (const faiss::FaissException&) {
            _exit(0);
        }
    });
}

TEST(RemapDimensionsBounds, LoadRejectsEntryBelowSentinel) {
    expect_child_refuses([] {
        const std::vector<uint8_t> bytes = poisoned_pretransform_bytes(-2);
        faiss::VectorIOReader reader;
        reader.data = bytes;
        try {
            std::unique_ptr<faiss::Index> idx(faiss::read_index(&reader));
        } catch (const faiss::FaissException&) {
            _exit(0);
        }
    });
}

TEST(RemapDimensionsBounds, LoadRejectsOutOfRangeTailEntry) {
    expect_child_refuses([] {
        BuiltPreTransform built;
        built.remap.map.push_back(kLargeEntry);
        faiss::VectorIOWriter writer;
        faiss::write_index(&built.pre, &writer);
        if (!contains_i32(writer.data, kLargeEntry)) {
            _exit(5);
        }
        faiss::VectorIOReader reader;
        reader.data = writer.data;
        try {
            std::unique_ptr<faiss::Index> idx(faiss::read_index(&reader));
        } catch (const faiss::FaissException&) {
            _exit(0);
        }
    });
}

TEST(RemapDimensionsBounds, LoadRejectsStandaloneVectorTransform) {
    expect_child_refuses([] {
        BuiltPreTransform built;
        built.remap.map[kDim - 1] = kLargeEntry;
        faiss::VectorIOWriter writer;
        faiss::write_VectorTransform(&built.remap, &writer);
        if (!contains_i32(writer.data, kLargeEntry)) {
            _exit(5);
        }
        faiss::VectorIOReader reader;
        reader.data = writer.data;
        try {
            std::unique_ptr<faiss::VectorTransform> vt(
                    faiss::read_VectorTransform(&reader));
        } catch (const faiss::FaissException&) {
            _exit(0);
        }
    });
}

TEST(RemapDimensionsBounds, LoadRejectsSpectralHashVectorTransform) {
    expect_child_refuses([] {
        faiss::IndexFlatL2 quantizer(kDim);
        std::vector<float> centroid(kDim, 0.2f);
        quantizer.add(1, centroid.data());
        const int entries[kDim] = {0, 1, 2, 3};
        faiss::RemapDimensionsTransform remap(kDim, kDim, entries);
        faiss::IndexIVFSpectralHash spectral(
                &quantizer, kDim, 1, kDim, /*period=*/1.f);
        spectral.replace_vt(&remap, false);
        remap.map[kDim - 1] = kLargeEntry;
        faiss::VectorIOWriter writer;
        faiss::write_index(&spectral, &writer);
        if (!contains_i32(writer.data, kLargeEntry)) {
            _exit(5);
        }
        faiss::VectorIOReader reader;
        reader.data = writer.data;
        try {
            std::unique_ptr<faiss::Index> idx(faiss::read_index(&reader));
        } catch (const faiss::FaissException&) {
            _exit(0);
        }
    });
}

TEST(RemapDimensionsBounds, LoadRejectsIndependentQuantizerVectorTransform) {
    expect_child_refuses([] {
        faiss::IndexFlatL2 coarse(kDim);
        faiss::IndexFlatL2 ivf_quantizer(kDim);
        faiss::IndexIVFFlat ivf(&ivf_quantizer, kDim, 1);
        const int entries[kDim] = {0, 1, 2, 3};
        faiss::RemapDimensionsTransform remap(kDim, kDim, entries);
        faiss::IndexIVFIndependentQuantizer indep(&coarse, &ivf, &remap);
        remap.map[kDim - 1] = kLargeEntry;
        faiss::VectorIOWriter writer;
        faiss::write_index(&indep, &writer);
        if (!contains_i32(writer.data, kLargeEntry)) {
            _exit(5);
        }
        faiss::VectorIOReader reader;
        reader.data = writer.data;
        try {
            std::unique_ptr<faiss::Index> idx(faiss::read_index(&reader));
        } catch (const faiss::FaissException&) {
            _exit(0);
        }
    });
}

TEST(RemapDimensionsBounds, LoadedSearchReadsPastQuery) {
    expect_child_refuses([] {
        std::unique_ptr<faiss::Index> idx =
                load_or_refuse(poisoned_pretransform_bytes(kDim));
        require_hostile_pretransform(idx.get(), kDim);
        GuardedFloats query(kDim);
        Watched<float> distances(1, -1234.f);
        Watched<faiss::idx_t> labels(1, static_cast<faiss::idx_t>(-777));
        try {
            idx->search(
                    1,
                    query.data(),
                    1,
                    distances.data(),
                    labels.data());
        } catch (const faiss::FaissException&) {
            refuse_on_faiss(
                    query.unchanged() && distances.unchanged() &&
                    labels.unchanged());
        }
    });
}

TEST(RemapDimensionsBounds, LoadedReconstructWritesPastBuffer) {
    expect_child_refuses([] {
        std::unique_ptr<faiss::Index> idx =
                load_or_refuse(poisoned_pretransform_bytes(kDim));
        require_hostile_pretransform(idx.get(), kDim);
        GuardedFloats reconstructed(kDim);
        try {
            idx->reconstruct(0, reconstructed.data());
        } catch (const faiss::FaissException&) {
            refuse_on_faiss(reconstructed.unchanged());
        }
    });
}

TEST(RemapDimensionsBounds, LoadedSearchAndReconstructReadsPastQuery) {
    expect_child_refuses([] {
        std::unique_ptr<faiss::Index> idx =
                load_or_refuse(poisoned_pretransform_bytes(kDim));
        require_hostile_pretransform(idx.get(), kDim);
        GuardedFloats query(kDim);
        Watched<float> distances(1, -1234.f);
        Watched<faiss::idx_t> labels(1, static_cast<faiss::idx_t>(-777));
        Watched<float> reconstructed(kDim, 42.f);
        try {
            idx->search_and_reconstruct(
                    1,
                    query.data(),
                    1,
                    distances.data(),
                    labels.data(),
                    reconstructed.data());
        } catch (const faiss::FaissException&) {
            refuse_on_faiss(
                    query.unchanged() && distances.unchanged() &&
                    labels.unchanged() && reconstructed.unchanged());
        }
    });
}

TEST(RemapDimensionsBounds, LoadedSaDecodeWritesPastBuffer) {
    expect_child_refuses([] {
        std::unique_ptr<faiss::Index> idx =
                load_or_refuse(poisoned_pretransform_bytes(kDim));
        require_hostile_pretransform(idx.get(), kDim);
        std::vector<uint8_t> code(idx->sa_code_size());
        GuardedFloats decoded(kDim);
        try {
            idx->sa_decode(1, code.data(), decoded.data());
        } catch (const faiss::FaissException&) {
            refuse_on_faiss(decoded.unchanged());
        }
    });
}

TEST(RemapDimensionsBounds, LoadedSearchLargeEntryReadsPastQuery) {
    expect_child_refuses([] {
        std::unique_ptr<faiss::Index> idx =
                load_or_refuse(poisoned_pretransform_bytes(kLargeEntry));
        require_hostile_pretransform(idx.get(), kLargeEntry);
        std::vector<float> query(kDim, 0.5f);
        float distance = -1234.f;
        faiss::idx_t label = -777;
        try {
            idx->search(1, query.data(), 1, &distance, &label);
        } catch (const faiss::FaissException&) {
            refuse_on_faiss(distance == -1234.f && label == -777);
        }
    });
}

TEST(RemapDimensionsBounds, ApplyNoallocRejectsBeforeRead) {
    expect_child_refuses([] {
        const int entries[kDim] = {0, 1, 2, 3};
        faiss::RemapDimensionsTransform remap(kDim, kDim, entries);
        remap.map[kDim - 1] = kDim;
        GuardedFloats input(kDim);
        Watched<float> output(kDim, 42.f);
        try {
            remap.apply_noalloc(1, input.data(), output.data());
        } catch (const faiss::FaissException&) {
            refuse_on_faiss(input.unchanged() && output.unchanged());
        }
    });
}

TEST(RemapDimensionsBounds, ApplyRejectsLargeEntryBeforeRead) {
    expect_child_refuses([] {
        const int entries[kDim] = {0, 1, 2, 3};
        faiss::RemapDimensionsTransform remap(kDim, kDim, entries);
        remap.map[1] = kLargeEntry;
        std::vector<float> input(kDim, 0.5f);
        Watched<float> output(kDim, 42.f);
        try {
            float* transformed = remap.apply(1, input.data());
            delete[] transformed;
            (void)output;
        } catch (const faiss::FaissException&) {
            refuse_on_faiss(output.unchanged());
        }
    });
}

TEST(RemapDimensionsBounds, ApplyNoallocRejectsNonSentinelNegative) {
    expect_child_refuses([] {
        const int entries[kDim] = {0, 1, 2, 3};
        faiss::RemapDimensionsTransform remap(kDim, kDim, entries);
        remap.map[1] = -2;
        std::vector<float> input(kDim, 0.5f);
        Watched<float> output(kDim, 42.f);
        try {
            remap.apply_noalloc(1, input.data(), output.data());
        } catch (const faiss::FaissException&) {
            refuse_on_faiss(output.unchanged());
        }
    });
}

TEST(RemapDimensionsBounds, ReverseTransformRejectsBeforeWrite) {
    expect_child_refuses([] {
        const int entries[kDim] = {0, 1, 2, 3};
        faiss::RemapDimensionsTransform remap(kDim, kDim, entries);
        remap.map[kDim - 1] = kDim;
        std::vector<float> transformed(kDim, 0.5f);
        GuardedFloats output(kDim);
        try {
            remap.reverse_transform(1, transformed.data(), output.data());
        } catch (const faiss::FaissException&) {
            refuse_on_faiss(output.unchanged());
        }
    });
}

TEST(RemapDimensionsBounds, ReverseTransformRejectsNonSentinelNegative) {
    expect_child_refuses([] {
        const int entries[kDim] = {0, 1, 2, 3};
        faiss::RemapDimensionsTransform remap(kDim, kDim, entries);
        remap.map[1] = -2;
        std::vector<float> transformed(kDim, 0.5f);
        Watched<float> output(kDim, 42.f);
        try {
            remap.reverse_transform(1, transformed.data(), output.data());
        } catch (const faiss::FaissException&) {
            refuse_on_faiss(output.unchanged());
        }
    });
}

TEST(RemapDimensionsBounds, PreTransformSearchRejectsBeforeRead) {
    expect_child_refuses([] {
        BuiltPreTransform built;
        built.remap.map[kDim - 1] = kDim;
        GuardedFloats query(kDim);
        Watched<float> distances(1, -1234.f);
        Watched<faiss::idx_t> labels(1, static_cast<faiss::idx_t>(-777));
        try {
            built.pre.search(
                    1, query.data(), 1, distances.data(), labels.data());
        } catch (const faiss::FaissException&) {
            refuse_on_faiss(
                    query.unchanged() && distances.unchanged() &&
                    labels.unchanged());
        }
    });
}

TEST(RemapDimensionsBounds, PreTransformRangeSearchRejectsBeforeRead) {
    expect_child_refuses([] {
        BuiltPreTransform built;
        built.remap.map[kDim - 1] = kDim;
        GuardedFloats query(kDim);
        faiss::RangeSearchResult result(1);
        try {
            built.pre.range_search(1, query.data(), 1e6f, &result);
        } catch (const faiss::FaissException&) {
            refuse_on_faiss(
                    query.unchanged() && result.labels == nullptr &&
                    result.distances == nullptr && result.lims[0] == 0 &&
                    result.lims[1] == 0);
        }
    });
}

TEST(RemapDimensionsBounds, PreTransformSearchSubsetRejectsBeforeRead) {
    expect_child_refuses([] {
        BuiltPreTransform built;
        built.remap.map[kDim - 1] = kDim;
        GuardedFloats query(kDim);
        const faiss::idx_t base = 0;
        Watched<float> distances(1, -1234.f);
        Watched<faiss::idx_t> labels(1, static_cast<faiss::idx_t>(-777));
        try {
            built.pre.search_subset(
                    1,
                    query.data(),
                    1,
                    &base,
                    1,
                    distances.data(),
                    labels.data());
        } catch (const faiss::FaissException&) {
            refuse_on_faiss(
                    query.unchanged() && distances.unchanged() &&
                    labels.unchanged());
        }
    });
}

TEST(RemapDimensionsBounds, PreTransformAddRejectsBeforeRead) {
    expect_child_refuses([] {
        BuiltPreTransform built;
        built.remap.map[kDim - 1] = kDim;
        const faiss::idx_t ntotal = built.pre.ntotal;
        GuardedFloats vectors(kDim);
        try {
            built.pre.add(1, vectors.data());
        } catch (const faiss::FaissException&) {
            refuse_on_faiss(vectors.unchanged() && built.pre.ntotal == ntotal);
        }
    });
}

TEST(RemapDimensionsBounds, PreTransformAddWithIdsRejectsBeforeRead) {
    expect_child_refuses([] {
        BuiltPreTransform built;
        built.remap.map[kDim - 1] = kDim;
        const faiss::idx_t ntotal = built.pre.ntotal;
        GuardedFloats vectors(kDim);
        const faiss::idx_t id = 7;
        try {
            built.pre.add_with_ids(1, vectors.data(), &id);
        } catch (const faiss::FaissException&) {
            refuse_on_faiss(vectors.unchanged() && built.pre.ntotal == ntotal);
        }
    });
}

TEST(RemapDimensionsBounds, PreTransformSaEncodeRejectsBeforeRead) {
    expect_child_refuses([] {
        BuiltPreTransform built;
        built.remap.map[kDim - 1] = kDim;
        GuardedFloats vectors(kDim);
        Watched<uint8_t> code(built.pre.sa_code_size(), 0x5a);
        try {
            built.pre.sa_encode(1, vectors.data(), code.data());
        } catch (const faiss::FaissException&) {
            refuse_on_faiss(vectors.unchanged() && code.unchanged());
        }
    });
}

TEST(RemapDimensionsBounds, PreTransformSaDecodeRejectsBeforeWrite) {
    expect_child_refuses([] {
        BuiltPreTransform built;
        built.remap.map[kDim - 1] = kDim;
        std::vector<uint8_t> code(built.pre.sa_code_size());
        GuardedFloats decoded(kDim);
        try {
            built.pre.sa_decode(1, code.data(), decoded.data());
        } catch (const faiss::FaissException&) {
            refuse_on_faiss(decoded.unchanged());
        }
    });
}

TEST(RemapDimensionsBounds, PreTransformReconstructRejectsBeforeWrite) {
    expect_child_refuses([] {
        BuiltPreTransform built;
        built.remap.map[kDim - 1] = kDim;
        GuardedFloats reconstructed(kDim);
        try {
            built.pre.reconstruct(0, reconstructed.data());
        } catch (const faiss::FaissException&) {
            refuse_on_faiss(reconstructed.unchanged());
        }
    });
}

TEST(RemapDimensionsBounds, PreTransformReconstructNRejectsBeforeWrite) {
    expect_child_refuses([] {
        BuiltPreTransform built;
        built.remap.map[kDim - 1] = kDim;
        GuardedFloats reconstructed(kDim);
        try {
            built.pre.reconstruct_n(0, 1, reconstructed.data());
        } catch (const faiss::FaissException&) {
            refuse_on_faiss(reconstructed.unchanged());
        }
    });
}

TEST(RemapDimensionsBounds, PreTransformSearchAndReconstructRejects) {
    expect_child_refuses([] {
        BuiltPreTransform built;
        built.remap.map[kDim - 1] = kDim;
        GuardedFloats query(kDim);
        Watched<float> distances(1, -1234.f);
        Watched<faiss::idx_t> labels(1, static_cast<faiss::idx_t>(-777));
        Watched<float> reconstructed(kDim, 42.f);
        try {
            built.pre.search_and_reconstruct(
                    1,
                    query.data(),
                    1,
                    distances.data(),
                    labels.data(),
                    reconstructed.data());
        } catch (const faiss::FaissException&) {
            refuse_on_faiss(
                    query.unchanged() && distances.unchanged() &&
                    labels.unchanged() && reconstructed.unchanged());
        }
    });
}

TEST(RemapDimensionsBounds, DistanceComputerRejectsBeforeRead) {
    expect_child_refuses([] {
        BuiltPreTransform built;
        built.remap.map[kDim - 1] = kDim;
        GuardedFloats query(kDim);
        std::unique_ptr<faiss::DistanceComputer> computer(
                built.pre.get_distance_computer());
        try {
            computer->set_query(query.data());
        } catch (const faiss::FaissException&) {
            refuse_on_faiss(query.unchanged());
        }
    });
}

TEST(RemapDimensionsBounds, SpectralHashSearchRejectsBeforeRead) {
    expect_child_refuses([] {
        faiss::IndexFlatL2 quantizer(kDim);
        std::vector<float> centroid(kDim, 0.2f);
        quantizer.add(1, centroid.data());
        const int entries[kDim] = {0, 1, 2, 3};
        faiss::RemapDimensionsTransform remap(kDim, kDim, entries);
        faiss::IndexIVFSpectralHash spectral(
                &quantizer, kDim, 1, kDim, /*period=*/1.f);
        spectral.replace_vt(&remap, false);
        if (!spectral.is_trained) {
            _exit(5);
        }
        remap.map[kDim - 1] = kDim;
        GuardedFloats query(kDim);
        Watched<float> distances(1, -1234.f);
        Watched<faiss::idx_t> labels(1, static_cast<faiss::idx_t>(-777));
        try {
            spectral.search(
                    1, query.data(), 1, distances.data(), labels.data());
        } catch (const faiss::FaissException&) {
            refuse_on_faiss(
                    query.unchanged() && distances.unchanged() &&
                    labels.unchanged());
        }
    });
}

TEST(RemapDimensionsBounds, SpectralHashEncodeRejectsBeforeRead) {
    expect_child_refuses([] {
        faiss::IndexFlatL2 quantizer(kDim);
        std::vector<float> centroid(kDim, 0.2f);
        quantizer.add(1, centroid.data());
        const int entries[kDim] = {0, 1, 2, 3};
        faiss::RemapDimensionsTransform remap(kDim, kDim, entries);
        faiss::IndexIVFSpectralHash spectral(
                &quantizer, kDim, 1, kDim, /*period=*/1.f);
        spectral.replace_vt(&remap, false);
        if (!spectral.is_trained) {
            _exit(5);
        }
        remap.map[kDim - 1] = kDim;
        GuardedFloats vectors(kDim);
        const faiss::idx_t list_no = 0;
        Watched<uint8_t> codes(spectral.code_size, 0x5a);
        try {
            spectral.encode_vectors(
                    1, vectors.data(), &list_no, codes.data(), false);
        } catch (const faiss::FaissException&) {
            refuse_on_faiss(vectors.unchanged() && codes.unchanged());
        }
    });
}

TEST(RemapDimensionsBounds, SpectralHashTrainEncoderRejectsBeforeRead) {
    expect_child_refuses([] {
        faiss::IndexFlatL2 quantizer(kDim);
        std::vector<float> centroid(kDim, 0.2f);
        quantizer.add(1, centroid.data());
        const int entries[kDim] = {0, 1, 2, 3};
        faiss::RemapDimensionsTransform remap(kDim, kDim, entries);
        faiss::IndexIVFSpectralHash spectral(
                &quantizer, kDim, 1, kDim, /*period=*/1.f);
        spectral.replace_vt(&remap, false);
        remap.map[0] = kLargeEntry;
        spectral.threshold_type =
                faiss::IndexIVFSpectralHash::Thresh_centroid;
        std::vector<float> train(kDim, 0.1f);
        try {
            spectral.train_encoder(1, train.data(), nullptr);
        } catch (const faiss::FaissException&) {
            _exit(0);
        }
    });
}

TEST(RemapDimensionsBounds, IndependentQuantizerSearchRejectsBeforeRead) {
    expect_child_refuses([] {
        faiss::IndexFlatL2 coarse(kDim);
        std::vector<float> centroid(kDim, 0.2f);
        coarse.add(1, centroid.data());
        faiss::IndexFlatL2 ivf_quantizer(kDim);
        faiss::IndexIVFFlat ivf(&ivf_quantizer, kDim, 1);
        const int entries[kDim] = {0, 1, 2, 3};
        faiss::RemapDimensionsTransform remap(kDim, kDim, entries);
        faiss::IndexIVFIndependentQuantizer indep(&coarse, &ivf, &remap);
        remap.map[kDim - 1] = kDim;
        GuardedFloats query(kDim);
        Watched<float> distances(1, -1234.f);
        Watched<faiss::idx_t> labels(1, static_cast<faiss::idx_t>(-777));
        try {
            indep.search(1, query.data(), 1, distances.data(), labels.data());
        } catch (const faiss::FaissException&) {
            refuse_on_faiss(
                    query.unchanged() && distances.unchanged() &&
                    labels.unchanged());
        }
    });
}

TEST(RemapDimensionsBounds, IndependentQuantizerAddRejectsBeforeRead) {
    expect_child_refuses([] {
        faiss::IndexFlatL2 coarse(kDim);
        std::vector<float> centroid(kDim, 0.2f);
        coarse.add(1, centroid.data());
        faiss::IndexFlatL2 ivf_quantizer(kDim);
        faiss::IndexIVFFlat ivf(&ivf_quantizer, kDim, 1);
        const int entries[kDim] = {0, 1, 2, 3};
        faiss::RemapDimensionsTransform remap(kDim, kDim, entries);
        faiss::IndexIVFIndependentQuantizer indep(&coarse, &ivf, &remap);
        remap.map[kDim - 1] = kDim;
        const faiss::idx_t ntotal = indep.ntotal;
        GuardedFloats vectors(kDim);
        try {
            indep.add(1, vectors.data());
        } catch (const faiss::FaissException&) {
            refuse_on_faiss(vectors.unchanged() && indep.ntotal == ntotal);
        }
    });
}
