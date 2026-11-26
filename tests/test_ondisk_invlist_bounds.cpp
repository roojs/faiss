/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// On-disk inverted lists ("ilod") store a mapped length and, per list, an
// offset, a size, and a capacity. Those extents are attacker-controlled
// fields of an index file. Loading that index and searching, adding, or
// reconstructing must refuse lists whose bytes do not lie entirely inside
// the mapped file, and must refuse a mapped length larger than the file.
//
// A refusal is a FaissException raised on the calling thread, either from
// read_index / read_index_binary or from the operation that would have used
// the list. The data file and the index file must be unchanged, and the
// loader must not create or delete any file. A crash, including one in a
// prefetch thread, is a failure.
//
// These tests fail until that refusal is in place.
//
// Operations that read or write the same extents and are not called here:
// IndexIVF::search_and_reconstruct, IndexIVF::search1,
// IndexIVF::check_ids_sorted, IndexBinaryIVF::range_search,
// IndexBinaryIVF::reconstruct_n, IndexBinaryIVF::add, and
// OnDiskInvertedLists::merge_from_multiple. They all go through get_codes,
// get_ids, prefetch_lists, or add_entries. IVF subclasses that do not
// override IndexIVF::search (IVFPQ and the scalar-quantizer indexes) share
// the float search path covered below.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <stdlib.h>
#include <unistd.h>

#include <faiss/IndexBinaryIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissException.h>
#include <faiss/index_io.h>
#include <faiss/invlists/OnDiskInvertedLists.h>

namespace {

template <typename T>
void push_val(std::vector<uint8_t>& buf, T val) {
    const auto* p = reinterpret_cast<const uint8_t*>(&val);
    buf.insert(buf.end(), p, p + sizeof(T));
}

template <typename T>
void push_vector(std::vector<uint8_t>& buf, const std::vector<T>& vec) {
    push_val<size_t>(buf, vec.size());
    if (vec.empty()) {
        return;
    }
    const auto* p = reinterpret_cast<const uint8_t*>(vec.data());
    buf.insert(buf.end(), p, p + vec.size() * sizeof(T));
}

void push_fourcc(std::vector<uint8_t>& buf, const char s[4]) {
    const auto* x = reinterpret_cast<const unsigned char*>(s);
    uint32_t h = x[0] | (x[1] << 8) | (x[2] << 16) | (x[3] << 24);
    push_val<uint32_t>(buf, h);
}

// Matches write_index_header. metric_type > 1 would also store metric_arg.
void push_index_header(
        std::vector<uint8_t>& buf,
        int d,
        faiss::idx_t ntotal,
        bool is_trained = true,
        int metric_type = faiss::METRIC_L2) {
    push_val<int>(buf, d);
    push_val<faiss::idx_t>(buf, ntotal);
    faiss::idx_t dummy = faiss::idx_t{1} << 20;
    push_val<faiss::idx_t>(buf, dummy);
    push_val<faiss::idx_t>(buf, dummy);
    push_val<uint8_t>(buf, is_trained ? 1 : 0);
    push_val<int>(buf, metric_type);
}

void push_direct_map_none(std::vector<uint8_t>& buf) {
    push_val<uint8_t>(buf, 0); // DirectMap::NoMap
    push_val<size_t>(buf, 0);  // empty array
}

void push_flat_l2_quantizer(std::vector<uint8_t>& buf, int d) {
    push_fourcc(buf, "IxF2");
    push_index_header(buf, d, /*ntotal=*/1);
    // READXBVECTOR stores the byte length divided by 4.
    std::vector<uint8_t> centroid(static_cast<size_t>(d) * sizeof(float), 0);
    push_val<size_t>(buf, centroid.size() / 4);
    buf.insert(buf.end(), centroid.begin(), centroid.end());
}

void push_ilod(
        std::vector<uint8_t>& buf,
        size_t nlist,
        size_t code_size,
        const faiss::OnDiskOneList& list,
        const std::string& filename,
        size_t totsize) {
    push_fourcc(buf, "ilod");
    push_val<size_t>(buf, nlist);
    push_val<size_t>(buf, code_size);
    push_vector<faiss::OnDiskOneList>(buf, {list});
    push_val<size_t>(buf, 0); // no free slots
    push_vector<char>(buf, std::vector<char>(filename.begin(), filename.end()));
    push_val<size_t>(buf, totsize);
}

struct Fixture {
    std::filesystem::path dir;
    std::filesystem::path index_path;
    std::filesystem::path data_path;
    std::vector<uint8_t> index_bytes;
    std::vector<uint8_t> data_bytes;

    explicit Fixture(std::string name) {
        std::string tmpl = "/tmp/faiss_ilod_" + name + "_XXXXXX";
        std::vector<char> buf(tmpl.begin(), tmpl.end());
        buf.push_back('\0');
        if (mkdtemp(buf.data()) == nullptr) {
            throw std::runtime_error("mkdtemp failed");
        }
        dir = buf.data();
        index_path = dir / "index.faiss";
        data_path = dir / "vectors.dat";
    }

    Fixture(const Fixture&) = delete;
    Fixture& operator=(const Fixture&) = delete;

    Fixture(Fixture&& other) noexcept
            : dir(std::move(other.dir)),
              index_path(std::move(other.index_path)),
              data_path(std::move(other.data_path)),
              index_bytes(std::move(other.index_bytes)),
              data_bytes(std::move(other.data_bytes)) {
        other.dir.clear();
    }

    Fixture& operator=(Fixture&& other) noexcept {
        if (this != &other) {
            std::error_code ec;
            if (!dir.empty()) {
                std::filesystem::remove_all(dir, ec);
            }
            dir = std::move(other.dir);
            index_path = std::move(other.index_path);
            data_path = std::move(other.data_path);
            index_bytes = std::move(other.index_bytes);
            data_bytes = std::move(other.data_bytes);
            other.dir.clear();
        }
        return *this;
    }

    ~Fixture() {
        if (dir.empty()) {
            return;
        }
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }

    void write_files() const {
        write_bytes(index_path, index_bytes);
        write_bytes(data_path, data_bytes);
    }

    std::set<std::filesystem::path> snapshot() const {
        std::set<std::filesystem::path> names;
        for (const auto& ent : std::filesystem::directory_iterator(dir)) {
            names.insert(ent.path().filename());
        }
        return names;
    }

    static void write_bytes(
            const std::filesystem::path& path,
            const std::vector<uint8_t>& bytes) {
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out.good());
        out.write(
                reinterpret_cast<const char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
        ASSERT_TRUE(out.good());
    }

    static std::vector<uint8_t> read_bytes(const std::filesystem::path& path) {
        std::ifstream in(path, std::ios::binary);
        EXPECT_TRUE(in.good());
        return std::vector<uint8_t>(
                std::istreambuf_iterator<char>(in),
                std::istreambuf_iterator<char>());
    }
};

size_t page_size() {
    long p = sysconf(_SC_PAGESIZE);
    EXPECT_GT(p, 0);
    return static_cast<size_t>(p);
}

faiss::OnDiskOneList make_list(size_t size, size_t capacity, size_t offset) {
    faiss::OnDiskOneList list;
    list.size = size;
    list.capacity = capacity;
    list.offset = offset;
    return list;
}

// One-list IVFFlat (fourcc IwFl) whose ilod payload points at data_path.
// d is 1, so each code is one float (4 bytes) and each id is one idx_t.
std::vector<uint8_t> make_ivf_flat_ilod(
        const std::string& data_path,
        faiss::idx_t ntotal,
        const faiss::OnDiskOneList& list,
        size_t totsize) {
    const int d = 1;
    const size_t code_size = sizeof(float) * static_cast<size_t>(d);
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IwFl");
    push_index_header(buf, d, ntotal);
    push_val<size_t>(buf, 1); // nlist
    push_val<size_t>(buf, 1); // nprobe
    push_flat_l2_quantizer(buf, d);
    push_direct_map_none(buf);
    push_ilod(buf, 1, code_size, list, data_path, totsize);
    return buf;
}

std::vector<uint8_t> make_binary_ivf_ilod(
        const std::string& data_path,
        const faiss::OnDiskOneList& list,
        size_t totsize) {
    const int d = 8;
    const int code_size = 1;
    std::vector<uint8_t> buf;
    push_fourcc(buf, "IBwF");
    push_val<int>(buf, d);
    push_val<int>(buf, code_size);
    push_val<faiss::idx_t>(buf, 0);
    push_val<uint8_t>(buf, 1); // is_trained
    push_val<int>(buf, faiss::METRIC_L2);
    push_val<size_t>(buf, 1); // nlist
    push_val<size_t>(buf, 1); // nprobe
    push_fourcc(buf, "IBxF");
    push_val<int>(buf, d);
    push_val<int>(buf, code_size);
    push_val<faiss::idx_t>(buf, 1); // one centroid
    push_val<uint8_t>(buf, 1);
    push_val<int>(buf, faiss::METRIC_L2);
    push_vector<uint8_t>(buf, {0});
    push_direct_map_none(buf);
    push_ilod(buf, 1, static_cast<size_t>(code_size), list, data_path, totsize);
    return buf;
}

// Load, then run op unless load itself refuses. Either refusal must leave
// the directory and both files byte-for-byte as they were.
template <typename IndexT, typename Load, typename Op>
void expect_use_refused(Fixture& fx, Load load, Op op) {
    fx.write_files();
    const auto before_names = fx.snapshot();
    const auto index_before = Fixture::read_bytes(fx.index_path);
    const auto data_before = Fixture::read_bytes(fx.data_path);

    std::unique_ptr<IndexT> index;
    try {
        index = load();
    } catch (const faiss::FaissException&) {
        EXPECT_EQ(fx.snapshot(), before_names);
        EXPECT_EQ(Fixture::read_bytes(fx.index_path), index_before);
        EXPECT_EQ(Fixture::read_bytes(fx.data_path), data_before);
        return;
    }
    ASSERT_NE(index, nullptr);
    EXPECT_THROW(op(*index), faiss::FaissException);
    EXPECT_EQ(fx.snapshot(), before_names);
    EXPECT_EQ(Fixture::read_bytes(fx.index_path), index_before);
    EXPECT_EQ(Fixture::read_bytes(fx.data_path), data_before);
}

template <typename Op>
void expect_float_use_refused(Fixture& fx, Op op) {
    expect_use_refused<faiss::Index>(
            fx,
            [&fx] { return faiss::read_index_up(fx.index_path.c_str()); },
            op);
}

Fixture float_fixture(
        const std::string& name,
        faiss::OnDiskOneList list,
        size_t totsize,
        size_t file_size,
        faiss::idx_t ntotal = 0) {
    Fixture fx(name);
    fx.data_bytes.assign(file_size, 0x11);
    fx.index_bytes =
            make_ivf_flat_ilod(fx.data_path.string(), ntotal, list, totsize);
    return fx;
}

void search_one(faiss::Index& index) {
    float query = 0.f;
    float dist = 0.f;
    faiss::idx_t label = -1;
    index.search(1, &query, 1, &dist, &label);
}

} // namespace

TEST(OnDiskInvlistBounds, ValidListRoundTrips) {
    const size_t page = page_size();
    Fixture fx("valid");
    fx.data_bytes.assign(page, 0);
    const float code = 1.5f;
    const faiss::idx_t id = 42;
    ASSERT_GE(fx.data_bytes.size(), sizeof(code) + sizeof(id));
    std::memcpy(fx.data_bytes.data(), &code, sizeof(code));
    std::memcpy(fx.data_bytes.data() + sizeof(code), &id, sizeof(id));
    // One entry: 4 code bytes followed by one id. Both fit in the file.
    fx.index_bytes = make_ivf_flat_ilod(
            fx.data_path.string(),
            /*ntotal=*/1,
            make_list(1, 1, 0),
            page);
    fx.write_files();

    std::unique_ptr<faiss::Index> index(
            faiss::read_index(fx.index_path.c_str()));
    ASSERT_NE(index, nullptr);
    float query = code;
    float dist = -1.f;
    faiss::idx_t label = -1;
    ASSERT_NO_THROW(index->search(1, &query, 1, &dist, &label));
    EXPECT_EQ(label, id);
    EXPECT_EQ(dist, 0.f);
    EXPECT_EQ(Fixture::read_bytes(fx.data_path), fx.data_bytes);
}

// Attack path: offset 0, capacity 0, size far past the mapped file. Codes
// are read from the start of the map and ids from offset + capacity * code
// width, which is the same address. The read walks off the mapping.
TEST(OnDiskInvlistBounds, SearchReadsPastMappedFile) {
    const size_t page = page_size();
    auto fx = float_fixture(
            "search",
            make_list(/*size=*/page, /*capacity=*/0, /*offset=*/0),
            page,
            page);
    expect_float_use_refused(fx, search_one);
}

// Same list, but prefetch is turned off after a successful load so the
// read is the scanner's, on the calling thread.
TEST(OnDiskInvlistBounds, ScannerReadsPastMappedFile) {
    const size_t page = page_size();
    auto fx = float_fixture(
            "scanner",
            make_list(/*size=*/page, /*capacity=*/0, /*offset=*/0),
            page,
            page);
    expect_float_use_refused(fx, [](faiss::Index& index) {
        auto* ivf = dynamic_cast<faiss::IndexIVF*>(&index);
        auto* od = ivf == nullptr
                ? nullptr
                : dynamic_cast<faiss::OnDiskInvertedLists*>(ivf->invlists);
        if (od == nullptr) {
            throw std::runtime_error("expected OnDiskInvertedLists");
        }
        od->prefetch_nthread = 0;
        search_one(index);
    });
}

// Ids live at offset + capacity * code width. capacity alone places that
// pointer on the next page, past a mapping one page long.
TEST(OnDiskInvlistBounds, SearchReadsIdsPastMappedFile) {
    const size_t page = page_size();
    const size_t code_size = sizeof(float);
    ASSERT_EQ(page % code_size, 0u);
    auto fx = float_fixture(
            "ids",
            make_list(/*size=*/1, /*capacity=*/page / code_size, /*offset=*/0),
            page,
            page);
    expect_float_use_refused(fx, search_one);
}

TEST(OnDiskInvlistBounds, RangeSearchReadsPastMappedFile) {
    const size_t page = page_size();
    auto fx = float_fixture("range", make_list(page, 0, 0), page, page);
    expect_float_use_refused(fx, [](faiss::Index& index) {
        float query = 0.f;
        faiss::RangeSearchResult result(1);
        index.range_search(1, &query, 1.0e20f, &result);
    });
}

// reconstruct_n walks every id. ntotal is 1, so the call is in range for
// the index header, and the list size is what runs off the mapping.
TEST(OnDiskInvlistBounds, ReconstructReadsPastMappedFile) {
    const size_t page = page_size();
    auto fx = float_fixture(
            "reconstruct", make_list(page, 0, 0), page, page, /*ntotal=*/1);
    expect_float_use_refused(fx, [](faiss::Index& index) {
        float out = 0.f;
        index.reconstruct_n(0, 1, &out);
    });
}

// size is already in the upper half of capacity, so add keeps the slot
// and writes the new id at offset + capacity * code width + size * id width,
// which is past the mapping. The data file must not grow or change.
TEST(OnDiskInvlistBounds, AddWritesPastMappedFile) {
    const size_t page = page_size();
    const size_t code_size = sizeof(float);
    const size_t capacity = page / code_size;
    const size_t size = capacity / 2 + 1;
    auto fx = float_fixture("add", make_list(size, capacity, 0), page, page);
    expect_float_use_refused(fx, [](faiss::Index& index) {
        float v = 0.25f;
        index.add(1, &v);
    });
}

// The mapped length is two pages and the file is one. The single id sits
// just past EOF but inside the declared length, so a check against the
// mapped length alone still accepts it.
TEST(OnDiskInvlistBounds, MappedLengthPastEndOfFile) {
    const size_t page = page_size();
    const size_t code_size = sizeof(float);
    auto fx = float_fixture(
            "totsize",
            make_list(/*size=*/1, /*capacity=*/page / code_size, /*offset=*/0),
            /*totsize=*/page * 2,
            /*file_size=*/page);
    expect_float_use_refused(fx, search_one);
}

// offset is the invalid-offset sentinel and size is not zero. get_codes
// returns a null pointer; the search must refuse instead of dereferencing it.
TEST(OnDiskInvlistBounds, NonZeroSizeWithInvalidOffset) {
    const size_t page = page_size();
    auto fx = float_fixture(
            "sentinel",
            make_list(
                    /*size=*/4,
                    /*capacity=*/0,
                    /*offset=*/std::numeric_limits<size_t>::max()),
            page,
            page);
    expect_float_use_refused(fx, search_one);
}

// size * code width and size * id width both wrap to a handful of bytes,
// which is less than the file. The scan still walks `size` entries.
TEST(OnDiskInvlistBounds, SearchRejectsWrappingSize) {
    const size_t page = page_size();
    const size_t code_size = sizeof(float);
    const size_t size = std::numeric_limits<size_t>::max() / code_size + 8;
    auto fx = float_fixture("wrapsize", make_list(size, 0, 0), page, page);
    expect_float_use_refused(fx, search_one);
}

// offset + size * code width wraps to a small length. The code pointer is
// offset bytes from the mapping, which is outside it.
TEST(OnDiskInvlistBounds, SearchRejectsWrappingOffset) {
    const size_t page = page_size();
    const size_t offset = std::numeric_limits<size_t>::max() - 16;
    auto fx = float_fixture(
            "wrapoff", make_list(/*size=*/8, 0, offset), page, page);
    expect_float_use_refused(fx, search_one);
}

// capacity * code width wraps, so the id pointer lands a few bytes into the
// mapping. The true product does not fit in the file.
TEST(OnDiskInvlistBounds, SearchRejectsWrappingCapacity) {
    const size_t page = page_size();
    const size_t code_size = sizeof(float);
    const size_t capacity = std::numeric_limits<size_t>::max() / code_size + 2;
    auto fx = float_fixture(
            "wrapcap",
            make_list(/*size=*/1, capacity, /*offset=*/0),
            page,
            page);
    expect_float_use_refused(fx, search_one);
}

TEST(OnDiskInvlistBounds, BinarySearchReadsPastMappedFile) {
    const size_t page = page_size();
    Fixture fx("binary");
    fx.data_bytes.assign(page, 0x11);
    fx.index_bytes = make_binary_ivf_ilod(
            fx.data_path.string(), make_list(page, 0, 0), page);
    expect_use_refused<faiss::IndexBinary>(
            fx,
            [&fx] {
                return faiss::read_index_binary_up(fx.index_path.c_str());
            },
            [](faiss::IndexBinary& index) {
                uint8_t query = 0;
                int32_t dist = 0;
                faiss::idx_t label = -1;
                index.search(1, &query, 1, &dist, &label);
            });
}
