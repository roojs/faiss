/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Spec for the embedded OnDiskInvertedLists filename.
//
// An index payload that names a filesystem path must not cause read_index,
// read_index_binary, or read_InvertedLists to open, map, or modify that
// path. Refusing the load is fine. A load that succeeds must not return
// bytes of that file as neighbor ids, and a later add, merge, or
// update_entries must not create, truncate, or write the file.
//
// IO_FLAG_ONDISK_SAME_DIR may select a basename in the index file's
// directory, but that name must not follow a symlink to another file, and
// a second lookup of the name must not observe a different inode than the
// first. Mutation of even that basename stays forbidden: these tests never
// pass a mutation opt-in.
//
// The tests fail while the current reader opens the stored path read-write
// and maps it MAP_SHARED.
//
// Not covered here, because they do not take a path from the index payload:
// OnDiskInvertedListsIOHook::read_ArrayInvertedLists maps the index file the
// caller already opened. Python deserialize_index and deserialize_index_binary
// call read_index and read_index_binary. Other IVF fourccs call
// read_InvertedLists. prefetch_lists, remove_ids, reset, copy_subset_to, and
// merge_from_multiple read or write that same mapping; search, add,
// merge_from, add_entries, and update_entries below drive those paths.

#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryIVF.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>
#include <faiss/invlists/OnDiskInvertedLists.h>

#include <gtest/gtest.h>

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kDim = 2;
constexpr size_t kCodeSize = sizeof(float) * kDim;
// One list slot is code_size bytes followed by one idx_t.
constexpr size_t kSlotBytes = kCodeSize + sizeof(faiss::idx_t);

// Marker stored only in the payload file. Not a credential.
constexpr faiss::idx_t kSecretId = 0x13579BDF2468ACE0LL;
constexpr faiss::idx_t kSafeId = 7;

struct FileState {
    bool exists = false;
    uint64_t dev = 0;
    uint64_t ino = 0;
    std::vector<uint8_t> bytes;
};

FileState capture(const std::filesystem::path& path) {
    FileState state;
    struct stat st;
    if (lstat(path.c_str(), &st) != 0) {
        return state;
    }
    state.exists = true;
    state.dev = static_cast<uint64_t>(st.st_dev);
    state.ino = static_cast<uint64_t>(st.st_ino);
    if (!S_ISREG(st.st_mode)) {
        return state;
    }
    std::ifstream in(path, std::ios::binary);
    state.bytes.assign(
            std::istreambuf_iterator<char>(in),
            std::istreambuf_iterator<char>());
    return state;
}

void expect_unchanged(
        const FileState& before,
        const std::filesystem::path& path) {
    FileState after = capture(path);
    EXPECT_EQ(before.exists, after.exists) << path;
    EXPECT_EQ(before.dev, after.dev) << path;
    EXPECT_EQ(before.ino, after.ino) << path;
    EXPECT_EQ(before.bytes, after.bytes) << path;
}

void write_file(
        const std::filesystem::path& path,
        const std::vector<uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.good()) << path;
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    ASSERT_TRUE(out.good()) << path;
}

std::vector<uint8_t> disclosure_bytes(faiss::idx_t id) {
    // codes[capacity=1] then ids[capacity=1], both 8-byte aligned.
    std::vector<uint8_t> bytes(kSlotBytes, 0);
    std::memcpy(bytes.data() + kCodeSize, &id, sizeof(id));
    return bytes;
}

struct TempDir {
    std::filesystem::path path;
    TempDir() {
        std::string tmpl =
                (std::filesystem::temp_directory_path() / "faiss_od_XXXXXX")
                        .string();
        if (mkdtemp(tmpl.data()) == nullptr) {
            throw std::runtime_error("mkdtemp failed");
        }
        path = tmpl;
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

struct ScopedChdir {
    std::string previous;
    explicit ScopedChdir(const std::filesystem::path& dir) {
        char buf[4096];
        if (getcwd(buf, sizeof(buf)) == nullptr) {
            throw std::runtime_error("getcwd failed");
        }
        previous = buf;
        if (chdir(dir.c_str()) != 0) {
            throw std::runtime_error("chdir failed");
        }
    }
    ~ScopedChdir() {
        if (!previous.empty()) {
            chdir(previous.c_str());
        }
    }
};

std::vector<uint8_t> serialize_flat_ivf(
        const std::string& embedded,
        size_t list_size,
        size_t list_capacity,
        size_t totsize) {
    faiss::IndexFlatL2 quantizer(kDim);
    float centroid[kDim] = {0.f, 0.f};
    quantizer.add(1, centroid);
    const size_t code_size = sizeof(float) * kDim;
    faiss::OnDiskInvertedLists od(1, code_size, embedded.c_str());
    od.lists[0].size = list_size;
    od.lists[0].capacity = list_capacity;
    od.lists[0].offset = 0;
    od.totsize = totsize;
    faiss::IndexIVFFlat index(&quantizer, kDim, 1);
    index.nprobe = 1;
    index.ntotal = static_cast<faiss::idx_t>(list_size);
    index.replace_invlists(&od, false);
    faiss::VectorIOWriter writer;
    faiss::write_index(&index, &writer);
    return writer.data;
}

std::vector<uint8_t> serialize_binary_ivf(
        const std::string& embedded,
        size_t list_size,
        size_t list_capacity,
        size_t totsize) {
    constexpr int d = 64;
    faiss::IndexBinaryFlat quantizer(d);
    std::vector<uint8_t> centroid(d / 8, 0);
    quantizer.add(1, centroid.data());
    faiss::OnDiskInvertedLists od(1, quantizer.code_size, embedded.c_str());
    od.lists[0].size = list_size;
    od.lists[0].capacity = list_capacity;
    od.lists[0].offset = 0;
    od.totsize = totsize;
    faiss::IndexBinaryIVF index(&quantizer, d, 1);
    index.nprobe = 1;
    index.ntotal = static_cast<faiss::idx_t>(list_size);
    index.replace_invlists(&od, false);
    faiss::VectorIOWriter writer;
    faiss::write_index_binary(&index, &writer);
    return writer.data;
}

std::vector<uint8_t> serialize_invlists(
        const std::string& embedded,
        size_t list_size,
        size_t list_capacity,
        size_t totsize) {
    faiss::OnDiskInvertedLists od(1, kCodeSize, embedded.c_str());
    od.lists[0].size = list_size;
    od.lists[0].capacity = list_capacity;
    od.lists[0].offset = 0;
    od.totsize = totsize;
    faiss::VectorIOWriter writer;
    faiss::write_InvertedLists(&od, &writer);
    return writer.data;
}

faiss::idx_t search_label(faiss::Index& index) {
    float query[kDim] = {0.f, 0.f};
    float distance = 0.f;
    faiss::idx_t label = -1;
    index.search(1, query, 1, &distance, &label);
    return label;
}

bool range_contains(faiss::Index& index, faiss::idx_t id) {
    float query[kDim] = {0.f, 0.f};
    faiss::RangeSearchResult result(1);
    index.range_search(1, query, 1.0f, &result);
    for (size_t i = result.lims[0]; i < result.lims[1]; ++i) {
        if (result.labels[i] == id) {
            return true;
        }
    }
    return false;
}

faiss::idx_t search_binary_label(faiss::IndexBinary& index) {
    std::vector<uint8_t> query(index.code_size, 0);
    int32_t distance = 0;
    faiss::idx_t label = -1;
    index.search(1, query.data(), 1, &distance, &label);
    return label;
}

template <typename Loader>
void expect_float_search_hides_file(
        Loader loader,
        const std::filesystem::path& secret,
        bool mutate) {
    FileState before = capture(secret);
    std::unique_ptr<faiss::Index> index;
    faiss::idx_t label = -1;
    bool ranged = false;
    try {
        index.reset(loader());
        if (index) {
            label = search_label(*index);
            ranged = range_contains(*index, kSecretId);
        }
    } catch (const faiss::FaissException&) {
        index.reset();
    }
    EXPECT_NE(label, kSecretId);
    EXPECT_FALSE(ranged);
    expect_unchanged(before, secret);
    // IndexIVF::add throws from inside an OpenMP region when the lists are
    // read-only, which terminates the process. Callers that loaded with
    // IO_FLAG_READ_ONLY check mutation through InvertedLists::add_entries.
    if (mutate && index) {
        try {
            float added[kDim] = {0.f, 0.f};
            index->add(1, added);
        } catch (const faiss::FaissException&) {
        }
        expect_unchanged(before, secret);
    }
}

} // namespace

TEST(OnDiskEmbeddedPath, ReadIndexFilenameDoesNotUseEmbeddedFile) {
    TempDir dir;
    auto secret = dir.path / "secret";
    write_file(secret, disclosure_bytes(kSecretId));
    FileState before_build = capture(secret);
    auto bytes = serialize_flat_ivf(secret.string(), 1, 1, kSlotBytes);
    expect_unchanged(before_build, secret);

    auto index_path = dir.path / "index.faiss";
    write_file(index_path, bytes);
    expect_float_search_hides_file(
            [&]() { return faiss::read_index(index_path.c_str(), 0); },
            secret,
            true);

    FILE* f = fopen(index_path.c_str(), "rb");
    ASSERT_NE(f, nullptr);
    expect_float_search_hides_file(
            [&]() {
                rewind(f);
                return faiss::read_index(f, 0);
            },
            secret,
            true);
    fclose(f);

    expect_float_search_hides_file(
            [&]() {
                faiss::VectorIOReader reader;
                reader.data = bytes;
                return faiss::read_index(&reader, 0);
            },
            secret,
            true);
}

TEST(OnDiskEmbeddedPath, ReadOnlyFlagDoesNotDiscloseEmbeddedFile) {
    TempDir dir;
    auto secret = dir.path / "secret";
    write_file(secret, disclosure_bytes(kSecretId));
    auto bytes = serialize_flat_ivf(secret.string(), 1, 1, kSlotBytes);
    auto index_path = dir.path / "index.faiss";
    write_file(index_path, bytes);
    FileState before = capture(secret);
    std::unique_ptr<faiss::Index> index;
    faiss::idx_t label = -1;
    try {
        index.reset(
                faiss::read_index(
                        index_path.c_str(), faiss::IO_FLAG_READ_ONLY));
        if (index) {
            label = search_label(*index);
        }
    } catch (const faiss::FaissException&) {
        index.reset();
    }
    EXPECT_NE(label, kSecretId);
    expect_unchanged(before, secret);
    if (!index) {
        return;
    }
    auto* ivf = dynamic_cast<faiss::IndexIVF*>(index.get());
    ASSERT_NE(ivf, nullptr);
    ASSERT_NE(ivf->invlists, nullptr);
    try {
        uint8_t code[kCodeSize] = {};
        faiss::idx_t id = 99;
        ivf->invlists->add_entries(0, 1, &id, code);
    } catch (const faiss::FaissException&) {
    }
    expect_unchanged(before, secret);
}

TEST(OnDiskEmbeddedPath, InPlaceAddDoesNotWriteEmbeddedFile) {
    TempDir dir;
    auto secret = dir.path / "secret";
    // capacity 4, size 2: one more entry stays inside the allocated slot and
    // is copied into the mapping by update_entries.
    constexpr size_t capacity = 4;
    constexpr size_t totsize = capacity * kSlotBytes;
    std::vector<uint8_t> payload(totsize, 0xA5);
    write_file(secret, payload);
    auto bytes = serialize_flat_ivf(secret.string(), 2, capacity, totsize);
    auto index_path = dir.path / "index.faiss";
    write_file(index_path, bytes);

    FileState before = capture(secret);
    std::unique_ptr<faiss::Index> index;
    try {
        index.reset(faiss::read_index(index_path.c_str(), 0));
    } catch (const faiss::FaissException&) {
    }
    expect_unchanged(before, secret);
    if (!index) {
        return;
    }
    try {
        float added[kDim] = {1.f, 2.f};
        index->add(1, added);
    } catch (const faiss::FaissException&) {
    }
    expect_unchanged(before, secret);
}

TEST(OnDiskEmbeddedPath, GrowingAddDoesNotTruncateEmbeddedFile) {
    TempDir dir;
    auto secret = dir.path / "secret";
    write_file(secret, disclosure_bytes(kSecretId));
    auto bytes = serialize_flat_ivf(secret.string(), 1, 1, kSlotBytes);
    auto index_path = dir.path / "index.faiss";
    write_file(index_path, bytes);

    FileState before = capture(secret);
    std::unique_ptr<faiss::Index> index;
    try {
        index.reset(faiss::read_index(index_path.c_str(), 0));
    } catch (const faiss::FaissException&) {
    }
    expect_unchanged(before, secret);
    if (!index) {
        return;
    }
    try {
        float added[kDim] = {1.f, 2.f};
        index->add(1, added);
    } catch (const faiss::FaissException&) {
    }
    expect_unchanged(before, secret);
}

TEST(OnDiskEmbeddedPath, MergeFromDoesNotModifyEmbeddedFile) {
    TempDir dir;
    auto secret = dir.path / "secret";
    write_file(secret, disclosure_bytes(kSecretId));
    auto bytes = serialize_flat_ivf(secret.string(), 1, 1, kSlotBytes);
    faiss::VectorIOReader reader;
    reader.data = bytes;

    FileState before = capture(secret);
    std::unique_ptr<faiss::Index> index;
    try {
        index.reset(faiss::read_index(&reader, 0));
    } catch (const faiss::FaissException&) {
    }
    expect_unchanged(before, secret);
    if (!index) {
        return;
    }
    try {
        faiss::IndexFlatL2 quantizer(kDim);
        float centroid[kDim] = {0.f, 0.f};
        quantizer.add(1, centroid);
        faiss::IndexIVFFlat other(&quantizer, kDim, 1);
        float added[kDim] = {0.f, 0.f};
        other.add(1, added);
        index->merge_from(other, 0);
    } catch (const faiss::FaissException&) {
    }
    expect_unchanged(before, secret);
}

TEST(OnDiskEmbeddedPath, ParentSegmentIsNotOpened) {
    TempDir root;
    auto sub = root.path / "sub";
    std::filesystem::create_directory(sub);
    auto secret = root.path / "secret";
    write_file(secret, disclosure_bytes(kSecretId));
    auto bytes = serialize_flat_ivf("../secret", 1, 1, kSlotBytes);
    ScopedChdir cwd(sub);
    expect_float_search_hides_file(
            [&]() {
                faiss::VectorIOReader reader;
                reader.data = bytes;
                return faiss::read_index(&reader, 0);
            },
            secret,
            true);
}

TEST(OnDiskEmbeddedPath, ReadInvertedListsDoesNotExposeOrModifyEmbeddedFile) {
    TempDir dir;
    auto secret = dir.path / "secret";
    write_file(secret, disclosure_bytes(kSecretId));
    auto bytes = serialize_invlists(secret.string(), 1, 1, kSlotBytes);
    faiss::VectorIOReader reader;
    reader.data = bytes;

    FileState before = capture(secret);
    std::unique_ptr<faiss::InvertedLists> lists;
    try {
        lists.reset(faiss::read_InvertedLists(&reader, 0));
    } catch (const faiss::FaissException&) {
    }
    expect_unchanged(before, secret);
    if (!lists) {
        return;
    }
    const faiss::idx_t* ids = lists->get_ids(0);
    if (ids != nullptr && lists->list_size(0) > 0) {
        EXPECT_NE(ids[0], kSecretId);
    }
    try {
        uint8_t code[kCodeSize] = {9, 8, 7, 6, 5, 4, 3, 2};
        faiss::idx_t id = 99;
        lists->update_entries(0, 0, 1, &id, code);
    } catch (const faiss::FaissException&) {
    }
    expect_unchanged(before, secret);
    try {
        uint8_t code[kCodeSize] = {1, 1, 1, 1, 1, 1, 1, 1};
        faiss::idx_t id = 100;
        lists->add_entries(0, 1, &id, code);
    } catch (const faiss::FaissException&) {
    }
    expect_unchanged(before, secret);
}

TEST(OnDiskEmbeddedPath, ReadIndexBinaryDoesNotExposeOrModifyEmbeddedFile) {
    TempDir dir;
    auto secret = dir.path / "secret";
    // Binary code_size for d=64 is 8, the same slot layout as the float index.
    write_file(secret, disclosure_bytes(kSecretId));
    auto bytes = serialize_binary_ivf(secret.string(), 1, 1, kSlotBytes);
    auto index_path = dir.path / "index.bin";
    write_file(index_path, bytes);

    FileState before = capture(secret);
    std::unique_ptr<faiss::IndexBinary> index;
    faiss::idx_t label = -1;
    try {
        index.reset(faiss::read_index_binary(index_path.c_str(), 0));
        if (index) {
            label = search_binary_label(*index);
        }
    } catch (const faiss::FaissException&) {
        index.reset();
    }
    EXPECT_NE(label, kSecretId);
    expect_unchanged(before, secret);
    if (!index) {
        return;
    }
    try {
        std::vector<uint8_t> added(index->code_size, 1);
        index->add(1, added.data());
    } catch (const faiss::FaissException&) {
    }
    expect_unchanged(before, secret);

    faiss::VectorIOReader reader;
    reader.data = bytes;
    label = -1;
    std::unique_ptr<faiss::IndexBinary> from_memory;
    try {
        from_memory.reset(faiss::read_index_binary(&reader, 0));
        if (from_memory) {
            label = search_binary_label(*from_memory);
        }
    } catch (const faiss::FaissException&) {
    }
    EXPECT_NE(label, kSecretId);
    expect_unchanged(before, secret);
}

TEST(OnDiskEmbeddedPath, SkipIvfDataDoesNotCreateOrTruncateEmbeddedFile) {
    TempDir dir;
    auto created = dir.path / "created_by_add";
    auto bytes = serialize_flat_ivf(created.string(), 0, 0, 0);
    faiss::VectorIOReader reader;
    reader.data = bytes;
    std::unique_ptr<faiss::Index> index;
    try {
        index.reset(faiss::read_index(&reader, faiss::IO_FLAG_SKIP_IVF_DATA));
    } catch (const faiss::FaissException&) {
    }
    EXPECT_FALSE(capture(created).exists) << created;
    if (index) {
        try {
            float added[kDim] = {0.f, 0.f};
            index->add(1, added);
        } catch (const faiss::FaissException&) {
        }
    }
    EXPECT_FALSE(capture(created).exists) << created;

    auto secret = dir.path / "secret";
    write_file(secret, disclosure_bytes(kSecretId));
    auto grow = serialize_flat_ivf(secret.string(), 1, 1, kSlotBytes);
    FileState before = capture(secret);
    faiss::VectorIOReader grow_reader;
    grow_reader.data = grow;
    std::unique_ptr<faiss::Index> grow_index;
    try {
        grow_index.reset(
                faiss::read_index(&grow_reader, faiss::IO_FLAG_SKIP_IVF_DATA));
    } catch (const faiss::FaissException&) {
    }
    if (grow_index) {
        try {
            float added[kDim] = {1.f, 2.f};
            grow_index->add(1, added);
        } catch (const faiss::FaissException&) {
        }
    }
    expect_unchanged(before, secret);
}

TEST(OnDiskEmbeddedPath, SameDirBasenameIsNotMutated) {
    TempDir dir;
    auto payload = dir.path / "lists.dat";
    write_file(payload, disclosure_bytes(kSecretId));
    // Absolute stored path. SAME_DIR keeps only the basename and joins it
    // onto the index directory. That file must still not be modified.
    auto bytes = serialize_flat_ivf(payload.string(), 1, 1, kSlotBytes);
    auto index_path = dir.path / "index.faiss";
    write_file(index_path, bytes);

    FileState before = capture(payload);
    std::unique_ptr<faiss::Index> index;
    try {
        index.reset(
                faiss::read_index(
                        index_path.c_str(), faiss::IO_FLAG_ONDISK_SAME_DIR));
    } catch (const faiss::FaissException&) {
    }
    expect_unchanged(before, payload);
    if (!index) {
        return;
    }
    try {
        float added[kDim] = {1.f, 2.f};
        index->add(1, added);
    } catch (const faiss::FaissException&) {
    }
    expect_unchanged(before, payload);
}

TEST(OnDiskEmbeddedPath, SymlinkTargetIsNotTheOpenedObject) {
    TempDir dir;
    TempDir outside;
    auto secret = outside.path / "secret";
    write_file(secret, disclosure_bytes(kSecretId));
    auto link = dir.path / "lists.dat";
    ASSERT_EQ(symlink(secret.c_str(), link.c_str()), 0);
    auto bytes = serialize_flat_ivf(link.string(), 1, 1, kSlotBytes);
    auto index_path = dir.path / "index.faiss";
    write_file(index_path, bytes);

    expect_float_search_hides_file(
            [&]() { return faiss::read_index(index_path.c_str(), 0); },
            secret,
            true);
    expect_float_search_hides_file(
            [&]() {
                return faiss::read_index(
                        index_path.c_str(), faiss::IO_FLAG_ONDISK_SAME_DIR);
            },
            secret,
            true);
}

TEST(OnDiskEmbeddedPath, SameDirSymlinkIsNotReresolved) {
    if (getenv("FAISS_OD_RELOOKUP") != nullptr) {
        const char* index_path = getenv("FAISS_OD_INDEX");
        const char* secret = getenv("FAISS_OD_SECRET");
        ASSERT_NE(index_path, nullptr);
        ASSERT_NE(secret, nullptr);
        FileState before = capture(secret);
        std::unique_ptr<faiss::Index> index;
        faiss::idx_t label = -1;
        try {
            index.reset(
                    faiss::read_index(
                            index_path, faiss::IO_FLAG_ONDISK_SAME_DIR));
            if (index) {
                label = search_label(*index);
            }
        } catch (const faiss::FaissException&) {
        }
        EXPECT_NE(label, kSecretId);
        expect_unchanged(before, secret);
        return;
    }

#ifndef FAISS_OD_RELOOKUP_PRELOAD
    FAIL() << "relookup preload was not built";
#else
    TempDir dir;
    TempDir outside;
    auto safe = dir.path / "safe_payload";
    auto secret = outside.path / "secret";
    write_file(safe, disclosure_bytes(kSafeId));
    write_file(secret, disclosure_bytes(kSecretId));
    auto link = dir.path / "faiss_od_relookup_link";
    ASSERT_EQ(symlink(safe.c_str(), link.c_str()), 0);
    // Stored name is absolute. SAME_DIR reduces it to the basename and
    // joins that onto the index directory, which is the symlink above.
    auto bytes = serialize_flat_ivf(link.string(), 1, 1, kSlotBytes);
    auto index_path = dir.path / "index.faiss";
    write_file(index_path, bytes);

    pid_t pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        setenv("FAISS_OD_RELOOKUP", "1", 1);
        setenv("FAISS_OD_RELOOKUP_PRELOAD_ACTIVE", "1", 1);
        setenv("LD_PRELOAD", FAISS_OD_RELOOKUP_PRELOAD, 1);
        setenv("FAISS_OD_LINK", link.c_str(), 1);
        setenv("FAISS_OD_SECRET", secret.c_str(), 1);
        setenv("FAISS_OD_INDEX", index_path.c_str(), 1);
        execl("/proc/self/exe",
              "faiss_test",
              "--gtest_filter=OnDiskEmbeddedPath.SameDirSymlinkIsNotReresolved",
              static_cast<char*>(nullptr));
        _exit(127);
    }
    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status)) << status;
    EXPECT_EQ(WIFEXITED(status) ? WEXITSTATUS(status) : -1, 0) << status;
#endif
}
