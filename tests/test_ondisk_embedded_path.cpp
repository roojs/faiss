/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Spec for on-disk inverted-list filenames stored inside an index.
//
// Loading an index must not open, map, or modify a file chosen by bytes inside
// that index, except a single relative path component placed in the same
// directory as the index file. These tests fail while an embedded absolute
// path, parent-directory component, cwd-relative name, or symlink is followed.

#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryIVF.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>
#include <faiss/invlists/OnDiskInvertedLists.h>

#include <gtest/gtest.h>

#include <fcntl.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

extern char** environ;

namespace {

constexpr faiss::idx_t kMarker = 0x1122334455667788LL;
constexpr faiss::idx_t kFiller = 0x2222222222222222LL;
constexpr size_t kFileBytes = 4096;
constexpr size_t kIdOffset = 8;

struct TrapState {
    int armed = 0;
    int checks = 0;
    int opens = 0;
    int in_realpath = 0;
    int swaps = 0;
    char watch[PATH_MAX] = {};
    char base[PATH_MAX] = {};
    char secret[PATH_MAX] = {};
};

TrapState g_trap;

int path_is_watch(const char* path) {
    if (!g_trap.armed || path == nullptr || g_trap.watch[0] == '\0') {
        return 0;
    }
    if (strcmp(path, g_trap.watch) == 0) {
        return 1;
    }
    if (g_trap.base[0] != '\0' && strcmp(path, g_trap.base) == 0) {
        return 1;
    }
    if (g_trap.base[0] != '\0' && strncmp(path, "./", 2) == 0 &&
        strcmp(path + 2, g_trap.base) == 0) {
        return 1;
    }
    return 0;
}

void note_check(const char* path) {
    if (path_is_watch(path)) {
        g_trap.checks++;
    }
}

void replace_with_symlink() {
    unlink(g_trap.watch);
    if (symlink(g_trap.secret, g_trap.watch) == 0) {
        g_trap.swaps++;
    }
}

// A containment check that looks the name up, then opens it again, must not
// observe a directory entry swapped between the two lookups. The first open
// sees the original file. A later open, or an open after stat/realpath, sees
// a symlink to the file outside the index directory.
void note_open(const char* path) {
    if (!path_is_watch(path) || g_trap.in_realpath) {
        return;
    }
    g_trap.opens++;
    if (g_trap.checks > 0 || g_trap.opens > 1) {
        replace_with_symlink();
    }
}

struct CwdGuard {
    char previous[PATH_MAX];
    CwdGuard() {
        if (getcwd(previous, sizeof(previous)) == nullptr) {
            previous[0] = '\0';
        }
    }
    ~CwdGuard() {
        if (previous[0] != '\0') {
            if (chdir(previous) != 0) {
                // Restoring the process cwd is best-effort at teardown.
            }
        }
    }
};

struct TempDir {
    std::string path;
    TempDir() {
        std::string pattern = "/tmp/faiss_od_XXXXXX";
        if (mkdtemp(pattern.data()) == nullptr) {
            throw std::runtime_error("mkdtemp failed");
        }
        path = pattern;
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
    std::string child(const std::string& name) const {
        return path + "/" + name;
    }
};

std::vector<uint8_t> slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    return std::vector<uint8_t>(
            (std::istreambuf_iterator<char>(in)),
            std::istreambuf_iterator<char>());
}

void write_bytes(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open " + path);
    }
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    if (!out) {
        throw std::runtime_error("failed to write " + path);
    }
}

std::vector<uint8_t> payload_with_id(faiss::idx_t id) {
    std::vector<uint8_t> bytes(kFileBytes, 0x5a);
    // The scanned code is a 0 float so the neighbor id is accepted.
    bytes[0] = bytes[1] = bytes[2] = bytes[3] = 0;
    memcpy(bytes.data() + kIdOffset, &id, sizeof(id));
    return bytes;
}

bool exists_now(const std::string& path) {
    struct stat st;
    return lstat(path.c_str(), &st) == 0;
}

void arm_trap(const std::string& watch, const std::string& secret) {
    memset(&g_trap, 0, sizeof(g_trap));
    snprintf(g_trap.watch, sizeof(g_trap.watch), "%s", watch.c_str());
    snprintf(g_trap.secret, sizeof(g_trap.secret), "%s", secret.c_str());
    const char* slash = strrchr(g_trap.watch, '/');
    snprintf(
            g_trap.base,
            sizeof(g_trap.base),
            "%s",
            slash == nullptr ? g_trap.watch : slash + 1);
    g_trap.armed = 1;
}

void disarm_trap() {
    g_trap.armed = 0;
}

void configure_flat_lists(faiss::OnDiskInvertedLists& od) {
    od.lists[0].offset = 0;
    od.lists[0].capacity = 2; // ids begin at code_size * capacity == 8
    od.lists[0].size = 1;
    od.totsize = kFileBytes;
}

std::vector<uint8_t> build_flat_index(const std::string& embedded) {
    faiss::IndexFlatL2 quantizer(1);
    float centroid = 0.f;
    quantizer.add(1, &centroid);
    faiss::IndexIVFFlat index(&quantizer, 1, 1);
    faiss::OnDiskInvertedLists od(
            index.nlist, index.code_size, embedded.c_str());
    index.nprobe = 1;
    index.ntotal = 1;
    index.is_trained = true;
    index.replace_invlists(&od, false);
    configure_flat_lists(od);
    faiss::VectorIOWriter writer;
    faiss::write_index(&index, &writer);
    return writer.data;
}

std::vector<uint8_t> build_binary_index(const std::string& embedded) {
    faiss::IndexBinaryFlat quantizer(8);
    uint8_t centroid = 0;
    quantizer.add(1, &centroid);
    faiss::IndexBinaryIVF index(&quantizer, 8, 1);
    faiss::OnDiskInvertedLists od(1, 1, embedded.c_str());
    index.nprobe = 1;
    index.ntotal = 1;
    index.is_trained = true;
    index.replace_invlists(&od, false);
    od.lists[0].offset = 0;
    od.lists[0].capacity = kIdOffset; // code_size 1, ids begin at offset 8
    od.lists[0].size = 1;
    od.totsize = kFileBytes;
    faiss::VectorIOWriter writer;
    faiss::write_index_binary(&index, &writer);
    return writer.data;
}

void write_index_file(const std::string& path, const std::vector<uint8_t>& bytes) {
    write_bytes(path, bytes);
}

struct FloatProbe {
    bool threw = false;
    std::string error;
    faiss::idx_t label = -1;
    bool saw_marker_ids = false;
};

FloatProbe probe_flat(faiss::Index* index) {
    FloatProbe probe;
    auto* ivf = dynamic_cast<faiss::IndexIVF*>(index);
    if (ivf != nullptr && ivf->invlists != nullptr &&
        ivf->invlists->list_size(0) >= 1) {
        const faiss::idx_t* ids = ivf->invlists->get_ids(0);
        if (ids != nullptr && ids[0] == kMarker) {
            probe.saw_marker_ids = true;
        }
    }
    float query = 0.f;
    float distance = 0.f;
    faiss::idx_t label = -1;
    index->search(1, &query, 1, &distance, &label);
    probe.label = label;
    return probe;
}

template <typename Loader>
FloatProbe load_and_probe(Loader&& loader) {
    FloatProbe probe;
    try {
        std::unique_ptr<faiss::Index> index(loader());
        if (!index) {
            probe.threw = true;
            probe.error = "null index";
            return probe;
        }
        probe = probe_flat(index.get());
    } catch (const std::exception& ex) {
        probe.threw = true;
        probe.error = ex.what();
    }
    return probe;
}

// index->add throws inside an OpenMP region when the lists are read-only,
// which terminates the process. Run the add in a child so a refusal does not
// abort the suite. The file contents are the assertion.
int spawn_add(const std::string& index_path, int flags, const char* kind) {
    std::string index_env = "OD_ADD_INDEX=" + index_path;
    std::string flags_env = "OD_ADD_FLAGS=" + std::to_string(flags);
    std::string kind_env = std::string("OD_ADD_KIND=") + kind;
    std::vector<std::string> storage;
    for (char** entry = environ; entry != nullptr && *entry != nullptr; ++entry) {
        std::string item(*entry);
        if (item.rfind("OD_ADD_", 0) == 0) {
            continue;
        }
        storage.push_back(std::move(item));
    }
    storage.push_back(std::move(index_env));
    storage.push_back(std::move(flags_env));
    storage.push_back(std::move(kind_env));
    std::vector<char*> envp;
    envp.reserve(storage.size() + 1);
    for (std::string& item : storage) {
        envp.push_back(item.data());
    }
    envp.push_back(nullptr);

    char arg0[] = "/proc/self/exe";
    char arg1[] = "--gtest_filter=OnDiskEmbeddedPath.MutationChild";
    char* argv[] = {arg0, arg1, nullptr};
    pid_t pid = 0;
    int rc = posix_spawn(&pid, arg0, nullptr, nullptr, argv, envp.data());
    if (rc != 0) {
        return -1;
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        return -1;
    }
    return status;
}

struct BinaryProbe {
    bool threw = false;
    std::string error;
    faiss::idx_t label = -1;
    bool saw_marker_ids = false;
};

BinaryProbe load_and_probe_binary(
        const std::function<faiss::IndexBinary*()>& loader) {
    BinaryProbe probe;
    try {
        std::unique_ptr<faiss::IndexBinary> index(loader());
        if (!index) {
            probe.threw = true;
            probe.error = "null index";
            return probe;
        }
        auto* ivf = dynamic_cast<faiss::IndexBinaryIVF*>(index.get());
        if (ivf != nullptr && ivf->invlists != nullptr &&
            ivf->invlists->list_size(0) >= 1) {
            const faiss::idx_t* ids = ivf->invlists->get_ids(0);
            if (ids != nullptr && ids[0] == kMarker) {
                probe.saw_marker_ids = true;
            }
        }
        uint8_t query = 0;
        int32_t distance = 0;
        faiss::idx_t label = -1;
        index->search(1, &query, 1, &distance, &label);
        probe.label = label;
    } catch (const std::exception& ex) {
        probe.threw = true;
        probe.error = ex.what();
    }
    return probe;
}

void expect_no_disclosure(const FloatProbe& probe, const std::string& where) {
    EXPECT_NE(probe.label, kMarker)
            << where << " search label is the 8 bytes at offset " << kIdOffset
            << " of the companion file";
    EXPECT_FALSE(probe.saw_marker_ids)
            << where << " inverted-list ids point at the companion file";
}

void expect_unchanged(
        const std::string& path,
        const std::vector<uint8_t>& before,
        const std::string& where) {
    EXPECT_TRUE(exists_now(path)) << where << " removed " << path;
    EXPECT_EQ(before, slurp(path)) << where << " modified " << path;
}

TEST(OnDiskEmbeddedPath, MutationChild) {
    const char* path = getenv("OD_ADD_INDEX");
    if (path == nullptr) {
        GTEST_SKIP();
    }
    int flags = 0;
    if (const char* flags_env = getenv("OD_ADD_FLAGS")) {
        flags = std::atoi(flags_env);
    }
    const char* kind = getenv("OD_ADD_KIND");
    if (kind != nullptr && std::strcmp(kind, "binary") == 0) {
        std::unique_ptr<faiss::IndexBinary> index(
                faiss::read_index_binary(path, flags));
        ASSERT_NE(index, nullptr);
        uint8_t added = 1;
        index->add(1, &added);
        return;
    }
    std::unique_ptr<faiss::Index> index(faiss::read_index(path, flags));
    ASSERT_NE(index, nullptr);
    float added = 1.f;
    index->add(1, &added);
}

} // namespace

extern "C" {

FILE* __real_fopen(const char* path, const char* mode);
FILE* __wrap_fopen(const char* path, const char* mode) {
    note_open(path);
    return __real_fopen(path, mode);
}

FILE* __real_fopen64(const char* path, const char* mode);
FILE* __wrap_fopen64(const char* path, const char* mode) {
    note_open(path);
    return __real_fopen64(path, mode);
}

int __real_open(const char* path, int flags, ...);
int __wrap_open(const char* path, int flags, ...) {
    mode_t mode = 0;
    if ((flags & O_CREAT) != 0) {
        va_list ap;
        va_start(ap, flags);
        mode = static_cast<mode_t>(va_arg(ap, int));
        va_end(ap);
    }
    note_open(path);
    return __real_open(path, flags, mode);
}

int __real_open64(const char* path, int flags, ...);
int __wrap_open64(const char* path, int flags, ...) {
    mode_t mode = 0;
    if ((flags & O_CREAT) != 0) {
        va_list ap;
        va_start(ap, flags);
        mode = static_cast<mode_t>(va_arg(ap, int));
        va_end(ap);
    }
    note_open(path);
    return __real_open64(path, flags, mode);
}

int __real_openat(int dirfd, const char* path, int flags, ...);
int __wrap_openat(int dirfd, const char* path, int flags, ...) {
    mode_t mode = 0;
    if ((flags & O_CREAT) != 0) {
        va_list ap;
        va_start(ap, flags);
        mode = static_cast<mode_t>(va_arg(ap, int));
        va_end(ap);
    }
    note_open(path);
    return __real_openat(dirfd, path, flags, mode);
}

int __real_openat64(int dirfd, const char* path, int flags, ...);
int __wrap_openat64(int dirfd, const char* path, int flags, ...) {
    mode_t mode = 0;
    if ((flags & O_CREAT) != 0) {
        va_list ap;
        va_start(ap, flags);
        mode = static_cast<mode_t>(va_arg(ap, int));
        va_end(ap);
    }
    note_open(path);
    return __real_openat64(dirfd, path, flags, mode);
}

int __real_stat(const char* path, struct stat* buf);
int __wrap_stat(const char* path, struct stat* buf) {
    note_check(path);
    return __real_stat(path, buf);
}

int __real_lstat(const char* path, struct stat* buf);
int __wrap_lstat(const char* path, struct stat* buf) {
    note_check(path);
    return __real_lstat(path, buf);
}

int __real_fstatat(int dirfd, const char* path, struct stat* buf, int flags);
int __wrap_fstatat(int dirfd, const char* path, struct stat* buf, int flags) {
    note_check(path);
    return __real_fstatat(dirfd, path, buf, flags);
}

char* __real_realpath(const char* path, char* resolved);
char* __wrap_realpath(const char* path, char* resolved) {
    int watched = path_is_watch(path);
    if (watched) {
        g_trap.in_realpath++;
        g_trap.checks++;
    }
    char* result = __real_realpath(path, resolved);
    if (watched) {
        g_trap.in_realpath--;
    }
    return result;
}

int __real_statx(
        int dirfd,
        const char* path,
        int flags,
        unsigned int mask,
        struct statx* buf);
int __wrap_statx(
        int dirfd,
        const char* path,
        int flags,
        unsigned int mask,
        struct statx* buf) {
    note_check(path);
    return __real_statx(dirfd, path, flags, mask, buf);
}

} // extern "C"

namespace {

TEST(OnDiskEmbeddedPath, AbsolutePathIsNotMappedOrModified) {
    TempDir root;
    const std::string victim = root.child("customer_tokens");
    const std::string index_path = root.child("upload.faiss");
    const auto original = payload_with_id(kMarker);
    write_bytes(victim, original);
    write_index_file(index_path, build_flat_index(victim));

    FloatProbe probe = load_and_probe([&]() {
        return faiss::read_index(index_path.c_str(), 0);
    });
    expect_no_disclosure(probe, "read_index default flags");
    spawn_add(index_path, 0, "flat");
    expect_unchanged(victim, original, "read_index default flags");
}

TEST(OnDiskEmbeddedPath, ReadOnlyFlagDoesNotMapAbsolutePath) {
    TempDir root;
    const std::string victim = root.child("customer_tokens");
    const std::string index_path = root.child("upload.faiss");
    const auto original = payload_with_id(kMarker);
    write_bytes(victim, original);
    write_index_file(index_path, build_flat_index(victim));

    FloatProbe probe = load_and_probe([&]() {
        return faiss::read_index(index_path.c_str(), faiss::IO_FLAG_READ_ONLY);
    });
    expect_no_disclosure(probe, "IO_FLAG_READ_ONLY");
    expect_unchanged(victim, original, "IO_FLAG_READ_ONLY");
}

TEST(OnDiskEmbeddedPath, AbsolutePathInsideIndexDirectoryIsRejected) {
    TempDir root;
    std::filesystem::create_directory(root.child("idx"));
    const std::string victim = root.child("idx/lists.ivfdata");
    const std::string index_path = root.child("idx/upload.faiss");
    const auto original = payload_with_id(kMarker);
    write_bytes(victim, original);
    write_index_file(index_path, build_flat_index(victim));

    FloatProbe probe = load_and_probe([&]() {
        return faiss::read_index(index_path.c_str(), 0);
    });
    expect_no_disclosure(probe, "absolute companion inside the index directory");
    expect_unchanged(victim, original, "absolute companion inside the index directory");
}

TEST(OnDiskEmbeddedPath, FileAndMemoryReadersFollowNoAbsolutePath) {
    TempDir root;
    const std::string victim = root.child("customer_tokens");
    const std::string index_path = root.child("upload.faiss");
    const auto original = payload_with_id(kMarker);
    write_bytes(victim, original);
    const auto bytes = build_flat_index(victim);
    write_index_file(index_path, bytes);

    {
        FILE* f = fopen(index_path.c_str(), "rb");
        ASSERT_NE(f, nullptr);
        FloatProbe probe = load_and_probe([&]() {
            return faiss::read_index(f, 0);
        });
        fclose(f);
        expect_no_disclosure(probe, "read_index(FILE*)");
        expect_unchanged(victim, original, "read_index(FILE*)");
    }
    {
        faiss::VectorIOReader reader;
        reader.data = bytes;
        FloatProbe probe = load_and_probe([&]() {
            return faiss::read_index(&reader, 0);
        });
        expect_no_disclosure(probe, "read_index(IOReader*)");
        expect_unchanged(victim, original, "read_index(IOReader*)");
    }
}

TEST(OnDiskEmbeddedPath, ParentDirectoryComponentDoesNotEscape) {
    TempDir root;
    std::filesystem::create_directory(root.child("idx"));
    const std::string victim = root.child("customer_tokens");
    const std::string index_path = root.child("idx/upload.faiss");
    const auto original = payload_with_id(kMarker);
    write_bytes(victim, original);
    const std::string embedded = root.child("idx") + "/../customer_tokens";
    write_index_file(index_path, build_flat_index(embedded));

    FloatProbe probe = load_and_probe([&]() {
        return faiss::read_index(index_path.c_str(), 0);
    });
    expect_no_disclosure(probe, "embedded .. component");
    expect_unchanged(victim, original, "embedded .. component");
}

TEST(OnDiskEmbeddedPath, MissingCompanionIsNotCreated) {
    TempDir root;
    std::filesystem::create_directory(root.child("idx"));
    const std::string missing = root.child("idx/does_not_exist.ivfdata");
    const std::string index_path = root.child("idx/upload.faiss");
    ASSERT_FALSE(exists_now(missing));
    write_index_file(index_path, build_flat_index(missing));

    try {
        std::unique_ptr<faiss::Index> index(faiss::read_index(index_path.c_str(), 0));
        if (index) {
            float query = 0.f;
            float distance = 0.f;
            faiss::idx_t label = -1;
            index->search(1, &query, 1, &distance, &label);
        }
    } catch (const std::exception&) {
    }
    spawn_add(index_path, 0, "flat");
    EXPECT_FALSE(exists_now(missing))
            << "loading created " << missing << " before refusing it";
}

TEST(OnDiskEmbeddedPath, RelativeSymlinkOutsideDirectoryIsNotFollowed) {
    CwdGuard cwd;
    TempDir root;
    std::filesystem::create_directory(root.child("idx"));
    const std::string secret = root.child("customer_tokens");
    const std::string link = root.child("idx/lists.ivfdata");
    const std::string index_path = root.child("idx/upload.faiss");
    const auto original = payload_with_id(kMarker);
    write_bytes(secret, original);
    ASSERT_EQ(symlink(secret.c_str(), link.c_str()), 0);
    write_index_file(index_path, build_flat_index("lists.ivfdata"));
    ASSERT_EQ(chdir(root.child("idx").c_str()), 0);

    FloatProbe probe = load_and_probe([&]() {
        return faiss::read_index(index_path.c_str(), 0);
    });
    expect_no_disclosure(probe, "relative symlink");
    expect_unchanged(secret, original, "relative symlink");
}

TEST(OnDiskEmbeddedPath, SameDirFlagDoesNotFollowSymlink) {
    TempDir root;
    std::filesystem::create_directory(root.child("idx"));
    const std::string secret = root.child("customer_tokens");
    const std::string link = root.child("idx/lists.ivfdata");
    const std::string index_path = root.child("idx/upload.faiss");
    const auto original = payload_with_id(kMarker);
    write_bytes(secret, original);
    ASSERT_EQ(symlink(secret.c_str(), link.c_str()), 0);
    write_index_file(
            index_path, build_flat_index("/unused/absolute/lists.ivfdata"));

    FloatProbe probe = load_and_probe([&]() {
        return faiss::read_index(
                index_path.c_str(), faiss::IO_FLAG_ONDISK_SAME_DIR);
    });
    expect_no_disclosure(probe, "IO_FLAG_ONDISK_SAME_DIR symlink");
    expect_unchanged(secret, original, "IO_FLAG_ONDISK_SAME_DIR symlink");
}

TEST(OnDiskEmbeddedPath, MemoryReaderDoesNotOpenCwdRelativeName) {
    CwdGuard cwd;
    TempDir root;
    const std::string victim = root.child("tokenstore");
    const auto original = payload_with_id(kMarker);
    write_bytes(victim, original);
    faiss::VectorIOReader reader;
    reader.data = build_flat_index("tokenstore");
    ASSERT_EQ(chdir(root.path.c_str()), 0);

    FloatProbe probe = load_and_probe([&]() {
        return faiss::read_index(&reader, 0);
    });
    expect_no_disclosure(probe, "VectorIOReader relative name");
    expect_unchanged(victim, original, "VectorIOReader relative name");
}

TEST(OnDiskEmbeddedPath, DirectInvertedListReadDoesNotMapAbsolutePath) {
    TempDir root;
    const std::string victim = root.child("customer_tokens");
    const auto original = payload_with_id(kMarker);
    write_bytes(victim, original);

    faiss::IndexFlatL2 quantizer(1);
    float centroid = 0.f;
    quantizer.add(1, &centroid);
    faiss::IndexIVFFlat index(&quantizer, 1, 1);
    faiss::OnDiskInvertedLists od(index.nlist, index.code_size, victim.c_str());
    index.replace_invlists(&od, false);
    configure_flat_lists(od);
    faiss::VectorIOWriter writer;
    faiss::write_InvertedLists(&od, &writer);

    bool saw = false;
    try {
        faiss::VectorIOReader reader;
        reader.data = writer.data;
        std::unique_ptr<faiss::InvertedLists> lists(
                faiss::read_InvertedLists(&reader, 0));
        if (lists && lists->list_size(0) >= 1) {
            const faiss::idx_t* ids = lists->get_ids(0);
            saw = ids != nullptr && ids[0] == kMarker;
        }
    } catch (const std::exception&) {
        saw = false;
    }
    EXPECT_FALSE(saw) << "read_InvertedLists mapped the embedded absolute path";
    expect_unchanged(victim, original, "read_InvertedLists");
}

TEST(OnDiskEmbeddedPath, BinaryIvfDoesNotMapOrModifyAbsolutePath) {
    TempDir root;
    const std::string victim = root.child("customer_tokens");
    const std::string index_path = root.child("upload.faiss");
    const auto original = payload_with_id(kMarker);
    write_bytes(victim, original);
    const auto bytes = build_binary_index(victim);
    write_index_file(index_path, bytes);

    auto check = [&](const BinaryProbe& probe, const char* where) {
        EXPECT_NE(probe.label, kMarker) << where << " returned the companion id";
        EXPECT_FALSE(probe.saw_marker_ids) << where << " mapped the companion";
        expect_unchanged(victim, original, where);
    };

    check(load_and_probe_binary([&]() {
              return faiss::read_index_binary(index_path.c_str(), 0);
          }),
          "read_index_binary");

    {
        FILE* f = fopen(index_path.c_str(), "rb");
        ASSERT_NE(f, nullptr);
        check(load_and_probe_binary([&]() {
                  return faiss::read_index_binary(f, 0);
              }),
              "read_index_binary(FILE*)");
        fclose(f);
    }
    {
        faiss::VectorIOReader reader;
        reader.data = bytes;
    check(load_and_probe_binary([&]() {
              return faiss::read_index_binary(&reader, 0);
          }),
          "read_index_binary(IOReader*)");
    }
    spawn_add(index_path, 0, "binary");
    expect_unchanged(victim, original, "binary add");
}

TEST(OnDiskEmbeddedPath, DefaultReadDoesNotMutateSameDirectoryCompanion) {
    CwdGuard cwd;
    TempDir root;
    std::filesystem::create_directory(root.child("idx"));
    const std::string companion = root.child("idx/lists.ivfdata");
    const std::string index_path = root.child("idx/upload.faiss");
    const auto original = payload_with_id(kFiller);
    write_bytes(companion, original);
    write_index_file(index_path, build_flat_index("lists.ivfdata"));
    ASSERT_EQ(chdir(root.child("idx").c_str()), 0);

    spawn_add(index_path, 0, "flat");
    expect_unchanged(companion, original, "default read of same-directory list");
}

TEST(OnDiskEmbeddedPath, SameDirectoryRelativeListStaysReadable) {
    CwdGuard cwd;
    TempDir root;
    std::filesystem::create_directory(root.child("idx"));
    const std::string companion = root.child("idx/lists.ivfdata");
    const std::string index_path = root.child("idx/upload.faiss");
    const auto original = payload_with_id(kMarker);
    write_bytes(companion, original);
    write_index_file(index_path, build_flat_index("lists.ivfdata"));
    ASSERT_EQ(chdir(root.child("idx").c_str()), 0);

    FloatProbe probe = load_and_probe([&]() {
        return faiss::read_index(index_path.c_str(), 0);
    });
    EXPECT_FALSE(probe.threw) << probe.error;
    EXPECT_EQ(probe.label, kMarker);
    EXPECT_TRUE(probe.saw_marker_ids);
    spawn_add(index_path, 0, "flat");
    // The default load may read the companion. It must not modify it.
    expect_unchanged(companion, original, "same-directory relative list");
}

TEST(OnDiskEmbeddedPath, SecondLookupDoesNotObserveSwappedSymlink) {
    CwdGuard cwd;
    TempDir root;
    std::filesystem::create_directory(root.child("idx"));
    const std::string secret = root.child("customer_tokens");
    const std::string watch = root.child("idx/od_payload_a533");
    const std::string index_path = root.child("idx/upload.faiss");
    const auto secret_bytes = payload_with_id(kMarker);
    const auto safe_bytes = payload_with_id(kFiller);
    write_bytes(secret, secret_bytes);
    write_bytes(watch, safe_bytes);
    write_index_file(index_path, build_flat_index("od_payload_a533"));

    arm_trap(watch, secret);
    struct stat st_before;
    ASSERT_EQ(stat(watch.c_str(), &st_before), 0);
    FILE* touched = fopen(watch.c_str(), "rb");
    if (touched != nullptr) {
        fclose(touched);
    }
    struct stat st_after;
    ASSERT_EQ(lstat(watch.c_str(), &st_after), 0);
    EXPECT_GT(g_trap.swaps, 0);
    EXPECT_TRUE(S_ISLNK(st_after.st_mode))
            << "lookup interposition did not swap the companion between checks";
    disarm_trap();
    // The swap leaves a symlink. Unlink it before restoring the regular file
    // so the write does not follow the link into the outside file.
    ASSERT_EQ(unlink(watch.c_str()), 0);
    write_bytes(watch, safe_bytes);
    memset(&g_trap, 0, sizeof(g_trap));
    arm_trap(watch, secret);
    ASSERT_EQ(chdir(root.child("idx").c_str()), 0);

    FloatProbe probe = load_and_probe([&]() {
        return faiss::read_index(index_path.c_str(), 0);
    });
    disarm_trap();
    expect_no_disclosure(probe, "second lookup of the companion name");
    expect_unchanged(secret, secret_bytes, "second lookup of the companion name");
}

} // namespace
