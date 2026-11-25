/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Spec for MappedFileIOReader short reads. These tests fail while a short
// tail is rounded up to a whole element: the cursor moves past EOF, and the
// next read underflows the remaining length and copies it into the caller
// buffer. A fixed reader returns only complete elements, does not advance
// past the bytes those elements occupy, and returns a short read when the
// cursor is already past EOF.
//
// Exercised product paths, all through IO_FLAG_MMAP_IFC:
//   - read_index(path) and read_index(FILE*) for IndexPreTransform + Viqt
//   - read_index for IndexPreTransform + Pcam (epsilon, then PCAMat)
//   - read_index for IndexPreTransform + Viqm (seed int, then have_bias)
//   - read_index for an IndexFlat header (idx_t field, then is_trained)
//   - read_index_binary for an IndexBinaryFlat header (ntotal, then is_trained)
//   - MappedFileIOReader::operator() and ::mmap directly
//
// Not given their own index files, because they call the same two methods
// with the element sizes already covered above: other READ1_BOOL sites in
// faiss/impl/index_read.cpp, HNSW / PiPNN / inverted-list views (mmap of
// int32 or uint8, whose following read is not a smaller element), and
// ZeroCopyIOReader. ZeroCopyIOReader::get_data_view rounds like mmap(), but
// ZeroCopyIOReader::operator() returns 0 when rp_ >= total_ and does not
// take this memcpy path. SVS acquire_mmap_span is not built in the default
// configuration.

#include <gtest/gtest.h>

#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/io.h>
#include <faiss/impl/mapped_io.h>
#include <faiss/index_io.h>

namespace {

struct ProcResult {
    bool exited = false;
    int code = -1;
    bool signaled = false;
    int sig = 0;
};

ProcResult run_child(const std::function<void()>& fn) {
    pid_t pid = fork();
    EXPECT_NE(pid, -1) << std::strerror(errno);
    if (pid < 0) {
        return {};
    }
    if (pid == 0) {
        signal(SIGSEGV, SIG_DFL);
        signal(SIGBUS, SIG_DFL);
        signal(SIGABRT, SIG_DFL);
        try {
            fn();
            _exit(0);
        } catch (const faiss::FaissException&) {
            _exit(2);
        } catch (...) {
            _exit(3);
        }
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        ADD_FAILURE() << "waitpid: " << std::strerror(errno);
        return {};
    }
    ProcResult result;
    if (WIFEXITED(status)) {
        result.exited = true;
        result.code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        result.signaled = true;
        result.sig = WTERMSIG(status);
    }
    return result;
}

// ADD_FAILURE rather than FAIL/ASSERT so a helper can record the result.
// A signal (the overflow) or any exit status other than `expected` fails
// the test. Status 2 is the FaissException path used by the index loaders.
#define EXPECT_CHILD_CODE(result, expected)                             \
    do {                                                                \
        const auto& _child_result = (result);                           \
        if (_child_result.signaled) {                                   \
            ADD_FAILURE() << "child aborted with signal "               \
                          << _child_result.sig;                         \
        } else if (!_child_result.exited) {                             \
            ADD_FAILURE() << "child did not exit";                      \
        } else if (_child_result.code != (expected)) {                  \
            ADD_FAILURE() << "child exit status " << _child_result.code \
                          << ", expected " << (expected);               \
        }                                                               \
    } while (0)

class TempFile {
   public:
    explicit TempFile(const uint8_t* data, size_t n) {
        char tmpl[] = "/tmp/faiss-mmap-round-XXXXXX";
        int fd = mkstemp(tmpl);
        if (fd < 0) {
            throw std::runtime_error(std::strerror(errno));
        }
        path_ = tmpl;
        size_t off = 0;
        while (off < n) {
            ssize_t wrote = ::write(fd, data + off, n - off);
            if (wrote < 0) {
                int err = errno;
                ::close(fd);
                throw std::runtime_error(std::strerror(err));
            }
            off += static_cast<size_t>(wrote);
        }
        if (::close(fd) != 0) {
            throw std::runtime_error(std::strerror(errno));
        }
    }

    explicit TempFile(const std::vector<uint8_t>& data)
            : TempFile(data.data(), data.size()) {}

    ~TempFile() {
        if (!path_.empty()) {
            ::unlink(path_.c_str());
        }
    }

    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    const std::string& path() const {
        return path_;
    }

   private:
    std::string path_;
};

size_t find_fourcc(const std::vector<uint8_t>& buf, const char tag[4]) {
    uint32_t want = faiss::fourcc(tag);
    size_t found = static_cast<size_t>(-1);
    for (size_t i = 0; i + sizeof(want) <= buf.size(); i++) {
        uint32_t got = 0;
        std::memcpy(&got, buf.data() + i, sizeof(got));
        if (got == want) {
            if (found != static_cast<size_t>(-1)) {
                throw std::runtime_error(std::string("duplicate fourcc ") + tag);
            }
            found = i;
        }
    }
    if (found == static_cast<size_t>(-1)) {
        throw std::runtime_error(std::string("missing fourcc ") + tag);
    }
    return found;
}

void expect_room_in_page(size_t length) {
    long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) {
        throw std::runtime_error("sysconf(_SC_PAGESIZE) failed");
    }
    size_t page_off = length % static_cast<size_t>(page);
    if (page_off == 0 ||
        static_cast<size_t>(page) - page_off <= 16u) {
        throw std::runtime_error(
                "truncated file ends too close to a page boundary");
    }
}

std::vector<uint8_t> truncate_to(const std::vector<uint8_t>& full, size_t n) {
    if (n == 0 || n >= full.size()) {
        throw std::runtime_error("truncation point is outside the file");
    }
    expect_room_in_page(n);
    return std::vector<uint8_t>(full.begin(), full.begin() + n);
}

size_t read_size_t(const std::vector<uint8_t>& buf, size_t off) {
    size_t value = 0;
    if (off + sizeof(value) > buf.size()) {
        throw std::runtime_error("size field extends past the serialized index");
    }
    std::memcpy(&value, buf.data() + off, sizeof(value));
    return value;
}

std::vector<uint8_t> serialize_index(const faiss::Index* index) {
    faiss::VectorIOWriter writer;
    faiss::write_index(index, &writer);
    return writer.data;
}

std::vector<uint8_t> itq_pretransform_bytes() {
    faiss::IndexFlatL2 flat(1);
    auto itq = std::make_unique<faiss::ITQTransform>(1, 1, false);
    itq->mean = {1.f};
    itq->is_trained = true;
    faiss::IndexPreTransform ix(itq.get(), &flat);
    return serialize_index(&ix);
}

std::vector<uint8_t> pca_pretransform_bytes() {
    auto pca = std::make_unique<faiss::PCAMatrix>(1, 1, 0.5f, true);
    pca->epsilon = 0.125f;
    pca->mean = {0.f};
    pca->eigenvalues = {1.f};
    pca->PCAMat = {1.f};
    pca->is_trained = true;
    faiss::IndexFlatL2 flat(1);
    faiss::IndexPreTransform ix(pca.get(), &flat);
    return serialize_index(&ix);
}

std::vector<uint8_t> itq_matrix_pretransform_bytes() {
    faiss::IndexFlatL2 flat(1);
    auto itqm = std::make_unique<faiss::ITQMatrix>(1);
    faiss::IndexPreTransform ix(itqm.get(), &flat);
    return serialize_index(&ix);
}

std::vector<uint8_t> flat_index_bytes() {
    faiss::IndexFlatL2 flat(4);
    return serialize_index(&flat);
}

std::vector<uint8_t> binary_flat_bytes() {
    faiss::IndexBinaryFlat flat(8);
    faiss::VectorIOWriter writer;
    faiss::write_index_binary(&flat, &writer);
    return writer.data;
}

size_t itq_mean_payload(const std::vector<uint8_t>& buf) {
    size_t at = find_fourcc(buf, "Viqt");
    size_t size_off = at + sizeof(uint32_t);
    EXPECT_EQ(read_size_t(buf, size_off), 1u);
    return size_off + sizeof(size_t);
}

size_t pca_epsilon_offset(const std::vector<uint8_t>& buf) {
    size_t at = find_fourcc(buf, "Pcam");
    return at + sizeof(uint32_t) + sizeof(float);
}

size_t pca_mat_payload(const std::vector<uint8_t>& buf) {
    size_t off = pca_epsilon_offset(buf) + sizeof(float);
    off += 1; // random_rotation
    off += sizeof(int); // balanced_bins
    for (int vec = 0; vec < 2; vec++) {
        size_t n = read_size_t(buf, off);
        off += sizeof(size_t) + n * sizeof(float);
    }
    EXPECT_EQ(read_size_t(buf, off), 1u);
    return off + sizeof(size_t);
}

size_t viqm_seed_offset(const std::vector<uint8_t>& buf) {
    size_t at = find_fourcc(buf, "Viqm");
    return at + sizeof(uint32_t) + sizeof(int); // skip max_iter
}

size_t flat_second_dummy_offset(const std::vector<uint8_t>& buf) {
    size_t at = find_fourcc(buf, "IxF2");
    EXPECT_EQ(at, 0u);
    return sizeof(uint32_t) + sizeof(int) + 2 * sizeof(faiss::idx_t);
}

size_t binary_ntotal_offset(const std::vector<uint8_t>& buf) {
    size_t at = find_fourcc(buf, "IBxF");
    EXPECT_EQ(at, 0u);
    return sizeof(uint32_t) + 2 * sizeof(int);
}

void expect_mmap_index_rejects(const std::vector<uint8_t>& bytes) {
    TempFile file(bytes);
    ProcResult result = run_child([&] {
        std::unique_ptr<faiss::Index> idx(
                faiss::read_index(file.path().c_str(), faiss::IO_FLAG_MMAP_IFC));
        (void)idx;
    });
    EXPECT_CHILD_CODE(result, 2);
}

void expect_mmap_index_file_rejects(const std::vector<uint8_t>& bytes) {
    TempFile file(bytes);
    ProcResult result = run_child([&] {
        FILE* f = std::fopen(file.path().c_str(), "rb");
        if (f == nullptr) {
            _exit(5);
        }
        std::unique_ptr<faiss::Index> idx(
                faiss::read_index(f, faiss::IO_FLAG_MMAP_IFC));
        std::fclose(f);
        (void)idx;
    });
    EXPECT_CHILD_CODE(result, 2);
}

void expect_mmap_binary_rejects(const std::vector<uint8_t>& bytes) {
    TempFile file(bytes);
    ProcResult result = run_child([&] {
        std::unique_ptr<faiss::IndexBinary> idx(faiss::read_index_binary(
                file.path().c_str(), faiss::IO_FLAG_MMAP_IFC));
        (void)idx;
    });
    EXPECT_CHILD_CODE(result, 2);
}

struct GuardedByte {
    void* map = nullptr;
    uint8_t* slot = nullptr;
    size_t bytes = 0;

    GuardedByte() {
        long page = sysconf(_SC_PAGESIZE);
        bytes = static_cast<size_t>(page) * 2;
        map = ::mmap(
                nullptr,
                bytes,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0);
        if (map == MAP_FAILED) {
            map = nullptr;
            return;
        }
        if (::mprotect(
                    static_cast<char*>(map) + page, page, PROT_NONE) != 0) {
            ::munmap(map, bytes);
            map = nullptr;
            return;
        }
        slot = static_cast<uint8_t*>(map) + page - 1;
        *slot = 0x5a;
    }

    ~GuardedByte() {
        if (map != nullptr) {
            ::munmap(map, bytes);
        }
    }
};

void child_partial_then_byte(
        const char* path,
        size_t elem,
        bool use_mmap,
        size_t nitems,
        size_t expect_n,
        size_t expect_pos_after_first,
        size_t expect_pos_after_byte,
        uint8_t expect_byte) {
    auto owner = std::make_shared<faiss::MmappedFileMappingOwner>(path);
    faiss::MappedFileIOReader reader(owner);
    std::vector<uint8_t> tmp(elem * nitems, 0);
    size_t n = 0;
    if (use_mmap) {
        void* view = nullptr;
        n = reader.mmap(&view, elem, nitems);
    } else {
        n = reader(&tmp[0], elem, nitems);
    }
    size_t pos_after_first = reader.pos;
    GuardedByte guard;
    if (guard.slot == nullptr) {
        _exit(6);
    }
    size_t nbyte = reader(guard.slot, 1, 1);
    if (n != expect_n || pos_after_first != expect_pos_after_first ||
        nbyte != 1 || reader.pos != expect_pos_after_byte ||
        *guard.slot != expect_byte) {
        _exit(10);
    }
    _exit(0);
}

} // namespace

TEST(MmapRoundedRead, CompleteFlatIndexLoads) {
    std::vector<uint8_t> bytes = flat_index_bytes();
    TempFile file(bytes);
    std::unique_ptr<faiss::Index> idx(
            faiss::read_index(file.path().c_str(), faiss::IO_FLAG_MMAP_IFC));
    ASSERT_NE(idx, nullptr);
    EXPECT_EQ(idx->d, 4);
    EXPECT_EQ(idx->ntotal, 0);
}

TEST(MmapRoundedRead, TruncatedItqWithoutMmapFlagThrows) {
    std::vector<uint8_t> full = itq_pretransform_bytes();
    std::vector<uint8_t> bytes = truncate_to(full, itq_mean_payload(full) + 1);
    TempFile file(bytes);
    EXPECT_THROW(
            {
                std::unique_ptr<faiss::Index> idx(
                        faiss::read_index(file.path().c_str(), 0));
                (void)idx;
            },
            faiss::FaissException);
}

class ItqMeanTail : public ::testing::TestWithParam<int> {};

TEST_P(ItqMeanTail, MmapReadThrowsInsteadOfOverflowing) {
    std::vector<uint8_t> full = itq_pretransform_bytes();
    size_t payload = itq_mean_payload(full);
    ASSERT_GE(full.size(), payload + sizeof(float));
    std::vector<uint8_t> bytes =
            truncate_to(full, payload + static_cast<size_t>(GetParam()));
    expect_mmap_index_rejects(bytes);
}

INSTANTIATE_TEST_SUITE_P(
        MmapRoundedRead,
        ItqMeanTail,
        ::testing::Values(1, 2, 3));

TEST(MmapRoundedRead, ItqMeanTailFromFileStar) {
    std::vector<uint8_t> full = itq_pretransform_bytes();
    std::vector<uint8_t> bytes = truncate_to(full, itq_mean_payload(full) + 2);
    expect_mmap_index_file_rejects(bytes);
}

class PcaEpsilonTail : public ::testing::TestWithParam<int> {};

TEST_P(PcaEpsilonTail, MmapReadThrowsInsteadOfOverflowing) {
    std::vector<uint8_t> full = pca_pretransform_bytes();
    size_t at = pca_epsilon_offset(full);
    ASSERT_GE(full.size(), at + sizeof(float));
    std::vector<uint8_t> bytes =
            truncate_to(full, at + static_cast<size_t>(GetParam()));
    expect_mmap_index_rejects(bytes);
}

INSTANTIATE_TEST_SUITE_P(
        MmapRoundedRead,
        PcaEpsilonTail,
        ::testing::Values(1, 2, 3));

class PcaMatTail : public ::testing::TestWithParam<int> {};

TEST_P(PcaMatTail, MmapReadThrowsInsteadOfOverflowing) {
    std::vector<uint8_t> full = pca_pretransform_bytes();
    size_t payload = pca_mat_payload(full);
    ASSERT_GE(full.size(), payload + sizeof(float));
    std::vector<uint8_t> bytes =
            truncate_to(full, payload + static_cast<size_t>(GetParam()));
    expect_mmap_index_rejects(bytes);
}

INSTANTIATE_TEST_SUITE_P(
        MmapRoundedRead,
        PcaMatTail,
        ::testing::Values(1, 2, 3));

class ViqmSeedTail : public ::testing::TestWithParam<int> {};

TEST_P(ViqmSeedTail, MmapReadThrowsInsteadOfOverflowing) {
    std::vector<uint8_t> full = itq_matrix_pretransform_bytes();
    size_t at = viqm_seed_offset(full);
    ASSERT_GE(full.size(), at + sizeof(int));
    std::vector<uint8_t> bytes =
            truncate_to(full, at + static_cast<size_t>(GetParam()));
    expect_mmap_index_rejects(bytes);
}

INSTANTIATE_TEST_SUITE_P(
        MmapRoundedRead,
        ViqmSeedTail,
        ::testing::Values(1, 2, 3));

class FlatHeaderTail : public ::testing::TestWithParam<int> {};

TEST_P(FlatHeaderTail, MmapReadThrowsInsteadOfOverflowing) {
    std::vector<uint8_t> full = flat_index_bytes();
    size_t at = flat_second_dummy_offset(full);
    faiss::idx_t dummy = faiss::idx_t{1} << 20;
    ASSERT_EQ(std::memcmp(full.data() + at, &dummy, sizeof(dummy)), 0);
    ASSERT_EQ(
            std::memcmp(
                    full.data() + at - sizeof(dummy), &dummy, sizeof(dummy)),
            0);
    std::vector<uint8_t> bytes =
            truncate_to(full, at + static_cast<size_t>(GetParam()));
    expect_mmap_index_rejects(bytes);
}

INSTANTIATE_TEST_SUITE_P(
        MmapRoundedRead,
        FlatHeaderTail,
        ::testing::Values(1, 2, 3, 4, 5, 6, 7));

class BinaryHeaderTail : public ::testing::TestWithParam<int> {};

TEST_P(BinaryHeaderTail, MmapReadThrowsInsteadOfOverflowing) {
    std::vector<uint8_t> full = binary_flat_bytes();
    size_t at = binary_ntotal_offset(full);
    faiss::idx_t ntotal = 0;
    ASSERT_EQ(std::memcmp(full.data() + at, &ntotal, sizeof(ntotal)), 0);
    std::vector<uint8_t> bytes =
            truncate_to(full, at + static_cast<size_t>(GetParam()));
    expect_mmap_binary_rejects(bytes);
}

INSTANTIATE_TEST_SUITE_P(
        MmapRoundedRead,
        BinaryHeaderTail,
        ::testing::Values(1, 2, 3, 4, 5, 6, 7));

class PartialElement : public ::testing::TestWithParam<size_t> {};

TEST_P(PartialElement, OperatorDoesNotRoundOrOverflowNextByte) {
    size_t elem = GetParam();
    std::vector<uint8_t> raw(3, 0);
    raw[0] = 0x11;
    raw[1] = 0x22;
    raw[2] = 0x33;
    TempFile file(raw);
    ProcResult result = run_child([&] {
        child_partial_then_byte(
                file.path().c_str(), elem, false, 1, 0, 0, 1, 0x11);
    });
    EXPECT_CHILD_CODE(result, 0);
}

INSTANTIATE_TEST_SUITE_P(
        MmapRoundedRead,
        PartialElement,
        ::testing::Values(
                sizeof(float),
                sizeof(int),
                sizeof(uint64_t)));

TEST(MmapRoundedRead, MmapPartialFloatDoesNotRoundOrOverflowNextByte) {
    std::vector<uint8_t> raw = {0x11, 0x22, 0x33};
    TempFile file(raw);
    ProcResult result = run_child([&] {
        child_partial_then_byte(
                file.path().c_str(),
                sizeof(float),
                true,
                1,
                0,
                0,
                1,
                0x11);
    });
    EXPECT_CHILD_CODE(result, 0);
}

TEST(MmapRoundedRead, TwoFloatsKeepOneCompleteElement) {
    std::vector<uint8_t> raw = {0x10, 0x21, 0x32, 0x43, 0x54, 0x65};
    TempFile file(raw);
    ProcResult result = run_child([&] {
        child_partial_then_byte(
                file.path().c_str(),
                sizeof(float),
                false,
                2,
                1,
                sizeof(float),
                sizeof(float) + 1,
                0x54);
    });
    EXPECT_CHILD_CODE(result, 0);
}

TEST(MmapRoundedRead, MmapTwoFloatsKeepOneCompleteElement) {
    std::vector<uint8_t> raw = {0x10, 0x21, 0x32, 0x43, 0x54, 0x65};
    TempFile file(raw);
    ProcResult result = run_child([&] {
        child_partial_then_byte(
                file.path().c_str(),
                sizeof(float),
                true,
                2,
                1,
                sizeof(float),
                sizeof(float) + 1,
                0x54);
    });
    EXPECT_CHILD_CODE(result, 0);
}

class CursorPastEnd : public ::testing::TestWithParam<size_t> {};

TEST_P(CursorPastEnd, ByteReadReturnsShortAndDoesNotWrite) {
    std::vector<uint8_t> raw(32, 0x7e);
    TempFile file(raw);
    size_t overshoot = GetParam();
    ProcResult result = run_child([&] {
        auto owner =
                std::make_shared<faiss::MmappedFileMappingOwner>(file.path());
        faiss::MappedFileIOReader reader(owner);
        reader.pos = owner->size() + overshoot;
        GuardedByte guard;
        if (guard.slot == nullptr) {
            _exit(6);
        }
        size_t n = reader(guard.slot, 1, 1);
        if (n != 0 || *guard.slot != 0x5a) {
            _exit(10);
        }
        _exit(0);
    });
    EXPECT_CHILD_CODE(result, 0);
}

INSTANTIATE_TEST_SUITE_P(
        MmapRoundedRead,
        CursorPastEnd,
        ::testing::Values(1u, 2u, 3u, 7u));

TEST(MmapRoundedRead, MmapPastEndReturnsShort) {
    std::vector<uint8_t> raw(32, 0x7e);
    TempFile file(raw);
    ProcResult result = run_child([&] {
        auto owner =
                std::make_shared<faiss::MmappedFileMappingOwner>(file.path());
        faiss::MappedFileIOReader reader(owner);
        // Element size 1: a cursor already past EOF must not wrap the
        // remaining length into a huge item count. A 4-byte element with a
        // small overshoot can divide that wrapped length down to zero and
        // hide the bug.
        reader.pos = owner->size() + 3;
        void* view = nullptr;
        size_t n = reader.mmap(&view, 1, 1);
        if (n != 0 || reader.pos != owner->size() + 3) {
            _exit(10);
        }
        _exit(0);
    });
    EXPECT_CHILD_CODE(result, 0);
}
