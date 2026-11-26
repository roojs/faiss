# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Out-of-range bucket ids must be rejected before the histogram update.

These tests fail while the vulnerability is present. The single-thread
bucket sort throws FaissException ("vals[i] < nbucket" / "vals[i] < vmax")
before it indexes the histogram. The threaded paths index the histogram
with the raw id. A value of nbucket + 1 is one past a histogram of length
nbucket + 1, which AddressSanitizer reports as a heap-buffer-overflow in
bucket_sort_inplace_parallel. The same missing check is in
bucket_sort_parallel, in both int32 and int64 in-place instantiations, and
in the single-thread path for negative ids (a signed `< nbucket` test treats
them as in range).

Big-batch IVF search always takes the threaded in-place path with 16
threads. It uses coarse-quantizer labels as bucket ids and nbucket =
nlist + 1, after adding one to every label. read_index stores nlist and the
coarse quantizer separately and does not require quantizer.ntotal == nlist,
so a loaded index can hand search a label that becomes nbucket + 1.

Run (AddressSanitizer build required so an overflow aborts instead of
looking like the later lims[nbucket] == nval check):

  cmake -S . -B build/asan -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_MKL=OFF \
    -DFAISS_ENABLE_PYTHON=ON -DFAISS_OPT_LEVEL=generic \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer" \
    -DCMAKE_MODULE_LINKER_FLAGS="-fsanitize=address" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address" \
    -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address"
  cmake --build build/asan -j --target swigfaiss
  # Link the interpreter with -fsanitize=address (LD_PRELOAD is not enough)
  # and put build/asan/faiss/python on PYTHONPATH as the faiss package.
  ASAN_OPTIONS=detect_leaks=0:halt_on_error=1:abort_on_error=1 \
    python3 tests/test_threaded_bucket_sort_bounds.py
"""

import unittest

import numpy as np

import faiss


def _assert_bounds_failure(testcase, fn, needle):
    """The call must raise FaissException for the id itself.

    A throw from the histogram consistency check (lims[nbucket] == nval or
    lims[vmax] == nval) happens only after the id has already been used as
    an index, so it does not count. Messages that start with "C++ exception "
    are not FaissException.
    """
    with testcase.assertRaises(RuntimeError) as cm:
        fn()
    msg = str(cm.exception)
    testcase.assertFalse(msg.startswith("C++ exception "), msg)
    testcase.assertIn(needle, msg)
    testcase.assertNotIn("lims[", msg)


class TestThreadedBucketSortBounds(unittest.TestCase):
    def test_serial_inplace_rejects_high_id(self):
        # Control: the single-thread path already rejects ids >= nbucket.
        tab = np.array([[0, 2, 0]], dtype="int32")
        _assert_bounds_failure(
            self,
            lambda: faiss.matrix_bucket_sort_inplace(tab.copy(), nbucket=2, nt=0),
            "vals[i] < nbucket",
        )

    def test_threaded_inplace_int32_rejects_id_past_histogram(self):
        # nbucket == 2 => histogram length 3. Id 3 is one past that buffer.
        # This is the id big-batch search produces from coarse label 2.
        tab = np.array([[0, 3, 0]], dtype="int32")
        _assert_bounds_failure(
            self,
            lambda: faiss.matrix_bucket_sort_inplace(
                tab.copy(), nbucket=2, nt=16
            ),
            "vals[i] < nbucket",
        )

    def test_threaded_inplace_int32_rejects_id_equal_nbucket(self):
        tab = np.array([[0, 2, 0]], dtype="int32")
        _assert_bounds_failure(
            self,
            lambda: faiss.matrix_bucket_sort_inplace(tab.copy(), nbucket=2, nt=1),
            "vals[i] < nbucket",
        )

    def test_threaded_inplace_int64_rejects_id_past_histogram(self):
        tab = np.array([[0, 3, 0]], dtype="int64")
        _assert_bounds_failure(
            self,
            lambda: faiss.matrix_bucket_sort_inplace(
                tab.copy(), nbucket=2, nt=16
            ),
            "vals[i] < nbucket",
        )

    def test_inplace_rejects_negative_ids(self):
        for dtype in ("int32", "int64"):
            for nt in (0, 16):
                with self.subTest(dtype=dtype, nt=nt):
                    tab = np.array([[0, -1, 0]], dtype=dtype)
                    _assert_bounds_failure(
                        self,
                        lambda: faiss.matrix_bucket_sort_inplace(
                            tab.copy(), nbucket=2, nt=nt
                        ),
                        "vals[i] < nbucket",
                    )

    def test_serial_bucket_sort_rejects_high_id(self):
        tab = np.array([0, 3, 0], dtype="int64")
        _assert_bounds_failure(
            self,
            lambda: faiss.bucket_sort(tab, nbucket=2, nt=0),
            "vals[i] < vmax",
        )

    def test_threaded_bucket_sort_rejects_id_past_histogram(self):
        tab = np.array([0, 3, 0], dtype="int64")
        _assert_bounds_failure(
            self,
            lambda: faiss.bucket_sort(tab.copy(), nbucket=2, nt=16),
            "vals[i] < vmax",
        )

    def test_threaded_bucket_sort_rejects_id_equal_vmax(self):
        tab = np.array([0, 2, 0], dtype="int64")
        _assert_bounds_failure(
            self,
            lambda: faiss.bucket_sort(tab.copy(), nbucket=2, nt=1),
            "vals[i] < vmax",
        )

    def test_in_range_threaded_sort_still_succeeds(self):
        tab = np.array([[0, 1, 0], [1, 0, 1]], dtype="int32")
        lims = faiss.matrix_bucket_sort_inplace(tab.copy(), nbucket=2, nt=4)
        self.assertEqual(int(lims[-1]), tab.size)


class TestBigBatchSearchBucketIds(unittest.TestCase):
    def _loaded_mismatched_ivf(self):
        """IVFFlat nlist=1 whose coarse quantizer has three centroids.

        The first two centroids are extreme; the last is the origin. An
        ordinary query is assigned to centroid 2. Serialization round-trips
        through write_index / read_index, which store nlist and the quantizer
        separately.
        """
        d = 4
        quantizer = faiss.IndexFlatL2(d)
        quantizer.add(
            np.array(
                [[1e6] * d, [-1e6] * d, [0.0] * d],
                dtype="float32",
            )
        )
        index = faiss.IndexIVFFlat(quantizer, d, 1)
        index.is_trained = True
        index.nprobe = 1
        loaded = faiss.deserialize_index(faiss.serialize_index(index))
        self.assertEqual(loaded.nlist, 1)
        self.assertEqual(int(loaded.quantizer.ntotal), 3)
        self.assertTrue(loaded.is_trained)
        return loaded

    def test_reader_keeps_quantizer_larger_than_nlist(self):
        loaded = self._loaded_mismatched_ivf()
        xq = np.zeros((3, loaded.d), dtype="float32")
        _, labels = loaded.quantizer.search(xq, 1)
        self.assertTrue(np.all(labels == 2))

    def test_big_batch_search_rejects_out_of_range_assignment(self):
        # Coarse label 2, plus the +1 shift in BigBatchSearcher.reorder_assign,
        # is bucket id 3 with nbucket = nlist + 1 == 2.
        from faiss.contrib.big_batch_search import big_batch_search

        loaded = self._loaded_mismatched_ivf()
        xq = np.zeros((3, loaded.d), dtype="float32")
        _assert_bounds_failure(
            self,
            lambda: big_batch_search(loaded, xq, 1),
            "vals[i] < nbucket",
        )


if __name__ == "__main__":
    unittest.main()
