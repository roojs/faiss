/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexLattice.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/simd_dispatch.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/hamming.h> // for the bitstring routines

namespace faiss {

namespace {

void check_scale_nbit(int scale_nbit) {
    FAISS_THROW_IF_NOT_FMT(
            scale_nbit >= 0 && scale_nbit < 63,
            "IndexLattice: scale_nbit=%d out of range [0, 63), "
            "code_size would be invalid",
            scale_nbit);
}

// Validates the constructor arguments before the sphere codec is built.
size_t lattice_dsq(idx_t d, int nsq, int scale_nbit) {
    FAISS_THROW_IF_NOT_FMT(
            nsq > 0 && d > 0 && d % nsq == 0,
            "IndexLattice: invalid d=%lld, nsq=%d",
            static_cast<long long>(d),
            nsq);
    check_scale_nbit(scale_nbit);
    return d / nsq;
}

/// Total number of bits sa_encode writes / sa_decode reads per vector.
uint64_t lattice_code_nbit(int lattice_nbit, int scale_nbit, int nsq) {
    check_scale_nbit(scale_nbit);
    FAISS_THROW_IF_NOT_FMT(
            lattice_nbit >= 0 && lattice_nbit < 64,
            "IndexLattice: lattice_nbit=%d out of range [0, 64)",
            lattice_nbit);
    FAISS_THROW_IF_NOT_FMT(nsq > 0, "IndexLattice: invalid nsq=%d", nsq);
    return (uint64_t(lattice_nbit) + uint64_t(scale_nbit)) * uint64_t(nsq);
}

// The fields are public and may come from a deserialized index, so they
// are re-checked against code_size before any code is read or written.
void check_codec_state(const IndexLattice& index) {
    uint64_t nbit =
            lattice_code_nbit(index.lattice_nbit, index.scale_nbit, index.nsq);
    FAISS_THROW_IF_NOT_FMT(
            nbit <= uint64_t(index.code_size) * 8,
            "IndexLattice: %llu code bits do not fit in code_size=%zd",
            static_cast<unsigned long long>(nbit),
            index.code_size);
    FAISS_THROW_IF_NOT_FMT(
            index.dsq * index.nsq == size_t(index.d),
            "IndexLattice: dsq=%zd * nsq=%d != d=%d",
            index.dsq,
            index.nsq,
            index.d);
    FAISS_THROW_IF_NOT_FMT(
            index.trained.size() >= 2 * size_t(index.nsq),
            "IndexLattice: trained has %zd entries, need %zd",
            index.trained.size(),
            2 * size_t(index.nsq));
}

} // namespace

IndexLattice::IndexLattice(idx_t d_in, int nsq_in, int scale_nbit_in, int r2)
        : IndexFlatCodes(0, d_in, METRIC_L2),
          nsq(nsq_in),
          dsq(lattice_dsq(d_in, nsq_in, scale_nbit_in)),
          zn_sphere_codec(static_cast<int>(dsq), r2),
          scale_nbit(scale_nbit_in) {
    lattice_nbit = 0;
    while (!(((uint64_t)1 << lattice_nbit) >= zn_sphere_codec.nv)) {
        lattice_nbit++;
        FAISS_THROW_IF_NOT_FMT(
                lattice_nbit < 64,
                "IndexLattice: nv=%zu too large, lattice code would exceed "
                "63 bits (likely corrupt r2/dsq)",
                (size_t)zn_sphere_codec.nv);
    }

    uint64_t total_nbit = lattice_code_nbit(lattice_nbit, scale_nbit, nsq);

    code_size = static_cast<size_t>((total_nbit + 7) / 8);

    is_trained = false;
}

void IndexLattice::train(idx_t n, const float* x) {
    // compute ranges per sub-block
    trained.resize(nsq * 2);
    float* mins = trained.data();
    float* maxs = trained.data() + nsq;
    for (int sq = 0; sq < nsq; sq++) {
        mins[sq] = HUGE_VAL;
        maxs[sq] = -1;
    }

    with_simd_level([&]<SIMDLevel SL>() {
        for (idx_t i = 0; i < n; i++) {
            for (int sq = 0; sq < nsq; sq++) {
                float norm2 = fvec_norm_L2sqr<SL>(x + i * d + sq * dsq, dsq);
                if (norm2 > maxs[sq]) {
                    maxs[sq] = norm2;
                }
                if (norm2 < mins[sq]) {
                    mins[sq] = norm2;
                }
            }
        }
    });

    for (int sq = 0; sq < nsq; sq++) {
        mins[sq] = sqrtf(mins[sq]);
        maxs[sq] = sqrtf(maxs[sq]);
    }

    is_trained = true;
}

/* The standalone codec interface */
size_t IndexLattice::sa_code_size() const {
    return code_size;
}

void IndexLattice::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    check_codec_state(*this);
    const float* mins = trained.data();
    const float* maxs = mins + nsq;
    int64_t sc = int64_t(1) << scale_nbit;

    with_simd_level([&]<SIMDLevel SL>() {
#pragma omp parallel for
        for (idx_t i = 0; i < n; i++) {
            BitstringWriter wr(bytes + i * code_size, code_size);
            const float* xi = x + i * d;
            for (int j = 0; j < nsq; j++) {
                float nj = (sqrtf(fvec_norm_L2sqr<SL>(xi, dsq)) - mins[j]) *
                        sc / (maxs[j] - mins[j]);
                if (nj < 0) {
                    nj = 0;
                }
                if (nj >= sc) {
                    nj = sc - 1;
                }
                wr.write((int64_t)nj, scale_nbit);
                wr.write(zn_sphere_codec.encode(xi), lattice_nbit);
                xi += dsq;
            }
        }
    });
}

void IndexLattice::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    check_codec_state(*this);
    const float* mins = trained.data();
    const float* maxs = mins + nsq;
    float sc = int64_t(1) << scale_nbit;
    float r = sqrtf(zn_sphere_codec.r2);

#pragma omp parallel for
    for (idx_t i = 0; i < n; i++) {
        BitstringReader rd(bytes + i * code_size, code_size);
        float* xi = x + i * d;
        for (int j = 0; j < nsq; j++) {
            float norm =
                    (rd.read(scale_nbit) + 0.5) * (maxs[j] - mins[j]) / sc +
                    mins[j];
            norm /= r;
            zn_sphere_codec.decode(rd.read(lattice_nbit), xi);
            for (size_t l = 0; l < dsq; l++) {
                xi[l] *= norm;
            }
            xi += dsq;
        }
    }
}

} // namespace faiss
