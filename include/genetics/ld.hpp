// =============================================================================
// ld.hpp — Linkage Disequilibrium (LD) statistics.
//
// Measures non-random association of alleles at different sites.
//
// Implemented:
//   - r² (squared correlation of allele indicators)
//   - D  (coefficient of linkage disequilibrium)
//   - D' (normalised D)
//   - LD decay curve (r² as a function of distance between sites)
//   - Pairwise LD matrix for a window of sites
//
// All functions work with any storage backend via the ElementAccessStorage
// concept, with specialised fast paths for PointerAccessStorage (Dense).
// =============================================================================
#pragma once

#include "concepts.hpp"
#include "types.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

namespace gensim {

// ── Pairwise LD result ──────────────────────────────────────────────────────
struct PairwiseLD {
    SiteIndex site_i = 0;
    SiteIndex site_j = 0;
    double    D      = 0.0;    // raw D
    double    D_prime = 0.0;   // normalised D
    double    r_sq   = 0.0;    // r²
};

// ── Core LD computation (two sites) ─────────────────────────────────────────

namespace detail {

/// Compute allele frequencies and haplotype frequencies for two biallelic
/// sites (reference vs non-reference).  Returns {p_A, p_B, p_AB} where
/// A = non-ref at site_i, B = non-ref at site_j.
template <typename Storage>
requires ElementAccessStorage<Storage>
inline auto ld_frequencies(const Storage& storage,
                           SiteIndex site_i, SiteIndex site_j)
{
    const std::size_t H = storage.num_haplotypes();
    std::size_t count_A  = 0;   // non-ref at site i
    std::size_t count_B  = 0;   // non-ref at site j
    std::size_t count_AB = 0;   // non-ref at both

    for (std::size_t h = 0; h < H; ++h) {
        bool a = (storage.get(h, site_i) != kRefAllele);
        bool b = (storage.get(h, site_j) != kRefAllele);
        count_A  += a;
        count_B  += b;
        count_AB += (a && b);
    }

    double n  = static_cast<double>(H);
    double pA = static_cast<double>(count_A)  / n;
    double pB = static_cast<double>(count_B)  / n;
    double pAB = static_cast<double>(count_AB) / n;

    struct Result { double pA, pB, pAB; };
    return Result{pA, pB, pAB};
}

/// PointerAccessStorage fast path.
template <typename Storage>
requires PointerAccessStorage<Storage>
inline auto ld_frequencies_ptr(const Storage& storage,
                               SiteIndex site_i, SiteIndex site_j)
{
    const std::size_t H = storage.num_haplotypes();
    std::size_t count_A  = 0;
    std::size_t count_B  = 0;
    std::size_t count_AB = 0;

    for (std::size_t h = 0; h < H; ++h) {
        const AlleleID* row = storage.hap_ptr(h);
        bool a = (row[site_i] != kRefAllele);
        bool b = (row[site_j] != kRefAllele);
        count_A  += a;
        count_B  += b;
        count_AB += (a && b);
    }

    double n  = static_cast<double>(H);
    struct Result { double pA, pB, pAB; };
    return Result{
        static_cast<double>(count_A)  / n,
        static_cast<double>(count_B)  / n,
        static_cast<double>(count_AB) / n
    };
}

}  // namespace detail

// ── Compute LD between two sites ────────────────────────────────────────────

/// Compute pairwise LD statistics (D, D', r²) between two sites.
/// Treats alleles as biallelic: reference vs any non-reference.
template <typename Storage>
requires ElementAccessStorage<Storage>
[[nodiscard]] PairwiseLD compute_ld(const Storage& storage,
                                     SiteIndex site_i, SiteIndex site_j)
{
    auto [pA, pB, pAB] = [&]() {
        if constexpr (PointerAccessStorage<Storage>)
            return detail::ld_frequencies_ptr(storage, site_i, site_j);
        else
            return detail::ld_frequencies(storage, site_i, site_j);
    }();

    PairwiseLD result;
    result.site_i = site_i;
    result.site_j = site_j;

    // D = p(AB) - p(A)*p(B)
    result.D = pAB - pA * pB;

    // D' = D / D_max
    if (std::abs(result.D) < 1e-15) {
        result.D_prime = 0.0;
        result.r_sq    = 0.0;
    } else {
        double D_max;
        if (result.D > 0.0)
            D_max = std::min(pA * (1.0 - pB), (1.0 - pA) * pB);
        else
            D_max = std::min(pA * pB, (1.0 - pA) * (1.0 - pB));

        result.D_prime = (D_max > 1e-15) ? result.D / D_max : 0.0;

        // r² = D² / (pA * (1-pA) * pB * (1-pB))
        double denom = pA * (1.0 - pA) * pB * (1.0 - pB);
        result.r_sq = (denom > 1e-15)
            ? (result.D * result.D) / denom
            : 0.0;
    }

    return result;
}

// ── LD decay curve ──────────────────────────────────────────────────────────
/// Compute mean r² as a function of inter-site distance.
/// Returns a vector of (distance, mean_r²) pairs, binned by `bin_size`.
/// Only sites in [start, end) are considered.  If end == 0, uses num_sites.
///
/// `max_pairs_per_bin`: cap on pairs sampled per distance bin (for speed).

struct LDDecayPoint {
    std::size_t distance  = 0;
    double      mean_r_sq = 0.0;
    std::size_t n_pairs   = 0;
};

template <typename Storage>
requires ElementAccessStorage<Storage>
[[nodiscard]] std::vector<LDDecayPoint>
compute_ld_decay(const Storage& storage,
                  std::size_t bin_size = 10,
                  SiteIndex start = 0,
                  SiteIndex end = 0,
                  std::size_t max_pairs_per_bin = 5000)
{
    const SiteIndex L = static_cast<SiteIndex>(storage.num_sites());
    if (end == 0 || end > L) end = L;
    assert(start < end);

    const std::size_t range = end - start;
    const std::size_t n_bins = (range + bin_size - 1) / bin_size;

    // Accumulate r² per distance bin.
    std::vector<double> sum_r2(n_bins, 0.0);
    std::vector<std::size_t> counts(n_bins, 0);

    // Iterate over site pairs.  For large ranges, subsample.
    for (SiteIndex i = start; i < end; ++i) {
        for (SiteIndex j = i + 1; j < end; ++j) {
            std::size_t dist = j - i;
            std::size_t bin = (dist - 1) / bin_size;
            if (bin >= n_bins) break;
            if (counts[bin] >= max_pairs_per_bin) continue;

            auto ld = compute_ld(storage, i, j);
            sum_r2[bin] += ld.r_sq;
            ++counts[bin];
        }
    }

    std::vector<LDDecayPoint> result;
    result.reserve(n_bins);
    for (std::size_t b = 0; b < n_bins; ++b) {
        if (counts[b] == 0) continue;
        result.push_back(LDDecayPoint{
            (b + 1) * bin_size,   // midpoint distance of this bin
            sum_r2[b] / static_cast<double>(counts[b]),
            counts[b]
        });
    }
    return result;
}

// ── Pairwise LD matrix ──────────────────────────────────────────────────────
/// Compute all pairwise r² values for sites in [start, end).
/// Returns a flat upper-triangle matrix (row-major).

template <typename Storage>
requires ElementAccessStorage<Storage>
[[nodiscard]] std::vector<PairwiseLD>
compute_ld_matrix(const Storage& storage,
                   SiteIndex start, SiteIndex end)
{
    assert(end <= static_cast<SiteIndex>(storage.num_sites()));
    assert(start < end);

    std::vector<PairwiseLD> result;
    const std::size_t range = end - start;
    result.reserve(range * (range - 1) / 2);

    for (SiteIndex i = start; i < end; ++i) {
        for (SiteIndex j = i + 1; j < end; ++j) {
            result.push_back(compute_ld(storage, i, j));
        }
    }
    return result;
}

}  // namespace gensim
