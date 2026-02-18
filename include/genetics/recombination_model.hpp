// =============================================================================
// recombination_model.hpp — Pluggable recombination policies.
//
// Models provided:
//
//   1. SingleCrossover  – one uniformly random crossover point.
//   2. MultiBreakpoint  – Poisson-distributed number of crossovers, uniform.
//   3. MappedCrossover  – variable recombination rate along the genome
//                         via a RecombinationMap (piecewise-constant rates).
//
// All are backend-aware: memcpy block copies for Dense, per-variant for Sparse.
//
// Extension point: any callable with signature
//   void operator()(const Storage& parents, Storage& offspring,
//                   size_t parent_hap0, size_t parent_hap1,
//                   size_t offspring_hap, RNG& rng) const;
// =============================================================================
#pragma once

#include "dense_haplotypes.hpp"
#include "sparse_variants.hpp"
#include "types.hpp"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <random>
#include <vector>

namespace gensim {

// =============================================================================
// RecombinationMap — piecewise-constant recombination rate along the genome.
//
// Defined by a vector of segment endpoints and per-segment rates.
// Segment i covers sites [endpoints[i], endpoints[i+1]) with rate rates[i].
// The rate is the per-site probability of a crossover occurring between two
// adjacent sites within that segment.
//
// The total expected number of crossovers = sum(rate_i * length_i).
// =============================================================================

struct RecombinationMap {
    /// Segment boundary positions.  Must be sorted.  First should be 0,
    /// last should be L (total number of sites).
    std::vector<SiteIndex> endpoints;

    /// Per-segment recombination rate.  Length = endpoints.size() - 1.
    std::vector<double>    rates;

    /// Build a uniform map (constant rate across L sites).
    static RecombinationMap uniform(std::size_t L, double rate) {
        RecombinationMap m;
        m.endpoints = {0, L};
        m.rates     = {rate};
        return m;
    }

    /// Total expected crossovers for the entire map.
    [[nodiscard]] double total_map_length() const noexcept {
        double total = 0.0;
        for (std::size_t i = 0; i < rates.size(); ++i) {
            double seg_len = static_cast<double>(endpoints[i+1] - endpoints[i]);
            total += rates[i] * seg_len;
        }
        return total;
    }

    /// Build a cumulative weight vector for weighted breakpoint placement.
    /// Each element is the cumulative rate*length up to that inter-site gap.
    [[nodiscard]] std::vector<double> cumulative_weights() const {
        // We need one weight per inter-site gap: gap j is between site j and j+1.
        // Total number of gaps = endpoints.back() - 1.
        std::size_t L = endpoints.back();
        std::vector<double> cw;
        cw.reserve(L - 1);
        double cum = 0.0;
        std::size_t seg = 0;
        for (std::size_t gap = 0; gap < L - 1; ++gap) {
            // Advance segment index if needed.
            while (seg + 1 < rates.size() && gap + 1 >= endpoints[seg + 1]) ++seg;
            cum += rates[seg];
            cw.push_back(cum);
        }
        return cw;
    }
};

// =============================================================================
// Helper: draw breakpoints from a RecombinationMap
// =============================================================================

/// Draw crossover breakpoints using a recombination map with variable rates.
/// Returns sorted, unique breakpoints in [1, L-1].
inline std::vector<SiteIndex>
draw_mapped_breakpoints(const RecombinationMap& rmap,
                         std::mt19937_64& rng)
{
    double total = rmap.total_map_length();
    if (total <= 0.0) return {};

    // Number of crossovers is Poisson with mean = total map length.
    std::poisson_distribution<std::size_t> ndist(total);
    std::size_t n = ndist(rng);
    if (n == 0) return {};

    // Pre-compute cumulative weights for weighted sampling.
    auto cw = rmap.cumulative_weights();
    std::uniform_real_distribution<double> u(0.0, cw.back());

    std::vector<SiteIndex> bps;
    bps.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        double r = u(rng);
        // Binary search: find the gap index whose cumulative weight >= r.
        auto it = std::lower_bound(cw.begin(), cw.end(), r);
        SiteIndex gap = static_cast<SiteIndex>(std::distance(cw.begin(), it));
        // Breakpoint position = gap + 1 (site index after the gap).
        SiteIndex bp = gap + 1;
        if (bp >= rmap.endpoints.back()) bp = rmap.endpoints.back() - 1;
        if (bp >= 1) bps.push_back(bp);
    }

    std::sort(bps.begin(), bps.end());
    bps.erase(std::unique(bps.begin(), bps.end()), bps.end());
    return bps;
}

// ─────────────────────────────────────────────────────────────────────────────
// SingleCrossover
// ─────────────────────────────────────────────────────────────────────────────
struct SingleCrossover {

    // ── Dense ───────────────────────────────────────────────────────────────
    void operator()(const DenseHaplotypes& parents, DenseHaplotypes& offspring,
                    std::size_t phap0, std::size_t phap1,
                    std::size_t ohap,
                    std::mt19937_64& rng) const
    {
        const std::size_t L = parents.num_sites();
        // Crossover point in [1, L-1] so both segments are non-empty.
        std::uniform_int_distribution<std::size_t> xdist(1, L - 1);
        std::size_t xp = xdist(rng);

        // Block copy: [0, xp) from phap0, [xp, L) from phap1.
        // We read from the parents' front buffer and write to offspring's
        // back buffer.  Because DenseHaplotypes has public copy_segment that
        // reads front→back on a *single* object, we work via raw pointers
        // here (parents and offspring may be the same object with double-
        // buffering, which is the normal case).
        const AlleleID* src0 = parents.hap_ptr(phap0);
        const AlleleID* src1 = parents.hap_ptr(phap1);
        AlleleID*       dst  = offspring.offspring_hap_ptr(ohap);

        std::memcpy(dst, src0, xp * sizeof(AlleleID));
        std::memcpy(dst + xp, src1 + xp, (L - xp) * sizeof(AlleleID));
    }

    // ── Sparse ──────────────────────────────────────────────────────────────
    void operator()(const SparseVariants& /*parents*/, SparseVariants& offspring,
                    std::size_t phap0, std::size_t phap1,
                    std::size_t ohap,
                    std::mt19937_64& rng) const
    {
        // parents and offspring are the same object (double-buffered):
        // copy_segment reads from the front (parent) buffer and writes
        // to the back (offspring) buffer.
        const std::size_t L = offspring.num_sites();
        std::uniform_int_distribution<SiteIndex> xdist(1, L - 1);
        SiteIndex xp = xdist(rng);

        offspring.copy_segment(phap0, ohap, 0, xp);
        offspring.copy_segment(phap1, ohap, xp, L);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// MultiBreakpoint
// ─────────────────────────────────────────────────────────────────────────────
struct MultiBreakpoint {
    double expected_breakpoints = 1.0;   // mean number of crossovers

    // ── Dense ───────────────────────────────────────────────────────────────
    void operator()(const DenseHaplotypes& parents, DenseHaplotypes& offspring,
                    std::size_t phap0, std::size_t phap1,
                    std::size_t ohap,
                    std::mt19937_64& rng) const
    {
        const std::size_t L = parents.num_sites();

        // Draw breakpoints.
        auto bps = draw_breakpoints(L, rng);

        const AlleleID* src[2] = { parents.hap_ptr(phap0),
                                   parents.hap_ptr(phap1) };
        AlleleID* dst = offspring.offspring_hap_ptr(ohap);

        std::size_t cur = 0;   // which parental haplotype we're copying from
        std::size_t prev = 0;
        for (auto bp : bps) {
            std::memcpy(dst + prev, src[cur] + prev, (bp - prev) * sizeof(AlleleID));
            prev = bp;
            cur ^= 1;
        }
        // Final segment.
        std::memcpy(dst + prev, src[cur] + prev, (L - prev) * sizeof(AlleleID));
    }

    // ── Sparse ──────────────────────────────────────────────────────────────
    void operator()(const SparseVariants& /*parents*/, SparseVariants& offspring,
                    std::size_t phap0, std::size_t phap1,
                    std::size_t ohap,
                    std::mt19937_64& rng) const
    {
        // parents and offspring are the same double-buffered object.
        const std::size_t L = offspring.num_sites();
        auto bps = draw_breakpoints(L, rng);

        std::size_t haps[2] = { phap0, phap1 };
        std::size_t cur = 0;
        SiteIndex prev = 0;
        for (auto bp : bps) {
            offspring.copy_segment(haps[cur], ohap, prev, bp);
            prev = bp;
            cur ^= 1;
        }
        offspring.copy_segment(haps[cur], ohap, prev, L);
    }

private:
    // Draw sorted breakpoints in [1, L-1].
    [[nodiscard]] std::vector<SiteIndex>
    draw_breakpoints(std::size_t L, std::mt19937_64& rng) const {
        std::poisson_distribution<std::size_t> ndist(expected_breakpoints);
        std::uniform_int_distribution<SiteIndex> pos_dist(1, L - 1);

        std::size_t n = ndist(rng);
        std::vector<SiteIndex> bps;
        bps.reserve(n);
        for (std::size_t i = 0; i < n; ++i) bps.push_back(pos_dist(rng));

        std::sort(bps.begin(), bps.end());
        // Remove duplicates.
        bps.erase(std::unique(bps.begin(), bps.end()), bps.end());
        return bps;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// MappedCrossover — variable recombination rate along the genome.
// ─────────────────────────────────────────────────────────────────────────────
// Uses a RecombinationMap to place crossover breakpoints with position-
// dependent probabilities.  Models recombination hotspots / coldspots.
// ─────────────────────────────────────────────────────────────────────────────
struct MappedCrossover {
    RecombinationMap rmap;

    // ── Dense ───────────────────────────────────────────────────────────────
    void operator()(const DenseHaplotypes& parents, DenseHaplotypes& offspring,
                    std::size_t phap0, std::size_t phap1,
                    std::size_t ohap,
                    std::mt19937_64& rng) const
    {
        const std::size_t L = parents.num_sites();
        auto bps = draw_mapped_breakpoints(rmap, rng);

        const AlleleID* src[2] = { parents.hap_ptr(phap0),
                                   parents.hap_ptr(phap1) };
        AlleleID* dst = offspring.offspring_hap_ptr(ohap);

        std::size_t cur = 0;
        std::size_t prev = 0;
        for (auto bp : bps) {
            std::memcpy(dst + prev, src[cur] + prev, (bp - prev) * sizeof(AlleleID));
            prev = bp;
            cur ^= 1;
        }
        std::memcpy(dst + prev, src[cur] + prev, (L - prev) * sizeof(AlleleID));
    }

    // ── Sparse ──────────────────────────────────────────────────────────────
    void operator()(const SparseVariants& /*parents*/, SparseVariants& offspring,
                    std::size_t phap0, std::size_t phap1,
                    std::size_t ohap,
                    std::mt19937_64& rng) const
    {
        const std::size_t L = offspring.num_sites();
        auto bps = draw_mapped_breakpoints(rmap, rng);

        std::size_t haps[2] = { phap0, phap1 };
        std::size_t cur = 0;
        SiteIndex prev = 0;
        for (auto bp : bps) {
            offspring.copy_segment(haps[cur], ohap, prev, bp);
            prev = bp;
            cur ^= 1;
        }
        offspring.copy_segment(haps[cur], ohap, prev, L);
    }
};

}  // namespace gensim
