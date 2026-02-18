// =============================================================================
// mutation_rate_map.hpp — Per-site and per-region variable mutation rates.
//
// Analogous to RecombinationMap but for mutation.  In real genomes, mutation
// rate varies substantially along the chromosome:
//   - CpG sites mutate ~10x faster than average
//   - Coding regions may have different rates than non-coding
//   - Repair mechanisms create mutation rate heterogeneity
//
// Provided:
//   MutationRateMap       — piecewise-constant per-site mutation rates
//   MutationTypeMap       — region → mutation type (neutral, coding, regulatory)
//   MappedMutation        — mutation policy that uses a MutationRateMap
//   RegionalDFEMutation   — different DFEs in different genomic regions
// =============================================================================
#pragma once

#include "allele_table.hpp"
#include "dense_haplotypes.hpp"
#include "dfe.hpp"
#include "sparse_variants.hpp"
#include "types.hpp"

#include <algorithm>
#include <random>
#include <string>
#include <vector>

namespace gensim {

// ── MutationRateMap ─────────────────────────────────────────────────────────
/// Piecewise-constant mutation rate along the genome.
/// Exactly like RecombinationMap but for mutation.
struct MutationRateMap {
    /// Segment boundary positions (sorted).  First = 0, last = L.
    std::vector<SiteIndex> endpoints;

    /// Per-segment mutation rate.  Length = endpoints.size() - 1.
    std::vector<double> rates;

    /// Build a uniform map.
    static MutationRateMap uniform(std::size_t L, double mu) {
        MutationRateMap m;
        m.endpoints = {0, L};
        m.rates = {mu};
        return m;
    }

    /// Total expected mutations per haplotype = Σ(rate_i * length_i).
    [[nodiscard]] double total_rate() const noexcept {
        double total = 0.0;
        for (std::size_t i = 0; i < rates.size(); ++i) {
            total += rates[i] * static_cast<double>(endpoints[i + 1] - endpoints[i]);
        }
        return total;
    }

    /// Get the mutation rate at a specific site.
    [[nodiscard]] double rate_at(SiteIndex site) const noexcept {
        for (std::size_t i = 0; i < rates.size(); ++i) {
            if (site < endpoints[i + 1]) return rates[i];
        }
        return rates.back();
    }

    /// Build cumulative rate vector for weighted site sampling.
    [[nodiscard]] std::vector<double> cumulative_weights() const {
        std::size_t L = endpoints.back();
        std::vector<double> cw;
        cw.reserve(L);
        double cum = 0.0;
        std::size_t seg = 0;
        for (std::size_t site = 0; site < L; ++site) {
            while (seg + 1 < rates.size() && site >= endpoints[seg + 1]) ++seg;
            cum += rates[seg];
            cw.push_back(cum);
        }
        return cw;
    }
};

// ── MutationTypeTag ─────────────────────────────────────────────────────────
/// Tags for different functional categories of genomic regions.
enum class MutationTypeTag : std::uint8_t {
    Neutral    = 0,
    Coding     = 1,
    Regulatory = 2,
    Intergenic = 3,
    Custom     = 4
};

// ── MutationTypeRegion ──────────────────────────────────────────────────────
/// A genomic region with a specific mutation type and associated DFE.
struct MutationTypeRegion {
    SiteIndex       start;
    SiteIndex       end;        // exclusive
    MutationTypeTag tag = MutationTypeTag::Neutral;
    DFEVariant      dfe = FixedDFE{0.0};
    double          dominance_h = 0.5;
};

// ── MutationTypeMap ─────────────────────────────────────────────────────────
/// Maps genomic regions to mutation types with region-specific DFEs.
struct MutationTypeMap {
    std::vector<MutationTypeRegion> regions;

    /// Find the region containing a site.  Returns nullptr if no region covers it.
    [[nodiscard]] const MutationTypeRegion* region_at(SiteIndex site) const noexcept {
        for (auto& r : regions) {
            if (site >= r.start && site < r.end) return &r;
        }
        return nullptr;
    }

    /// Add a region.
    MutationTypeMap& add_region(SiteIndex start, SiteIndex end,
                                 MutationTypeTag tag,
                                 DFEVariant dfe,
                                 double h = 0.5)
    {
        regions.push_back({start, end, tag, std::move(dfe), h});
        return *this;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// MappedMutation — infinite-alleles mutation with variable per-site rates.
// ─────────────────────────────────────────────────────────────────────────────
// Uses a MutationRateMap to determine where mutations fall, instead of
// uniform sampling.  Each new mutation gets a unique AlleleID.
struct MappedMutation {
    MutationRateMap rate_map;

    void operator()(DenseHaplotypes& storage, AlleleTable& table,
                    Generation gen, std::mt19937_64& rng) const
    {
        const std::size_t H = storage.num_haplotypes();
        double total_rate = rate_map.total_rate();

        std::poisson_distribution<std::size_t> pois(total_rate);
        auto cw = rate_map.cumulative_weights();

        std::uniform_real_distribution<double> u(0.0, cw.back());

        for (std::size_t h = 0; h < H; ++h) {
            AlleleID* row = storage.offspring_hap_ptr(h);
            std::size_t n_mut = pois(rng);

            for (std::size_t m = 0; m < n_mut; ++m) {
                double r = u(rng);
                auto it = std::lower_bound(cw.begin(), cw.end(), r);
                SiteIndex site = static_cast<SiteIndex>(
                    std::distance(cw.begin(), it));
                if (site >= storage.num_sites()) site = storage.num_sites() - 1;

                AlleleID new_id = table.new_allele(
                    AlleleInfo{gen, 0.0, 0.5, {}});
                row[site] = new_id;
            }
        }
    }

    void operator()(SparseVariants& storage, AlleleTable& table,
                    Generation gen, std::mt19937_64& rng) const
    {
        const std::size_t H = storage.num_haplotypes();
        double total_rate = rate_map.total_rate();

        std::poisson_distribution<std::size_t> pois(total_rate);
        auto cw = rate_map.cumulative_weights();
        std::uniform_real_distribution<double> u(0.0, cw.back());

        for (std::size_t h = 0; h < H; ++h) {
            std::size_t n_mut = pois(rng);
            for (std::size_t m = 0; m < n_mut; ++m) {
                double r = u(rng);
                auto it = std::lower_bound(cw.begin(), cw.end(), r);
                SiteIndex site = static_cast<SiteIndex>(
                    std::distance(cw.begin(), it));
                if (site >= storage.num_sites()) site = storage.num_sites() - 1;

                AlleleID new_id = table.new_allele(
                    AlleleInfo{gen, 0.0, 0.5, {}});
                storage.offspring_set(h, site, new_id);
            }
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// RegionalDFEMutation — different DFEs for different genomic regions.
// ─────────────────────────────────────────────────────────────────────────────
// Mutation rate comes from a MutationRateMap, and the selection coefficient
// and dominance for each new mutation come from the MutationTypeMap region
// that the site falls in.  This enables realistic genome architectures where
// e.g. exons have purifying selection (gamma DFE) while introns are neutral.
struct RegionalDFEMutation {
    MutationRateMap rate_map;
    MutationTypeMap type_map;

    void operator()(DenseHaplotypes& storage, AlleleTable& table,
                    Generation gen, std::mt19937_64& rng) const
    {
        const std::size_t H = storage.num_haplotypes();
        double total_rate = rate_map.total_rate();

        std::poisson_distribution<std::size_t> pois(total_rate);
        auto cw = rate_map.cumulative_weights();
        std::uniform_real_distribution<double> u(0.0, cw.back());

        for (std::size_t h = 0; h < H; ++h) {
            AlleleID* row = storage.offspring_hap_ptr(h);
            std::size_t n_mut = pois(rng);

            for (std::size_t m = 0; m < n_mut; ++m) {
                double r = u(rng);
                auto it = std::lower_bound(cw.begin(), cw.end(), r);
                SiteIndex site = static_cast<SiteIndex>(
                    std::distance(cw.begin(), it));
                if (site >= storage.num_sites()) site = storage.num_sites() - 1;

                // Look up the region for this site.
                double s_val = 0.0;
                double h_val = 0.5;
                auto* region = type_map.region_at(site);
                if (region) {
                    DFEVariant local_dfe = region->dfe;
                    s_val = draw_s(local_dfe, rng);
                    h_val = region->dominance_h;
                }

                AlleleID new_id = table.new_allele(
                    AlleleInfo{gen, s_val, h_val, {}});
                row[site] = new_id;
            }
        }
    }

    void operator()(SparseVariants& storage, AlleleTable& table,
                    Generation gen, std::mt19937_64& rng) const
    {
        const std::size_t H = storage.num_haplotypes();
        double total_rate = rate_map.total_rate();

        std::poisson_distribution<std::size_t> pois(total_rate);
        auto cw = rate_map.cumulative_weights();
        std::uniform_real_distribution<double> u(0.0, cw.back());

        for (std::size_t h = 0; h < H; ++h) {
            std::size_t n_mut = pois(rng);
            for (std::size_t m = 0; m < n_mut; ++m) {
                double r = u(rng);
                auto it = std::lower_bound(cw.begin(), cw.end(), r);
                SiteIndex site = static_cast<SiteIndex>(
                    std::distance(cw.begin(), it));
                if (site >= storage.num_sites()) site = storage.num_sites() - 1;

                double s_val = 0.0;
                double h_val = 0.5;
                auto* region = type_map.region_at(site);
                if (region) {
                    DFEVariant local_dfe = region->dfe;
                    s_val = draw_s(local_dfe, rng);
                    h_val = region->dominance_h;
                }

                AlleleID new_id = table.new_allele(
                    AlleleInfo{gen, s_val, h_val, {}});
                storage.offspring_set(h, site, new_id);
            }
        }
    }
};

}  // namespace gensim
