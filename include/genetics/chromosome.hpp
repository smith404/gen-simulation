// =============================================================================
// chromosome.hpp — Multi-chromosome genome architecture.
//
// Real organisms have multiple chromosomes (linkage groups) with free
// recombination between them and variable recombination within.  This header
// provides:
//
//   ChromosomeSpec    — describes one chromosome (length, type, recomb rates)
//   GenomeSpec        — describes the full genome (vector of ChromosomeSpecs)
//   MultiChromGenome  — dense storage for a multi-chromosome diploid genome
//                       with per-chromosome double-buffered DenseHaplotypes
//
// Sex chromosome inheritance:
//   Autosomes  — diploid in all individuals
//   X          — diploid in females, hemizygous in males (one copy)
//   Y          — present only in males (hemizygous)
//   Mito       — haploid, maternally inherited, no recombination
//
// This module integrates with the existing single-chromosome storage by
// treating each chromosome's data as an independent DenseHaplotypes block.
// The recombination policy is applied per-chromosome, and chromosomes
// segregate independently (free recombination = independent assortment).
// =============================================================================
#pragma once

#include "allele_table.hpp"
#include "dense_haplotypes.hpp"
#include "recombination_model.hpp"
#include "types.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace gensim {

// ── ChromosomeSpec ──────────────────────────────────────────────────────────
/// Describes a single chromosome's physical and genetic properties.
struct ChromosomeSpec {
    ChromosomeID    id       = 0;
    std::string     name;               // e.g. "chr1", "chrX"
    std::size_t     num_sites = 0;      // number of sites on this chromosome
    ChromosomeType  type     = ChromosomeType::Autosome;

    /// Per-site recombination rate (uniform within chromosome).
    /// Set to 0 for non-recombining chromosomes (Y, mito, …).
    double recomb_rate = 1e-4;

    /// Optional: full recombination map for this chromosome.
    /// If empty, a uniform map is built from `recomb_rate`.
    RecombinationMap recomb_map;

    /// Build the effective recombination map (lazy).
    [[nodiscard]] RecombinationMap effective_map() const {
        if (!recomb_map.endpoints.empty()) return recomb_map;
        return RecombinationMap::uniform(num_sites, recomb_rate);
    }
};

// ── GenomeSpec ───────────────────────────────────────────────────────────────
/// Full genome architecture: an ordered list of chromosomes.
struct GenomeSpec {
    std::vector<ChromosomeSpec> chromosomes;

    /// Total number of sites across all chromosomes.
    [[nodiscard]] std::size_t total_sites() const noexcept {
        std::size_t total = 0;
        for (auto& c : chromosomes) total += c.num_sites;
        return total;
    }

    /// Number of chromosomes.
    [[nodiscard]] std::size_t num_chromosomes() const noexcept {
        return chromosomes.size();
    }

    /// Return the site offset into a flat genome for chromosome `ci`.
    [[nodiscard]] std::size_t chromosome_offset(std::size_t ci) const noexcept {
        std::size_t off = 0;
        for (std::size_t i = 0; i < ci && i < chromosomes.size(); ++i)
            off += chromosomes[i].num_sites;
        return off;
    }

    // ── Builders ────────────────────────────────────────────────────────────

    /// Create a simple genome with `n` autosomes of equal length.
    static GenomeSpec uniform_autosomes(std::size_t n_chrom,
                                         std::size_t sites_per_chrom,
                                         double recomb_rate = 1e-4)
    {
        GenomeSpec gs;
        gs.chromosomes.resize(n_chrom);
        for (std::size_t i = 0; i < n_chrom; ++i) {
            gs.chromosomes[i].id         = static_cast<ChromosomeID>(i);
            gs.chromosomes[i].name       = "chr" + std::to_string(i + 1);
            gs.chromosomes[i].num_sites  = sites_per_chrom;
            gs.chromosomes[i].type       = ChromosomeType::Autosome;
            gs.chromosomes[i].recomb_rate = recomb_rate;
        }
        return gs;
    }

    /// Add a sex chromosome pair (X + Y).
    GenomeSpec& add_sex_chromosomes(std::size_t x_sites, std::size_t y_sites,
                                     double x_recomb = 1e-4,
                                     double y_recomb = 0.0)
    {
        ChromosomeSpec cx;
        cx.id = static_cast<ChromosomeID>(chromosomes.size());
        cx.name = "chrX";
        cx.num_sites = x_sites;
        cx.type = ChromosomeType::X;
        cx.recomb_rate = x_recomb;
        chromosomes.push_back(cx);

        ChromosomeSpec cy;
        cy.id = static_cast<ChromosomeID>(chromosomes.size());
        cy.name = "chrY";
        cy.num_sites = y_sites;
        cy.type = ChromosomeType::Y;
        cy.recomb_rate = y_recomb;  // Y typically doesn't recombine
        chromosomes.push_back(cy);

        return *this;
    }

    /// Add a mitochondrial chromosome.
    GenomeSpec& add_mitochondrial(std::size_t sites) {
        ChromosomeSpec cm;
        cm.id = static_cast<ChromosomeID>(chromosomes.size());
        cm.name = "chrM";
        cm.num_sites = sites;
        cm.type = ChromosomeType::Mitochondrial;
        cm.recomb_rate = 0.0;
        chromosomes.push_back(cm);
        return *this;
    }
};

// ── MultiChromGenome ────────────────────────────────────────────────────────
/// Multi-chromosome diploid genome storage.
///
/// This stores genotypes as a single flat DenseHaplotypes block where
/// chromosomes are laid out contiguously.  The GenomeSpec provides the
/// boundaries.
///
/// For sex chromosomes, males have only one copy of X and one Y.
/// The hemizygous copy is stored in the haplotype-0 slot; haplotype-1
/// for the X in males is filled with kRefAllele (masked).
///
/// This class handles:
///   - Per-chromosome recombination with independent assortment
///   - Sex-aware inheritance (X from mother, Y from father, mito from mother)
///   - Mutation mapped to per-chromosome regions
class MultiChromGenome {
public:
    MultiChromGenome(std::size_t N, const GenomeSpec& spec)
        : spec_{spec}
        , N_{N}
        , storage_{N, spec.total_sites()}
    {}

    // ── Accessors ───────────────────────────────────────────────────────────
    [[nodiscard]] const GenomeSpec& spec() const noexcept { return spec_; }
    [[nodiscard]] std::size_t num_individuals() const noexcept { return N_; }
    [[nodiscard]] std::size_t total_sites() const noexcept { return spec_.total_sites(); }
    [[nodiscard]] DenseHaplotypes& storage() noexcept { return storage_; }
    [[nodiscard]] const DenseHaplotypes& storage() const noexcept { return storage_; }

    // ── Per-chromosome allele access ────────────────────────────────────────

    /// Get the allele at a specific chromosome position for a haplotype.
    [[nodiscard]] AlleleID get(std::size_t haplotype,
                               std::size_t chrom_idx,
                               std::size_t site_in_chrom) const
    {
        std::size_t flat = spec_.chromosome_offset(chrom_idx) + site_in_chrom;
        return storage_.get(haplotype, flat);
    }

    /// Set an offspring allele at a chromosome position.
    void offspring_set(std::size_t haplotype,
                       std::size_t chrom_idx,
                       std::size_t site_in_chrom,
                       AlleleID allele)
    {
        std::size_t flat = spec_.chromosome_offset(chrom_idx) + site_in_chrom;
        storage_.offspring_set(haplotype, flat, allele);
    }

    // ── Recombination with independent assortment ───────────────────────────
    /// Produce one offspring haplotype for a given chromosome by recombining
    /// two parental haplotypes.  The caller decides which parental haps to
    /// pass based on independent assortment and sex chromosome rules.
    void recombine_chromosome(std::size_t chrom_idx,
                               std::size_t parent_hap0,
                               std::size_t parent_hap1,
                               std::size_t offspring_hap,
                               std::mt19937_64& rng)
    {
        const auto& cs = spec_.chromosomes[chrom_idx];
        std::size_t offset = spec_.chromosome_offset(chrom_idx);
        std::size_t L = cs.num_sites;

        // Draw breakpoints using the chromosome's map.
        auto rmap = cs.effective_map();
        auto bps = draw_mapped_breakpoints(rmap, rng);

        const AlleleID* src[2] = {
            storage_.hap_ptr(parent_hap0) + offset,
            storage_.hap_ptr(parent_hap1) + offset
        };
        AlleleID* dst = storage_.offspring_hap_ptr(offspring_hap) + offset;

        std::size_t cur = 0;
        std::size_t prev = 0;
        for (auto bp : bps) {
            if (bp > L) bp = L;
            if (bp <= prev) continue;
            std::memcpy(dst + prev, src[cur] + prev,
                        (bp - prev) * sizeof(AlleleID));
            prev = bp;
            cur ^= 1;
        }
        std::memcpy(dst + prev, src[cur] + prev,
                    (L - prev) * sizeof(AlleleID));
    }

    // ── Full-genome meiosis with independent assortment ─────────────────────
    /// Produce one offspring haplotype by performing meiosis on a parent's
    /// diploid genome.  Each chromosome independently picks which parental
    /// copy goes first (= independent assortment), then recombines.
    ///
    /// For autosomal chromosomes:
    ///   - Randomly choose which parental hap is hap0 vs hap1
    ///   - Recombine to produce the offspring chrom copy
    ///
    /// maternal/paternal flags are used for sex chromosome logic.
    void meiosis(std::size_t parent_idx,
                 std::size_t offspring_hap,
                 Sex parent_sex,
                 bool is_maternal,  // true = this parent is the mother
                 std::mt19937_64& rng)
    {
        std::bernoulli_distribution coin(0.5);

        for (std::size_t ci = 0; ci < spec_.num_chromosomes(); ++ci) {
            const auto& cs = spec_.chromosomes[ci];
            std::size_t h0 = 2 * parent_idx;
            std::size_t h1 = 2 * parent_idx + 1;

            switch (cs.type) {
            case ChromosomeType::Autosome: {
                // Independent assortment: randomly choose orientation.
                if (coin(rng)) std::swap(h0, h1);
                recombine_chromosome(ci, h0, h1, offspring_hap, rng);
                break;
            }
            case ChromosomeType::X: {
                if (parent_sex == Sex::Female) {
                    // Female has two X copies: recombine normally.
                    if (coin(rng)) std::swap(h0, h1);
                    recombine_chromosome(ci, h0, h1, offspring_hap, rng);
                } else if (parent_sex == Sex::Male) {
                    // Male has one X (hap0): pass it directly.
                    // Only daughters receive it; sons don't get X from father.
                    copy_chromosome(ci, h0, offspring_hap);
                }
                break;
            }
            case ChromosomeType::Y: {
                if (parent_sex == Sex::Male) {
                    // Male passes Y to sons only.
                    copy_chromosome(ci, h0, offspring_hap);
                } else {
                    // Female: fill with reference (no Y).
                    fill_chromosome_ref(ci, offspring_hap);
                }
                break;
            }
            case ChromosomeType::Mitochondrial: {
                if (is_maternal) {
                    // Maternal inheritance, no recombination.
                    copy_chromosome(ci, h0, offspring_hap);
                } else {
                    // Paternal mito not passed (or fill ref).
                    fill_chromosome_ref(ci, offspring_hap);
                }
                break;
            }
            }
        }
    }

    // ── Buffer management ───────────────────────────────────────────────────
    void swap_buffers() { storage_.swap_buffers(); }

    void prepare_offspring() {
        // For DenseHaplotypes, nothing needed (double-buffered).
    }

private:
    /// Copy one chromosome from a parent haplotype to an offspring haplotype
    /// without recombination.
    void copy_chromosome(std::size_t chrom_idx,
                          std::size_t src_hap,
                          std::size_t dst_hap)
    {
        std::size_t offset = spec_.chromosome_offset(chrom_idx);
        std::size_t L = spec_.chromosomes[chrom_idx].num_sites;
        const AlleleID* src = storage_.hap_ptr(src_hap) + offset;
        AlleleID* dst = storage_.offspring_hap_ptr(dst_hap) + offset;
        std::memcpy(dst, src, L * sizeof(AlleleID));
    }

    /// Fill a chromosome region in the offspring buffer with reference allele.
    void fill_chromosome_ref(std::size_t chrom_idx, std::size_t hap) {
        std::size_t offset = spec_.chromosome_offset(chrom_idx);
        std::size_t L = spec_.chromosomes[chrom_idx].num_sites;
        AlleleID* dst = storage_.offspring_hap_ptr(hap) + offset;
        std::fill(dst, dst + L, kRefAllele);
    }

    GenomeSpec        spec_;
    std::size_t       N_;
    DenseHaplotypes   storage_;
};

}  // namespace gensim
