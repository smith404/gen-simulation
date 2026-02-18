// =============================================================================
// sparse_variants.hpp — Sparse haplotype storage backend.
//
// Instead of materialising every (haplotype × site) cell, we store only those
// sites that carry at least one non-reference allele.  Each such site is a
// Variant record:
//
//   struct Variant { SiteIndex pos;  std::vector<AlleleID> alleles; };
//
// `alleles` has length 2*N — one entry per haplotype.  If a haplotype carries
// the reference allele at that site, its entry is kRefAllele (0).
//
// Variants are kept sorted by `pos` for binary-search lookup.  This backend
// is memory-efficient when the number of segregating sites is much smaller
// than L (typical under neutral models with low μ).
//
// Double-buffering follows the same pattern as DenseHaplotypes: offspring are
// assembled in `back_variants_`, then swapped to `front_variants_`.
// =============================================================================
#pragma once

#include "types.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace gensim {

// ── Variant record ──────────────────────────────────────────────────────────
struct Variant {
    SiteIndex              pos;       // site position (0-based)
    std::vector<AlleleID>  alleles;   // length = 2*N, one per haplotype

    // Keep sorted by position for binary search.
    bool operator<(const Variant& o) const noexcept { return pos < o.pos; }
};

// ── SparseVariants ──────────────────────────────────────────────────────────
class SparseVariants {
public:
    SparseVariants() = default;

    SparseVariants(std::size_t num_individuals, std::size_t num_sites)
        : n_{num_individuals}, l_{num_sites}
    {
        // Variants vectors start empty — all sites are reference.
        // Reserve a reasonable amount to avoid early reallocs.
        front_.reserve(256);
        back_.reserve(256);
    }

    // ── HaplotypeStorage concept interface ──────────────────────────────────

    [[nodiscard]] std::size_t num_individuals() const noexcept { return n_; }
    [[nodiscard]] std::size_t num_sites()       const noexcept { return l_; }
    [[nodiscard]] std::size_t num_haplotypes()  const noexcept { return 2 * n_; }

    // NOTE: hap_ptr() is NOT provided for the sparse backend because there is
    // no contiguous per-haplotype row.  Instead, we expose per-site accessors.
    // Algorithms (recombination, mutation) are specialised per-backend.

    // ── Per-site access ─────────────────────────────────────────────────────

    /// Get allele for haplotype `hap` at site `pos`.
    [[nodiscard]] AlleleID get(std::size_t hap, SiteIndex pos) const noexcept {
        auto it = find_variant(pos);
        if (it == front_.end() || it->pos != pos) return kRefAllele;
        return it->alleles[hap];
    }

    /// Set allele for haplotype `hap` at site `pos` in the FRONT buffer.
    void set(std::size_t hap, SiteIndex pos, AlleleID allele) {
        auto it = find_variant_mut(pos);
        if (it == front_.end() || it->pos != pos) {
            // Insert a new Variant record (all ref, then set this haplotype).
            Variant v;
            v.pos = pos;
            v.alleles.assign(2 * n_, kRefAllele);
            v.alleles[hap] = allele;
            front_.insert(it, std::move(v));  // keeps sorted order
        } else {
            it->alleles[hap] = allele;
        }
    }

    // ── Offspring buffer access ─────────────────────────────────────────────

    [[nodiscard]] AlleleID offspring_get(std::size_t hap, SiteIndex pos) const noexcept {
        auto it = find_back_variant(pos);
        if (it == back_.end() || it->pos != pos) return kRefAllele;
        return it->alleles[hap];
    }

    void offspring_set(std::size_t hap, SiteIndex pos, AlleleID allele) {
        auto it = find_back_variant_mut(pos);
        if (it == back_.end() || it->pos != pos) {
            Variant v;
            v.pos = pos;
            v.alleles.assign(2 * n_, kRefAllele);
            v.alleles[hap] = allele;
            back_.insert(it, std::move(v));
        } else {
            it->alleles[hap] = allele;
        }
    }

    // ── Bulk offspring operations ───────────────────────────────────────────

    /// Copy the allele at every variant site from parent haplotype `src` (front)
    /// to offspring haplotype `dst` (back) for sites in [begin, end).
    void copy_segment(std::size_t src_hap, std::size_t dst_hap,
                      SiteIndex site_begin, SiteIndex site_end)
    {
        // Walk front variants in [site_begin, site_end).
        auto lo = std::lower_bound(front_.begin(), front_.end(), site_begin,
                                   [](const Variant& v, SiteIndex p){ return v.pos < p; });
        for (auto it = lo; it != front_.end() && it->pos < site_end; ++it) {
            AlleleID a = it->alleles[src_hap];
            if (a != kRefAllele) {
                offspring_set(dst_hap, it->pos, a);
            }
        }
    }

    // ── Buffer management ───────────────────────────────────────────────────

    void clear_offspring_buffer() { back_.clear(); }

    void swap_buffers() {
        front_.swap(back_);
        back_.clear();           // fresh buffer for next generation
    }

    // ── Resize ──────────────────────────────────────────────────────────────

    void resize(std::size_t new_n, std::size_t new_l) {
        n_ = new_n;
        l_ = new_l;
        front_.clear();
        back_.clear();
    }

    // ── Statistics ──────────────────────────────────────────────────────────

    [[nodiscard]] std::size_t count_segregating_sites() const noexcept {
        // Every Variant with at least one non-ref allele is segregating.
        std::size_t seg = 0;
        for (auto& v : front_) {
            for (auto a : v.alleles) {
                if (a != kRefAllele) { ++seg; break; }
            }
        }
        return seg;
    }

    [[nodiscard]] std::vector<std::pair<AlleleID, std::size_t>>
    allele_counts(SiteIndex site) const {
        std::vector<std::pair<AlleleID, std::size_t>> counts;
        auto it = find_variant(site);
        if (it == front_.end() || it->pos != site) {
            // All haplotypes carry the reference allele.
            counts.emplace_back(kRefAllele, 2 * n_);
            return counts;
        }
        for (auto a : it->alleles) {
            auto jt = std::find_if(counts.begin(), counts.end(),
                                   [a](auto& p){ return p.first == a; });
            if (jt != counts.end()) ++jt->second;
            else counts.emplace_back(a, 1);
        }
        return counts;
    }

    // ── Direct variant access (for mutation model) ──────────────────────────

    [[nodiscard]] const std::vector<Variant>& variants() const noexcept { return front_; }
    [[nodiscard]]       std::vector<Variant>& variants()       noexcept { return front_; }
    [[nodiscard]] const std::vector<Variant>& back_variants() const noexcept { return back_; }
    [[nodiscard]]       std::vector<Variant>& back_variants()       noexcept { return back_; }

private:
    std::size_t n_ = 0;
    std::size_t l_ = 0;
    std::vector<Variant> front_;   // current-generation variants
    std::vector<Variant> back_;    // offspring-generation variants

    // ── Binary-search helpers (front) ───────────────────────────────────────
    [[nodiscard]] std::vector<Variant>::const_iterator
    find_variant(SiteIndex pos) const noexcept {
        return std::lower_bound(front_.cbegin(), front_.cend(), pos,
                                [](const Variant& v, SiteIndex p){ return v.pos < p; });
    }

    [[nodiscard]] std::vector<Variant>::iterator
    find_variant_mut(SiteIndex pos) noexcept {
        return std::lower_bound(front_.begin(), front_.end(), pos,
                                [](const Variant& v, SiteIndex p){ return v.pos < p; });
    }

    // ── Binary-search helpers (back) ────────────────────────────────────────
    [[nodiscard]] std::vector<Variant>::const_iterator
    find_back_variant(SiteIndex pos) const noexcept {
        return std::lower_bound(back_.cbegin(), back_.cend(), pos,
                                [](const Variant& v, SiteIndex p){ return v.pos < p; });
    }

    [[nodiscard]] std::vector<Variant>::iterator
    find_back_variant_mut(SiteIndex pos) noexcept {
        return std::lower_bound(back_.begin(), back_.end(), pos,
                                [](const Variant& v, SiteIndex p){ return v.pos < p; });
    }
};

}  // namespace gensim
