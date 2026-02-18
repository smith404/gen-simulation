// =============================================================================
// ancestry.hpp — Pedigree and lineage tracking for the simulation.
//
// Records parent-offspring relationships and recombination breakpoints each
// generation.  This enables:
//   - Pedigree reconstruction
//   - Identity-by-descent (IBD) analysis
//   - Coalescent-time estimation
//   - Tree-sequence recording (à la tskit / msprime)
//
// The tracker is a passive observer: it records events reported by the
// simulation loop and stores them in compact form.  It does NOT modify the
// simulation.
//
// Memory note: full pedigree for N individuals over G generations uses
// O(N·G) BirthRecords.  For very long runs, periodically flush old
// generations to disk or drop records beyond a window.
// =============================================================================
#pragma once

#include "types.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <optional>
#include <unordered_map>
#include <vector>

namespace gensim {

// ── IndividualID — globally unique individual identifier ────────────────────
// Generation + within-generation index.  This avoids a monotonic counter
// that might overflow, and makes generation-based queries natural.
struct IndividualID {
    Generation  gen   = 0;
    std::size_t index = 0;

    bool operator==(const IndividualID&) const noexcept = default;
};

struct IndividualIDHash {
    std::size_t operator()(const IndividualID& id) const noexcept {
        // Combine gen and index with a hash mix (safe for 32-bit size_t).
        std::size_t h1 = std::hash<std::uint64_t>{}(id.gen);
        std::size_t h2 = std::hash<std::size_t>{}(id.index);
        // boost::hash_combine style mixer — works on both 32- and 64-bit.
        h1 ^= h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
        return h1;
    }
};

// ── RecombinationRecord — where a crossover happened ────────────────────────
struct RecombinationRecord {
    std::size_t parent_index;              // index of the parent individual
    std::vector<SiteIndex> breakpoints;    // sorted crossover positions
};

// ── BirthRecord — one offspring's parentage ─────────────────────────────────
struct BirthRecord {
    IndividualID offspring;

    // Parent for haplotype 0 (maternal) and haplotype 1 (paternal).
    IndividualID parent0;
    IndividualID parent1;

    // Recombination breakpoints within each parent's genome.
    std::vector<SiteIndex> breakpoints0;   // crossover points for parent0
    std::vector<SiteIndex> breakpoints1;   // crossover points for parent1
};

// ── AncestryTracker ─────────────────────────────────────────────────────────
class AncestryTracker {
public:
    AncestryTracker() = default;

    /// Reserve space for an expected total number of birth events.
    explicit AncestryTracker(std::size_t reserve_hint) {
        records_.reserve(reserve_hint);
    }

    // ── Recording API (called by the simulation loop) ───────────────────────

    /// Begin a new generation.  Must be called before record_birth().
    void begin_generation(Generation gen) {
        current_gen_ = gen;
    }

    /// Record the birth of offspring `offspring_idx` from two parents.
    /// `breakpoints0` and `breakpoints1` are the crossover positions used
    /// to derive haplotype 0 and haplotype 1 respectively.
    void record_birth(std::size_t offspring_idx,
                      std::size_t parent0_idx,
                      std::size_t parent1_idx,
                      std::vector<SiteIndex> breakpoints0 = {},
                      std::vector<SiteIndex> breakpoints1 = {})
    {
        BirthRecord rec;
        rec.offspring   = {current_gen_ + 1, offspring_idx};
        rec.parent0     = {current_gen_, parent0_idx};
        rec.parent1     = {current_gen_, parent1_idx};
        rec.breakpoints0 = std::move(breakpoints0);
        rec.breakpoints1 = std::move(breakpoints1);
        records_.push_back(std::move(rec));
    }

    // ── Query API ───────────────────────────────────────────────────────────

    /// All birth records (chronological order).
    [[nodiscard]] const std::vector<BirthRecord>& records() const noexcept {
        return records_;
    }

    /// Birth records for a specific generation's offspring.
    [[nodiscard]] std::vector<const BirthRecord*>
    offspring_of_generation(Generation offspring_gen) const {
        std::vector<const BirthRecord*> result;
        for (auto& r : records_) {
            if (r.offspring.gen == offspring_gen)
                result.push_back(&r);
        }
        return result;
    }

    /// Find the birth record for a specific individual.
    [[nodiscard]] std::optional<BirthRecord>
    find(IndividualID id) const {
        for (auto& r : records_) {
            if (r.offspring == id) return r;
        }
        return std::nullopt;
    }

    /// Trace the matrilineal (parent0) lineage of an individual back
    /// through `max_depth` generations.
    [[nodiscard]] std::vector<IndividualID>
    trace_lineage(IndividualID start, std::size_t max_depth = 100) const {
        // Build a lookup map (lazy, could be cached).
        std::unordered_map<IndividualID, const BirthRecord*, IndividualIDHash>
            lookup;
        for (auto& r : records_) lookup[r.offspring] = &r;

        std::vector<IndividualID> lineage;
        lineage.push_back(start);
        IndividualID current = start;
        for (std::size_t d = 0; d < max_depth; ++d) {
            auto it = lookup.find(current);
            if (it == lookup.end()) break;
            current = it->second->parent0;
            lineage.push_back(current);
        }
        return lineage;
    }

    /// Count how many distinct parents contributed to a generation
    /// (effective number of breeders).
    [[nodiscard]] std::size_t
    effective_breeders(Generation offspring_gen) const {
        std::vector<IndividualID> parents;
        for (auto& r : records_) {
            if (r.offspring.gen != offspring_gen) continue;
            parents.push_back(r.parent0);
            parents.push_back(r.parent1);
        }
        std::sort(parents.begin(), parents.end(),
            [](auto& a, auto& b) {
                return a.gen != b.gen ? a.gen < b.gen : a.index < b.index;
            });
        parents.erase(std::unique(parents.begin(), parents.end()), parents.end());
        return parents.size();
    }

    /// Number of recorded birth events.
    [[nodiscard]] std::size_t size() const noexcept { return records_.size(); }

    /// Drop records older than `keep_from` generation (to save memory).
    void prune_before(Generation keep_from) {
        records_.erase(
            std::remove_if(records_.begin(), records_.end(),
                [keep_from](auto& r) { return r.offspring.gen < keep_from; }),
            records_.end());
    }

    /// Clear all records.
    void clear() { records_.clear(); }

private:
    Generation current_gen_ = 0;
    std::vector<BirthRecord> records_;
};

}  // namespace gensim
