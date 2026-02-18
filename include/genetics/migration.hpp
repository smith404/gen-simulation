// =============================================================================
// migration.hpp — Multiple subpopulations (demes) with migration.
//
// Implements a metapopulation model where individuals are assigned to demes.
// Migration is modelled by a migration matrix M where M[i][j] is the
// probability that an offspring in deme i has a parent from deme j.
// Rows must sum to 1.0 (or be normalised automatically).
//
// This module provides:
//   - DemeAssignment: maps individual index -> DemeID
//   - MigrationMatrix: stores M[i][j] and samples a source deme
//   - Helpers for the Simulation to use during parent selection
//
// Integration with the existing design:
//   The Population class remains unchanged.  The Simulation driver uses
//   DemeAssignment to partition individuals and MigrationMatrix to decide
//   which deme to draw each parent from.  This is a "soft" migration model
//   (backward-looking: where did the parent come from?) which is standard
//   in Wright-Fisher simulators.
// =============================================================================
#pragma once

#include "types.hpp"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace gensim {

// ── DemeAssignment ──────────────────────────────────────────────────────────
/// Maps each individual index to a DemeID.  For a simple island model with
/// equal deme sizes, individual i belongs to deme (i * num_demes / N).

class DemeAssignment {
public:
    DemeAssignment() = default;

    /// Construct with `num_demes` demes of equal size from N individuals.
    DemeAssignment(std::size_t N, std::size_t num_demes)
        : num_demes_{num_demes}
    {
        assignment_.resize(N);
        for (std::size_t i = 0; i < N; ++i) {
            assignment_[i] = static_cast<DemeID>(i * num_demes / N);
        }
        rebuild_index();
    }

    /// Construct with explicit per-individual deme assignments.
    explicit DemeAssignment(std::vector<DemeID> assignment)
        : assignment_{std::move(assignment)}
    {
        if (!assignment_.empty()) {
            num_demes_ = *std::max_element(assignment_.begin(),
                                            assignment_.end()) + 1;
        }
        rebuild_index();
    }

    [[nodiscard]] DemeID deme_of(std::size_t individual) const noexcept {
        return assignment_[individual];
    }

    [[nodiscard]] std::size_t num_demes() const noexcept { return num_demes_; }

    /// Get all individual indices belonging to a given deme.
    [[nodiscard]] const std::vector<std::size_t>&
    individuals_in(DemeID d) const noexcept {
        return deme_members_[d];
    }

    /// Number of individuals in a given deme.
    [[nodiscard]] std::size_t deme_size(DemeID d) const noexcept {
        return deme_members_[d].size();
    }

    /// Total individuals.
    [[nodiscard]] std::size_t total_size() const noexcept {
        return assignment_.size();
    }

    /// Reassign an individual to a new deme (e.g., after migration event).
    void set_deme(std::size_t individual, DemeID new_deme) {
        assignment_[individual] = new_deme;
        // Note: rebuild_index() should be called after batch updates.
    }

    /// Rebuild the per-deme member lists.  Call after batch set_deme() calls.
    void rebuild_index() {
        deme_members_.clear();
        deme_members_.resize(num_demes_);
        for (std::size_t i = 0; i < assignment_.size(); ++i) {
            deme_members_[assignment_[i]].push_back(i);
        }
    }

private:
    std::size_t                           num_demes_ = 1;
    std::vector<DemeID>                   assignment_;
    std::vector<std::vector<std::size_t>> deme_members_;
};

// ── MigrationMatrix ─────────────────────────────────────────────────────────
/// M[i][j] = probability that an offspring in deme i has a parent from deme j.
/// Diagonal entries M[i][i] = probability of local mating (no migration).
///
/// For a symmetric island model with migration rate m:
///   M[i][i] = 1 - m
///   M[i][j] = m / (D - 1)  for j != i, where D = number of demes.

class MigrationMatrix {
public:
    MigrationMatrix() = default;

    /// Construct a D×D matrix, all zeros (to be filled manually).
    explicit MigrationMatrix(std::size_t num_demes)
        : D_{num_demes}, matrix_(num_demes, std::vector<double>(num_demes, 0.0))
    {
        // Default: no migration (identity).
        for (std::size_t i = 0; i < D_; ++i) matrix_[i][i] = 1.0;
    }

    /// Construct a symmetric island model with migration rate m.
    static MigrationMatrix island_model(std::size_t num_demes, double m) {
        MigrationMatrix mm(num_demes);
        double off_diag = m / static_cast<double>(num_demes - 1);
        for (std::size_t i = 0; i < num_demes; ++i) {
            for (std::size_t j = 0; j < num_demes; ++j) {
                mm.matrix_[i][j] = (i == j) ? (1.0 - m) : off_diag;
            }
        }
        return mm;
    }

    /// Set a specific entry.
    void set(DemeID target, DemeID source, double prob) {
        matrix_[target][source] = prob;
    }

    /// Get a specific entry.
    [[nodiscard]] double get(DemeID target, DemeID source) const noexcept {
        return matrix_[target][source];
    }

    /// Sample a source deme for an offspring in deme `target`.
    [[nodiscard]] DemeID sample_source_deme(DemeID target,
                                             std::mt19937_64& rng) const
    {
        std::discrete_distribution<std::size_t> dist(
            matrix_[target].begin(), matrix_[target].end());
        return static_cast<DemeID>(dist(rng));
    }

    /// Number of demes.
    [[nodiscard]] std::size_t num_demes() const noexcept { return D_; }

    /// Validate: check that all rows sum to ~1.
    [[nodiscard]] bool is_valid(double tol = 1e-6) const noexcept {
        for (std::size_t i = 0; i < D_; ++i) {
            double sum = std::accumulate(matrix_[i].begin(),
                                          matrix_[i].end(), 0.0);
            if (std::abs(sum - 1.0) > tol) return false;
        }
        return true;
    }

private:
    std::size_t D_ = 0;
    std::vector<std::vector<double>> matrix_;
};

}  // namespace gensim
