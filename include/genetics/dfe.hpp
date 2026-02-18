// =============================================================================
// dfe.hpp — Distribution of Fitness Effects (DFE).
//
// When a new mutation arises, its selection coefficient s is drawn from a DFE
// rather than being fixed.  Standard distributions from the population
// genetics literature are provided:
//
//   Fixed        – constant s (point mass)
//   Exponential  – deleterious mutations (Eyre-Walker & Keightley 2007)
//   Gamma        – flexible shape for deleterious DFE
//   Normal       – symmetric around a mean s
//   Uniform      – bounded range [lo, hi]
//   Mixture      – weighted mixture of any of the above
//
// Each DFE is a lightweight callable: double operator()(RNG& rng) const.
// =============================================================================
#pragma once

#include <cmath>
#include <random>
#include <variant>
#include <vector>

namespace gensim {

// ── Concrete DFE types ──────────────────────────────────────────────────────

/// Point mass: always returns the same s.
struct FixedDFE {
    double s = 0.0;

    double operator()(std::mt19937_64& /*rng*/) const noexcept { return s; }
};

/// Exponential distribution (mean = mean_s).
/// Typically mean_s < 0 for deleterious mutations.
struct ExponentialDFE {
    double mean_s = -0.01;  // negative = deleterious

    double operator()(std::mt19937_64& rng) const {
        std::exponential_distribution<double> dist(1.0 / std::abs(mean_s));
        double val = dist(rng);
        return (mean_s < 0) ? -val : val;
    }
};

/// Gamma distribution (shape alpha, mean = mean_s).
/// Scale beta = |mean_s| / alpha.
struct GammaDFE {
    double mean_s = -0.01;
    double shape  = 0.3;     // alpha < 1: leptokurtic (many weak, few strong)

    double operator()(std::mt19937_64& rng) const {
        double scale = std::abs(mean_s) / shape;
        std::gamma_distribution<double> dist(shape, scale);
        double val = dist(rng);
        return (mean_s < 0) ? -val : val;
    }
};

/// Normal distribution (mean, stddev).
struct NormalDFE {
    double mean_s  = -0.01;
    double stddev  = 0.005;

    double operator()(std::mt19937_64& rng) const {
        std::normal_distribution<double> dist(mean_s, stddev);
        return dist(rng);
    }
};

/// Uniform distribution on [lo, hi].
struct UniformDFE {
    double lo = -0.05;
    double hi =  0.0;

    double operator()(std::mt19937_64& rng) const {
        std::uniform_real_distribution<double> dist(lo, hi);
        return dist(rng);
    }
};

// ── Type-erased DFE (variant-based, no vtable) ─────────────────────────────

/// A DFE that can hold any of the concrete types and be called uniformly.
/// Use this when you want to choose the DFE at runtime.
using DFEVariant = std::variant<FixedDFE, ExponentialDFE, GammaDFE,
                                 NormalDFE, UniformDFE>;

/// Draw a selection coefficient from a type-erased DFE.
inline double draw_s(DFEVariant& dfe, std::mt19937_64& rng) {
    return std::visit([&](auto& d) { return d(rng); }, dfe);
}

// ── Mixture DFE ─────────────────────────────────────────────────────────────

/// Weighted mixture of DFE components.
/// Example: 70% neutral (s=0), 25% gamma deleterious, 5% exponential beneficial.
struct MixtureDFE {
    struct Component {
        double     weight;
        DFEVariant dfe;
    };
    std::vector<Component> components;

    double operator()(std::mt19937_64& rng) const {
        // Build cumulative weights.
        double total = 0.0;
        for (auto& c : components) total += c.weight;
        std::uniform_real_distribution<double> u(0.0, total);
        double r = u(rng);
        double cum = 0.0;
        for (auto& c : components) {
            cum += c.weight;
            if (r <= cum) {
                // const_cast safe: draw_s doesn't mutate the DFE state.
                return draw_s(const_cast<DFEVariant&>(c.dfe), rng);
            }
        }
        // Fallback (shouldn't happen).
        return draw_s(const_cast<DFEVariant&>(components.back().dfe), rng);
    }
};

}  // namespace gensim
