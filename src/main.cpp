// =============================================================================
// main.cpp — Runnable example for the diploid population genetics simulator.
//
// Demonstrates all library features:
//   1. Dense backend with neutral evolution + summary statistics
//   2. Sparse backend with neutral evolution
//   3. Additive fitness with DFE-drawn mutations + dominance
//   4. Recombination hotspot map (MappedCrossover)
//   5. Demographic bottleneck schedule
//   6. Garbage collection for fixed/lost mutations
//   7. Island-model migration across demes
//
// Build (MSVC):
//   cl /O2 /std:c++20 /EHsc /Iinclude /W4 src\main.cpp /Fe:gensim.exe
//
// Build (GCC/Clang):
//   g++ -O3 -std=c++20 -Iinclude -o gensim src/main.cpp
//
// Run:
//   ./gensim                          # defaults
//   ./gensim 200 2000 40 1e-4 42      # N L G mu seed
// =============================================================================

#include "genetics/allele_table.hpp"
#include "genetics/demographics.hpp"
#include "genetics/dense_haplotypes.hpp"
#include "genetics/dfe.hpp"
#include "genetics/fitness_model.hpp"
#include "genetics/gc.hpp"
#include "genetics/migration.hpp"
#include "genetics/mutation_model.hpp"
#include "genetics/population.hpp"
#include "genetics/recombination_model.hpp"
#include "genetics/simulation.hpp"
#include "genetics/sparse_variants.hpp"
#include "genetics/statistics.hpp"
#include "genetics/types.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// =====================================================================
// 1. Dense neutral demo with summary statistics
// =====================================================================
static void run_dense_neutral_demo(std::size_t N, std::size_t L,
                                    gensim::Generation G,
                                    double mu, std::uint64_t seed)
{
    using namespace gensim;
    std::cout << "============================================================\n"
              << " [1] Dense neutral | N=" << N << " L=" << L
              << " G=" << G << " mu=" << mu << "\n"
              << "============================================================\n";

    AlleleTable table;
    BernoulliInfiniteAlleles mut{mu};
    SingleCrossover          rec;
    NeutralFitness           fit;

    using Pop = Population<DenseHaplotypes, BernoulliInfiniteAlleles,
                           SingleCrossover, NeutralFitness>;
    Pop pop(N, L, mut, rec, fit, table);
    Simulation<Pop> sim(pop, seed);

    auto t0 = std::chrono::high_resolution_clock::now();

    sim.run(G, [&](Generation g, const Pop& p) {
        if ((g + 1) % 10 == 0 || g == 0) {
            auto ss = compute_summary_dense(p.storage());
            std::cout << "gen " << std::setw(4) << (g + 1)
                      << "  seg=" << std::setw(5) << ss.segregating_sites
                      << "  pi=" << std::fixed << std::setprecision(4)
                      << ss.pi
                      << "  thetaW=" << ss.theta_w
                      << "  D=" << std::setprecision(3) << ss.tajimas_d
                      << "  H=" << std::setprecision(5) << ss.mean_heterozygosity
                      << "\n";
        }
    });

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "  Time: " << std::fixed << std::setprecision(3)
              << std::chrono::duration<double>(t1 - t0).count() << " s\n";

    // Final SFS
    auto final_ss = compute_summary_dense(pop.storage());
    std::cout << "  Final SFS (first 15 bins):\n";
    print_sfs(final_ss.sfs, 15);
    std::cout << "\n";
}

// =====================================================================
// 2. Sparse neutral demo
// =====================================================================
static void run_sparse_neutral_demo(std::size_t N, std::size_t L,
                                     gensim::Generation G,
                                     double mu, std::uint64_t seed)
{
    using namespace gensim;
    std::cout << "============================================================\n"
              << " [2] Sparse neutral | N=" << N << " L=" << L
              << " G=" << G << " mu=" << mu << "\n"
              << "============================================================\n";

    AlleleTable table;
    BernoulliInfiniteAlleles mut{mu};
    SingleCrossover          rec;
    NeutralFitness           fit;

    using Pop = Population<SparseVariants, BernoulliInfiniteAlleles,
                           SingleCrossover, NeutralFitness>;
    Pop pop(N, L, mut, rec, fit, table);
    Simulation<Pop> sim(pop, seed);

    auto t0 = std::chrono::high_resolution_clock::now();

    sim.run(G, [&](Generation g, const Pop& p) {
        if ((g + 1) % 10 == 0 || g == 0) {
            auto ss = compute_summary_sparse(p.storage());
            std::cout << "gen " << std::setw(4) << (g + 1)
                      << "  seg=" << std::setw(5) << ss.segregating_sites
                      << "  pi=" << std::fixed << std::setprecision(4)
                      << ss.pi
                      << "  thetaW=" << ss.theta_w
                      << "\n";
        }
    });

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "  Time: " << std::fixed << std::setprecision(3)
              << std::chrono::duration<double>(t1 - t0).count() << " s\n\n";
}

// =====================================================================
// 3. Additive fitness with DFE + dominance
// =====================================================================
static void run_dfe_additive_demo(std::uint64_t seed)
{
    using namespace gensim;
    const std::size_t N = 200, L = 1000;
    const Generation  G = 30;
    const double      mu = 5e-4;

    std::cout << "============================================================\n"
              << " [3] Additive fitness + Gamma DFE | N=" << N << " L=" << L
              << " G=" << G << " mu=" << mu << "\n"
              << "============================================================\n";

    AlleleTable table;

    // DFE-based mutation model: gamma-distributed s, h = 0.3 (partially recessive)
    DFEMutation mut{mu, GammaDFE{-0.01, 0.3}, 0.3};
    MultiBreakpoint rec{2.0};      // ~2 crossovers per meiosis
    AdditiveFitness fit;

    using Pop = Population<DenseHaplotypes, DFEMutation,
                           MultiBreakpoint, AdditiveFitness>;
    Pop pop(N, L, mut, rec, fit, table);
    Simulation<Pop> sim(pop, seed);

    sim.run(G, [&](Generation g, const Pop& p) {
        auto ss = compute_summary_dense(p.storage());
        std::cout << "gen " << std::setw(3) << (g + 1)
                  << "  mean_w=" << std::fixed << std::setprecision(4)
                  << p.mean_fitness()
                  << "  seg=" << ss.segregating_sites
                  << "  pi=" << ss.pi
                  << "  D=" << std::setprecision(3) << ss.tajimas_d
                  << "\n";
    });
    std::cout << "\n";
}

// =====================================================================
// 4. Recombination hotspot map (MappedCrossover)
// =====================================================================
static void run_recomb_map_demo(std::uint64_t seed)
{
    using namespace gensim;
    const std::size_t N = 200, L = 2000;
    const Generation  G = 30;
    const double      mu = 3e-4;

    std::cout << "============================================================\n"
              << " [4] Recombination hotspot map | N=" << N << " L=" << L
              << " G=" << G << "\n"
              << "============================================================\n";

    // Build a map with a 10x hotspot in the middle
    RecombinationMap rmap;
    rmap.endpoints = {0, 800, 1200, 2000};
    rmap.rates     = {1e-4, 1e-3, 1e-4};      // background / hotspot / background
    std::cout << "  Map: background=1e-4, hotspot[800-1200]=1e-3\n"
              << "  Total map length = " << std::fixed << std::setprecision(4)
              << rmap.total_map_length() << "\n";

    AlleleTable table;
    BernoulliInfiniteAlleles mut{mu};
    MappedCrossover          rec{rmap};
    NeutralFitness           fit;

    using Pop = Population<DenseHaplotypes, BernoulliInfiniteAlleles,
                           MappedCrossover, NeutralFitness>;
    Pop pop(N, L, mut, rec, fit, table);
    Simulation<Pop> sim(pop, seed);

    sim.run(G, [&](Generation g, const Pop& p) {
        if ((g + 1) % 10 == 0) {
            auto ss = compute_summary_dense(p.storage());
            std::cout << "gen " << std::setw(3) << (g + 1)
                      << "  seg=" << ss.segregating_sites
                      << "  pi=" << std::fixed << std::setprecision(4) << ss.pi
                      << "\n";
        }
    });
    std::cout << "\n";
}

// =====================================================================
// 5. Demographic bottleneck
// =====================================================================
static void run_demography_demo(std::uint64_t seed)
{
    using namespace gensim;
    const std::size_t N0 = 500, L = 1000;
    const Generation  G = 60;
    const double      mu = 2e-4;

    std::cout << "============================================================\n"
              << " [5] Demographic bottleneck | N0=" << N0 << " L=" << L
              << " G=" << G << "\n"
              << "============================================================\n";

    // Schedule: bottleneck to 50 at gen 20 for 10 generations, restore to 500
    DemographicSchedule sched;
    sched.add_bottleneck(20, 50, 10, N0);
    std::cout << "  Schedule: bottleneck N=50 at gen 20-30, restore N=500\n";

    AlleleTable table;
    BernoulliInfiniteAlleles mut{mu};
    SingleCrossover          rec;
    NeutralFitness           fit;

    // We'll manually drive the sim loop so we can resize the population
    // according to the demographic schedule.
    // For simplicity we re-create the population when N changes.
    // In production code you'd add a resize() method to Population.

    std::size_t current_N = N0;
    using Pop = Population<DenseHaplotypes, BernoulliInfiniteAlleles,
                           SingleCrossover, NeutralFitness>;

    // We need to manage pop lifetime carefully.  Use a pointer.
    auto pop = std::make_unique<Pop>(current_N, L, mut, rec, fit, table);
    std::mt19937_64 rng(seed);

    for (Generation g = 0; g < G; ++g) {
        std::size_t new_N = sched.population_size(g, current_N);

        if (new_N != current_N) {
            // Rebuild population with new size.
            // In a real simulator, you'd copy genotypes from the old
            // population (sampling with replacement to grow, or truncating
            // to shrink).  Here we just start fresh at the new size to
            // demonstrate the schedule.
            current_N = new_N;
            pop = std::make_unique<Pop>(current_N, L, mut, rec, fit, table);
        }

        // Run one generation step manually.
        pop->compute_fitness();
        pop->prepare_offspring_buffer();
        const auto& fv = pop->fitness();
        std::discrete_distribution<std::size_t> pdist(fv.begin(), fv.end());
        for (std::size_t i = 0; i < current_N; ++i) {
            pop->recombine(pdist(rng), 2 * i,     rng);
            pop->recombine(pdist(rng), 2 * i + 1, rng);
        }
        pop->mutate(g, rng);
        pop->advance();

        if ((g + 1) % 5 == 0 || g == 0) {
            auto ss = compute_summary_dense(pop->storage());
            std::cout << "gen " << std::setw(3) << (g + 1)
                      << "  N=" << std::setw(4) << current_N
                      << "  seg=" << ss.segregating_sites
                      << "  pi=" << std::fixed << std::setprecision(4) << ss.pi
                      << "\n";
        }
    }
    std::cout << "\n";
}

// =====================================================================
// 6. Garbage collection demo
// =====================================================================
static void run_gc_demo(std::uint64_t seed)
{
    using namespace gensim;
    const std::size_t N = 100, L = 500;
    const Generation  G = 50;
    const double      mu = 1e-3;  // high mutation rate for visibility

    std::cout << "============================================================\n"
              << " [6] Garbage collection | N=" << N << " L=" << L
              << " G=" << G << " mu=" << mu << "\n"
              << "============================================================\n";

    AlleleTable table;
    BernoulliInfiniteAlleles mut{mu};
    SingleCrossover          rec;
    NeutralFitness           fit;

    using Pop = Population<DenseHaplotypes, BernoulliInfiniteAlleles,
                           SingleCrossover, NeutralFitness>;
    Pop pop(N, L, mut, rec, fit, table);
    Simulation<Pop> sim(pop, seed);

    double total_fitness_offset = 0.0;
    std::size_t total_fixed = 0, total_lost = 0;

    sim.run(G, [&](Generation g, const Pop& p) {
        // Run GC every 10 generations.
        if ((g + 1) % 10 == 0) {
            // We need non-const storage for GC.  The Population exposes
            // a mutable storage() accessor.
            auto& mutable_pop = const_cast<Pop&>(p);
            auto result = gc_sweep_dense(mutable_pop.storage(), table,
                                          g + 1);
            total_fixed += result.fixed_count;
            total_lost  += result.lost_count;
            total_fitness_offset += result.fitness_offset;

            std::cout << "gen " << std::setw(3) << (g + 1)
                      << "  GC: fixed=" << result.fixed_count
                      << " lost=" << result.lost_count
                      << " subs=" << result.substitutions.size()
                      << "  table_size=" << table.size()
                      << "  seg=" << p.storage().count_segregating_sites()
                      << "\n";
        }
    });

    std::cout << "  Totals: " << total_fixed << " fixations, "
              << total_lost << " lost (tracked via sweep)\n"
              << "  Fitness offset from fixations: " << std::fixed
              << std::setprecision(6) << total_fitness_offset << "\n\n";
}

// =====================================================================
// 7. Island-model migration demo
// =====================================================================
static void run_migration_demo(std::uint64_t seed)
{
    using namespace gensim;
    const std::size_t   num_demes = 3;
    const std::size_t   N_per_deme = 60;
    const std::size_t   N_total = num_demes * N_per_deme;
    const std::size_t   L = 500;
    const Generation    G = 40;
    const double        mu = 3e-4;
    const double        mig_rate = 0.05;  // 5% migration per generation

    std::cout << "============================================================\n"
              << " [7] Island-model migration | " << num_demes << " demes x "
              << N_per_deme << " = " << N_total << " | m=" << mig_rate
              << " | G=" << G << "\n"
              << "============================================================\n";

    AlleleTable table;
    BernoulliInfiniteAlleles mut{mu};
    SingleCrossover          rec;
    NeutralFitness           fit;

    using Pop = Population<DenseHaplotypes, BernoulliInfiniteAlleles,
                           SingleCrossover, NeutralFitness>;
    Pop pop(N_total, L, mut, rec, fit, table);

    DemeAssignment demes(N_total, num_demes);
    MigrationMatrix mig = MigrationMatrix::island_model(num_demes, mig_rate);

    std::cout << "  Migration matrix valid: "
              << (mig.is_valid() ? "yes" : "NO") << "\n";
    for (std::size_t i = 0; i < num_demes; ++i) {
        std::cout << "    deme " << i << " has "
                  << demes.deme_size(static_cast<DemeID>(i))
                  << " individuals\n";
    }

    std::mt19937_64 rng(seed);

    for (Generation g = 0; g < G; ++g) {
        pop.compute_fitness();
        pop.prepare_offspring_buffer();

        const auto& fv = pop.fitness();

        // For each offspring, determine its deme, then sample source deme
        // via the migration matrix, then pick a parent from that source deme.
        for (std::size_t i = 0; i < N_total; ++i) {
            DemeID offspring_deme = demes.deme_of(i);

            for (int hap_idx = 0; hap_idx < 2; ++hap_idx) {
                // Where does this parent come from?
                DemeID src = mig.sample_source_deme(offspring_deme, rng);
                const auto& members = demes.individuals_in(src);

                // Build fitness weights for just this deme's members.
                std::vector<double> deme_fit;
                deme_fit.reserve(members.size());
                for (auto m : members) deme_fit.push_back(fv[m]);

                std::discrete_distribution<std::size_t> ddist(
                    deme_fit.begin(), deme_fit.end());
                std::size_t parent = members[ddist(rng)];

                pop.recombine(parent, 2 * i + hap_idx, rng);
            }
        }

        pop.mutate(g, rng);
        pop.advance();

        if ((g + 1) % 10 == 0 || g == 0) {
            auto ss = compute_summary_dense(pop.storage());
            std::cout << "gen " << std::setw(3) << (g + 1)
                      << "  seg=" << ss.segregating_sites
                      << "  pi=" << std::fixed << std::setprecision(4)
                      << ss.pi
                      << "  thetaW=" << ss.theta_w
                      << "\n";
        }
    }
    std::cout << "\n";
}

// =====================================================================
// main
// =====================================================================
int main(int argc, char* argv[])
{
    std::size_t        N    = 200;
    std::size_t        L    = 2000;
    gensim::Generation G    = 40;
    double             mu   = 1e-4;
    std::uint64_t      seed = 42;

    if (argc >= 2) N    = static_cast<std::size_t>(std::atol(argv[1]));
    if (argc >= 3) L    = static_cast<std::size_t>(std::atol(argv[2]));
    if (argc >= 4) G    = static_cast<gensim::Generation>(std::atol(argv[3]));
    if (argc >= 5) mu   = std::atof(argv[4]);
    if (argc >= 6) seed = static_cast<std::uint64_t>(std::atoll(argv[5]));

    std::cout << "+----------------------------------------------------------+\n"
              << "|  Diploid Population Genetics Simulator  --  Full Demo    |\n"
              << "+----------------------------------------------------------+\n\n";

    // 1. Dense neutral with summary statistics
    run_dense_neutral_demo(N, L, G, mu, seed);

    // 2. Sparse neutral (smaller params)
    const std::size_t sp_N = std::min(N, std::size_t{100});
    const std::size_t sp_L = std::min(L, std::size_t{1000});
    const auto        sp_G = std::min(G, gensim::Generation{30});
    run_sparse_neutral_demo(sp_N, sp_L, sp_G, mu, seed);

    // 3. Additive fitness with DFE + dominance
    run_dfe_additive_demo(seed);

    // 4. Recombination hotspot map
    run_recomb_map_demo(seed);

    // 5. Demographic bottleneck
    run_demography_demo(seed);

    // 6. Garbage collection
    run_gc_demo(seed);

    // 7. Island-model migration
    run_migration_demo(seed);

    std::cout << "+----------------------------------------------------------+\n"
              << "|  All demos completed successfully.                       |\n"
              << "+----------------------------------------------------------+\n";

    return 0;
}
