# Diploid Population Genetics Simulator

A header-only C++20 library for forward-in-time, Wright–Fisher-style simulation of diploid populations with configurable storage backends, pluggable mutation/recombination/fitness/selection policies, demographic events, migration, ancestry tracking, and comprehensive population-genetics statistics.

---

## Repository layout

```
include/genetics/
  gensim.hpp               Single-include convenience header (includes everything)
  types.hpp                Core type aliases, Individual struct
  concepts.hpp             C++20 concepts formalising every pluggable interface
  allele_table.hpp         Global allele registry with metadata
  dense_haplotypes.hpp     Contiguous-memory (vector) haplotype storage
  sparse_variants.hpp      Sparse variant-list haplotype storage
  mutation_model.hpp       Bernoulli infinite-alleles, finite-alleles, DFE-based
  recombination_model.hpp  Single-crossover, multi-breakpoint, mapped (hotspots)
  fitness_model.hpp        Neutral, additive, multiplicative, stabilizing selection
  selection.hpp            Pluggable parent selection (roulette, tournament, rank, …)
  mating_system.hpp        Mating system policies (sexual, selfing, assortative, …)
  dfe.hpp                  Distribution of Fitness Effects (fixed, gamma, mixture, …)
  chromosome.hpp           Multi-chromosome genome architecture + sex chromosomes
  quantitative_trait.hpp   Quantitative trait models (additive, dominance, multi-trait)
  mutation_rate_map.hpp    Variable mutation rates & region-specific DFEs
  output_formats.hpp       VCF, MS, PLINK, EIGENSTRAT export
  population.hpp           Population container (templated on backend + policies)
  simulation.hpp           Generation-loop drivers (basic + advanced)
  events.hpp               Fine-grained event/hook system for the simulation loop
  demographics.hpp         Demographic events (bottleneck, growth, size changes)
  migration.hpp            Multiple demes with migration matrix
  ancestry.hpp             Pedigree / lineage tracking
  statistics.hpp           SFS, pi, theta_W, Tajima's D, heterozygosity
  ld.hpp                   Linkage disequilibrium (r², D, D', LD decay)
  fst.hpp                  Fst / population differentiation, inbreeding (Fis)
  gc.hpp                   Garbage collection for fixed/lost mutations
  serialization.hpp        Binary checkpoint / restore
src/
  main.cpp                 Runnable demo (all features)
README.md                  This file
```

---

## Quick start

```cpp
#include "genetics/gensim.hpp"   // single include — everything

using namespace gensim;

int main() {
    AlleleTable table;
    BernoulliInfiniteAlleles mut{1e-5};
    SingleCrossover          rec;
    NeutralFitness           fit;

    using Pop = Population<DenseHaplotypes, decltype(mut), decltype(rec), decltype(fit)>;
    Pop pop(1000, 10000, mut, rec, fit, table);

    // Basic driver (backward-compatible)
    Simulation<Pop> sim(pop, /*seed=*/42);
    sim.run(100, [](Generation g, const Pop& p) {
        if ((g+1) % 10 == 0)
            std::cout << "gen " << g+1 << "  seg=" << p.segregating_sites() << "\n";
    });
}
```

### Advanced driver with hooks, selection, GC, and ancestry

```cpp
#include "genetics/gensim.hpp"

using namespace gensim;

int main() {
    AlleleTable table;
    DFEMutation          mut{1e-4, GammaDFE{-0.01, 0.3}, 0.3};
    MultiBreakpoint      rec{2.0};
    AdditiveFitness      fit;
    TournamentSelection  sel{5};   // 5-way tournament

    using Pop = Population<DenseHaplotypes, decltype(mut), decltype(rec), decltype(fit)>;
    Pop pop(500, 5000, mut, rec, fit, table);

    AdvancedSimulation<Pop, TournamentSelection> sim(pop, 42, sel);

    // Configure hooks
    SimulationEvents<Pop> events;
    events.on_generation_end = [](Generation g, const Pop& p) {
        if ((g+1) % 10 == 0)
            std::cout << "gen " << g+1 << "  w=" << p.mean_fitness() << "\n";
    };
    events.should_stop = [](Generation g, const Pop&) { return g >= 200; };

    sim.set_events(std::move(events))
       .set_gc_interval(20)
       .enable_ancestry();

    sim.run(500);  // may stop early via should_stop

    // Query ancestry
    if (auto* anc = sim.ancestry())
        std::cout << "Breeders gen 50: " << anc->effective_breeders(50) << "\n";
}
```

---

## Build

Requires a C++20-capable compiler.  No external dependencies.

### Linux / macOS (GCC ≥ 12 or Clang ≥ 15)

```bash
g++ -O3 -std=c++20 -Iinclude -o gensim src/main.cpp
```

### Windows (MSVC ≥ 2022 17.0)

```powershell
cl /O2 /std:c++20 /EHsc /Iinclude src\main.cpp /Fe:gensim.exe
```

### Windows (MinGW-w64 / g++)

```powershell
g++ -O3 -std=c++20 -Iinclude -o gensim.exe src/main.cpp
```

---

## Architecture overview

The library uses **compile-time polymorphism** throughout.  Every configurable aspect is a template parameter or a lightweight policy object — no virtual functions in the hot path.

```
┌──────────────────────────────────────────────────────────────────┐
│                         User Code                                │
│  (configure policies, build Population, run Simulation)          │
└──────────────┬───────────────────────────────────────────────────┘
               │
       ┌───────▼───────┐     ┌─────────────────┐
       │  Simulation /  │────▶│  SimulationEvents│  (hooks)
       │  Advanced      │     └─────────────────┘
       │  Simulation    │────▶ DemographicSchedule
       │                │────▶ AncestryTracker
       └───────┬───────┘
               │
       ┌───────▼───────┐
       │   Population   │     Policy objects:
       │   <Storage,    │       • MutationPolicy
       │    Mut, Rec,   │       • RecombinationPolicy
       │    Fit>        │       • FitnessPolicy
       └───────┬───────┘       • SelectionPolicy
               │
       ┌───────▼───────┐
       │    Storage     │     DenseHaplotypes
       │    Backend     │     SparseVariants
       │                │     (or custom — satisfy concepts)
       └───────┬───────┘
               │
       ┌───────▼───────┐
       │  AlleleTable   │     Shared allele registry
       └───────────────┘
```

### C++20 Concepts (`concepts.hpp`)

Every pluggable interface is formalised as a concept:

| Concept | What it constrains |
|---|---|
| `HaplotypeStorage` | Storage backends (size queries, copy_segment, swap_buffers) |
| `PointerAccessStorage` | Backends with contiguous per-haplotype memory (hap_ptr) |
| `ElementAccessStorage` | Backends with per-element get/set |
| `MutationPolicyFor<S>` | Mutation callables for a given storage |
| `RecombinationPolicyFor<S>` | Recombination callables |
| `FitnessPolicyFor<S>` | Fitness evaluation callables |
| `SelectionPolicyConcept` | Parent selection algorithms |
| `MatingSystemConcept` | Mating system parent-pairing (in `mating_system.hpp`) |
| `DFEConcept` | Selection-coefficient distributions |

### Storage backends

| Backend | Best for | Access pattern |
|---|---|---|
| `DenseHaplotypes` | Large L, high mutation rates | Pointer-based, memcpy block copies |
| `SparseVariants` | Low mutation rate, few segregating sites | Per-site binary search |
| Custom | Your scenario | Satisfy `HaplotypeStorage` (+ optionally `ElementAccessStorage`) |

Both built-in backends now satisfy both `PointerAccessStorage` (Dense) and `ElementAccessStorage` (both), so generic algorithms work with either.

### Policy catalogue

| Category | Policies | Header |
|---|---|---|
| **Mutation** | `BernoulliInfiniteAlleles`, `BernoulliFiniteAlleles`, `DFEMutation`, `MappedMutation`, `RegionalDFEMutation` | `mutation_model.hpp`, `mutation_rate_map.hpp` |
| **Recombination** | `SingleCrossover`, `MultiBreakpoint`, `MappedCrossover` | `recombination_model.hpp` |
| **Fitness** | `NeutralFitness`, `AdditiveFitness`, `MultiplicativeFitness`, `StabilizingSelectionFitness` | `fitness_model.hpp` |
| **Selection** | `FitnessProportionate`, `TournamentSelection`, `TruncationSelection`, `RankSelection`, `BoltzmannSelection` | `selection.hpp` |
| **Mating** | `RandomMating`, `SexualMating`, `SelfingMating`, `MonogamousMating`, `AssortativeMating`, `ClonalMating`, `SexualSelfingMating` | `mating_system.hpp` |
| **DFE** | `FixedDFE`, `ExponentialDFE`, `GammaDFE`, `NormalDFE`, `UniformDFE`, `MixtureDFE` | `dfe.hpp` |
| **Trait** | `AdditiveTraitModel`, `DominanceTraitModel`, `MultiTraitAdditiveModel` | `quantitative_trait.hpp` |
| **Trait fitness** | `StabilizingTraitFitness`, `DirectionalTraitFitness`, `CorrelationalTraitFitness`, `FrequencyDependentFitness` | `quantitative_trait.hpp` |
| **Output** | `write_vcf`, `write_ms`, `write_plink`, `write_eigenstrat`, `write_genotype_matrix` | `output_formats.hpp` |

### Statistics

| Statistic | Function | Header |
|---|---|---|
| Site frequency spectrum | `compute_sfs(storage)` | `statistics.hpp` |
| Nucleotide diversity (π) | `compute_pi(storage)` | `statistics.hpp` |
| Watterson's θ | `compute_theta_w(S, n)` | `statistics.hpp` |
| Tajima's D | `compute_tajimas_d(pi, theta, S, n)` | `statistics.hpp` |
| Heterozygosity | `mean_heterozygosity(storage)` | `statistics.hpp` |
| All-in-one | `compute_summary(storage)` | `statistics.hpp` |
| Linkage disequilibrium | `compute_ld(storage, i, j)` | `ld.hpp` |
| LD decay curve | `compute_ld_decay(storage)` | `ld.hpp` |
| Nei's Gst | `nei_gst(storage, demes)` | `fst.hpp` |
| Hudson's Fst | `hudson_fst(storage, deme_a, deme_b)` | `fst.hpp` |
| Pairwise Fst matrix | `pairwise_fst_matrix(storage, demes)` | `fst.hpp` |
| Inbreeding (Fis) | `individual_fis(storage, ind)` | `fst.hpp` |

### Mating systems (`mating_system.hpp`)

Control how parents are paired.  The mating system sits between selection (which scores individuals) and recombination (which produces offspring genomes).

| Policy | Description |
|---|---|
| `RandomMating` | Any two individuals (Wright–Fisher default) |
| `SexualMating` | Requires one Male + one Female parent |
| `SelfingMating` | Fraction σ of offspring from a single parent |
| `MonogamousMating` | Each individual mates with at most one partner |
| `AssortativeMating` | Mate similarity weighted by a quantitative trait |
| `ClonalMating` | Asexual: offspring = mutated copy of one parent |
| `SexualSelfingMating` | Mixed sexual + selfing with configurable rate |

```cpp
using Pop = Population<DenseHaplotypes, DFEMutation, MultiBreakpoint, AdditiveFitness>;
Pop pop(500, 5000, mut, rec, fit, table);
pop.assign_sex(0.5, rng);  // 50% male

AdvancedSimulation<Pop, FitnessProportionate, SexualMating> sim(pop, 42);
sim.run(100);
```

### Multi-chromosome genomes (`chromosome.hpp`)

Model multiple chromosomes (linkage groups) with independent assortment and sex chromosome inheritance (XX/XY, ZW, mitochondrial).

```cpp
GenomeSpec genome;
genome = GenomeSpec::uniform_autosomes(22, 5000, 1e-4);  // 22 autosomes
genome.add_sex_chromosomes(3000, 600);                     // X + Y
genome.add_mitochondrial(500);                             // mtDNA

MultiChromGenome mcg(genome, 100);  // 100 individuals
mcg.meiosis(parent, offspring, parent_sex, rng);
```

### Quantitative traits (`quantitative_trait.hpp`)

Map genotype → phenotype → fitness with the standard P = G + E model.

| Component | Description |
|---|---|
| `AdditiveTraitModel` | G = Σ allele effect sizes |
| `DominanceTraitModel` | Heterozygote vs homozygote deviations |
| `MultiTraitAdditiveModel` | Pleiotropic effects across multiple traits |
| `StabilizingTraitFitness` | w = exp(−(z−θ)²/(2Vs)) |
| `DirectionalTraitFitness` | w = 1 + β·z |
| `CorrelationalTraitFitness` | Two-trait fitness with correlation |
| `MovingOptimumSchedule` | Time-varying optimum (linear + sinusoidal) |
| `FrequencyDependentFitness` | Rare-allele advantage/disadvantage |

### Variable mutation rates (`mutation_rate_map.hpp`)

Model mutation rate heterogeneity along the genome and region-specific DFEs.

| Component | Description |
|---|---|
| `MutationRateMap` | Piecewise-constant per-site mutation rates |
| `MutationTypeMap` | Region → mutation type (neutral, coding, regulatory) |
| `MappedMutation` | Mutation policy with variable rates |
| `RegionalDFEMutation` | Different DFEs in different genomic regions |

### Output formats (`output_formats.hpp`)

Export simulation data in standard formats for downstream analysis.

| Function | Format | Tools |
|---|---|---|
| `write_vcf()` | VCF 4.2 | bcftools, PLINK, GATK |
| `write_ms()` | Hudson's ms | ms, msprime, analysis pipelines |
| `write_plink()` | PED/MAP | PLINK |
| `write_eigenstrat()` | .geno/.snp/.ind | SmartPCA, EIGENSOFT |
| `write_genotype_matrix()` | Tab-separated dosage | R, Python, custom scripts |

---

## Design choices

| Decision | Rationale |
|---|---|
| **Compile-time polymorphism (templates)** | Zero-overhead dispatch; the compiler can inline policy calls and devirtualise storage access. |
| **C++20 concepts** | Machine-readable interface contracts, better error messages, auto-validation of custom backends/policies. |
| **Double-buffered storage** | Parents are read-only while offspring are written. No aliasing issues, enables future lock-free parallel offspring production. `swap_buffers()` is O(1). |
| **Poisson-scattered mutations** | When μ·L ≪ 1 we avoid touching every site. Expected work is O(μ·L·2N) per generation. |
| **AlleleTable as external shared object** | Passed by reference, not copied. Allows multiple populations to share one table (e.g., migration models). |
| **Separate Dense / Sparse backends** | Dense is cache-friendly and uses memcpy; Sparse saves memory when few sites are polymorphic. User picks at compile time. |
| **Policy objects (not inheritance)** | Lightweight structs with `operator()`. Users compose custom policies by wrapping or inheriting; no vtable overhead. |
| **Two simulation drivers** | `Simulation` for simplicity; `AdvancedSimulation` for full feature integration. No overhead when advanced features aren't used. |
| **Event hooks (not inheritance)** | `std::function` callbacks attached to named phases. Users inject behaviour without subclassing. |

---

## Where to extend

- **Custom storage backend**: implement any type satisfying `HaplotypeStorage` (and optionally `ElementAccessStorage` / `PointerAccessStorage`). Generic statistics, GC, LD, and Fst will work automatically.
- **Custom selection**: write any callable `(const vector<Fitness>&, RNG&) → size_t` and pass it to `AdvancedSimulation`.
- **Compressed haplotype blocks**: implement a run-length or PBWT-based backend satisfying the concepts.
- **Parallel offspring production**: the loop in `Simulation::step()` is embarrassingly parallel. Give each thread its own RNG and local AlleleTable batch.
- **File I/O**: add VCF, PLINK, or tree-sequence exporters using the element-access API.
- **Non-Wright–Fisher models**: replace the selection step via events or a custom simulation loop.
- **Epistasis**: extend `FitnessPolicy` to consider allele interactions across sites.
- **Spatial models**: use `on_pre_selection_mut` hooks to implement stepping-stone or continuous-space migration.
- **Environmental fluctuation**: change selection coefficients or fitness model parameters via `on_pre_selection_mut`.

---

## Thread safety

The library is single-threaded by default.  The code is annotated with `Thread-safety note` comments at every point where parallelism is possible:

- **Fitness evaluation**: read-only, trivially parallel.
- **Offspring production**: per-individual, needs per-thread RNG.
- **Mutation**: per-haplotype, independent if `AlleleTable::new_allele()` is made atomic or thread-local.
- **AlleleTable**: append-only; protect with `std::mutex` or use a lock-free concurrent vector for parallel mutation.

---

## License

Public domain / MIT — use freely.
