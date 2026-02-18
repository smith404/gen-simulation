# Diploid Population Genetics Simulator

A header-mostly C++20 library for forward-in-time, Wright–Fisher-style simulation of diploid populations with configurable storage backends, mutation/recombination/fitness models, and a runnable example.

---

## Repository layout

```
include/genetics/
  types.hpp                  Core type aliases (AlleleID, Fitness, …)
  allele_table.hpp           Global allele registry with metadata
  dense_haplotypes.hpp       Contiguous-memory (vector) haplotype storage
  sparse_variants.hpp        Sparse variant-list haplotype storage
  mutation_model.hpp         Bernoulli infinite-alleles & finite-alleles
  recombination_model.hpp    Single-crossover & multi-breakpoint
  fitness_model.hpp          Neutral, additive, multiplicative
  population.hpp             Population container (templated on backend + policies)
  simulation.hpp             Generation-loop driver
src/
  main.cpp                   Runnable demo (Dense, Sparse, additive-fitness)
README.md                    This file
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

## Run

```bash
# Default parameters: N=1000, L=10000, G=100, mu=1e-5, seed=42
./gensim

# Custom: N=500, L=5000, G=50, mu=1e-4
./gensim 500 5000 50 1e-4

# Custom + explicit seed
./gensim 200 2000 20 1e-4 12345
```

The demo runs three simulations:
1. **Dense backend, neutral** – full-scale parameters.
2. **Sparse backend, neutral** – smaller parameters (Sparse is slower for high segregating-site counts).
3. **Additive fitness with deleterious mutations** – demonstrates selection reducing mean fitness, multi-breakpoint recombination, and custom mutation wrappers.

It also prints an illustrative per-thread RNG seeding example.

---

## Design choices

| Decision | Rationale |
|---|---|
| **Compile-time polymorphism (templates)** | Zero-overhead dispatch; the compiler can inline policy calls and devirtualise storage access. Users who prefer runtime polymorphism can wrap policies in `std::function` or add a virtual `IHaplotypeStorage` base — the API is structured to make this straightforward. |
| **Double-buffered storage** | Parents are read-only while offspring are written. No aliasing issues, enables future lock-free parallel offspring production. `swap_buffers()` is O(1) (pointer swap). |
| **Poisson-scattered mutations** | When μ·L ≪ 1 we avoid touching every site. Expected work is O(μ·L·2N) per generation instead of O(L·2N). |
| **AlleleTable as external shared object** | Passed by reference, not copied into Population. Allows multiple populations to share one table (e.g., migration models). |
| **Separate Dense / Sparse backends** | Dense is cache-friendly and uses `memcpy` block copies; Sparse saves memory when few sites are polymorphic. The user picks at compile time via the template parameter. |
| **Policy objects (not inheritance)** | Mutation, recombination, and fitness are lightweight structs with `operator()`. Users compose custom policies by wrapping or inheriting; no vtable overhead in the hot loop. |

---

## Where to extend

- **Recombination maps**: replace the uniform crossover-point distribution in `SingleCrossover` / `MultiBreakpoint` with a position-weighted distribution (e.g., read a HapMap-style genetic map).
- **Compressed haplotype blocks**: implement a run-length or PBWT-based storage backend that satisfies the same concept; useful for very large L with long identical-by-descent tracts.
- **Parallel offspring production**: the loop in `Simulation::step()` is embarrassingly parallel across individuals.  Give each thread its own `std::mt19937_64` (seeded deterministically) and a thread-local `AlleleTable` batch; merge after the loop.
- **File I/O / mmap**: add serialization for `DenseHaplotypes` (raw binary dump) and `SparseVariants` (VCF-like or custom format).  For very large populations, memory-map the storage arrays.
- **Migration / multiple demes**: hold a `std::vector<Population>` and exchange individuals between demes each generation.
- **Non-Wright–Fisher models**: replace fitness-proportionate sampling with other demographic models (e.g., exponential growth, bottlenecks, overlapping generations).
- **Dominance / epistasis**: extend `FitnessPolicy` to take both alleles at each site and model dominance coefficients.
- **Ancestry tracking / tree sequences**: record parent–offspring relationships and recombination breakpoints to build a succinct tree sequence (à la tskit).

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
