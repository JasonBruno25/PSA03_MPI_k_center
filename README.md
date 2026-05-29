# PSA03_MPI_k_center

# CMDA 3634 PSA03 : MPI k‑center

**Author:** Jason Bruno Terceros  
**Course:** CMDA 3634 – Parallel Programming  
**Date:** March 2024 - April 2024

---

## 📌 Overview

This project implements a **parallel approximation algorithm** for the **k‑center problem** using **MPI** (Message Passing Interface). The k‑center problem is to choose `k` points from a given dataset of `n` points such that the **maximum distance** from any point to its nearest chosen center is minimized.

Because the brute‑force search over all \(\binom{n}{k}\) combinations is infeasible (for `n=120, k=8` there are over 840 billion possibilities), we use a **randomized approach**: each MPI rank tests `m` randomly selected sets of `k` centers and reports the best solution. The master rank (rank 0) collects the results and prints the overall best cost and centers.

The dataset consists of 120 or 128 cities in 2D. We use **point‑to‑point communication** (`MPI_Send` / `MPI_Recv`) to aggregate results. All work is done on Virginia Tech’s ARC system using Slurm batch scripts.

---

## 🧩 Problem Definition

Given `n` points \(p_1, p_2, \dots, p_n\) in 2D space, choose `k` centers from the points to minimize:

\[
\text{cost}(c_1,\dots,c_k) = \max_{1 \le i \le n} \min_{1 \le j \le k} \|p_i - c_j\|^2
\]

This is the **k‑center problem** – a facility location problem that minimizes the worst‑case distance to the nearest center.

---

## ⚙️ Implementation

### Sequential baseline
A sequential program (`mpi_kcenter.c`) is provided. It reads a dataset file, `k`, `m` (number of random sets to check), and a random seed `s`. It then uses a distance lookup table and early abort to evaluate `m` random tuples of `k` indices, keeping the best (lowest cost).

### Parallelization with MPI

We modify the code so that **each MPI rank** independently checks `m` random sets. To ensure different ranks test different tuples, we add the **rank number** to the random seed:

```c
srandom(atoi(argv[4]) + rank);   // seed = base_seed + rank
```

After all ranks finish, rank 0 collects results from the other ranks using **point‑to‑point communication:**

```c
if (rank == 0) {
    for (int src = 1; src < size; src++) {
        double rmcs;       // rank's minimal cost squared
        int centers[k];    // rank's best center indices
        MPI_Recv(&rmcs, 1, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(centers, k, MPI_INT, src, 0, MPI_COMM_WORLD, &status);
        if (rmcs < min_cost_sq) {
            min_cost_sq = rmcs;
            for (int i = 0; i < k; i++) optimal_centers[i] = centers[i];
        }
    }
} else {
    MPI_Send(&min_cost_sq, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    MPI_Send(optimal_centers, k, MPI_INT, 0, 0, MPI_COMM_WORLD);
}
```

Thus, **each non‑root rank sends exactly two messages** (cost + centers), and **rank 0 receives two messages per other rank.**

---

## 📊 Testing & Results
We ran the code on ARC using the provided batch scripts. The dataset was `cities120.dat` with `k=8`, `m=20,000,000` (20 million sets per rank), and base seed `20`. The debug script used `--nodes=2 --ntasks-per-node=4`, i.e., **8 MPI ranks**.

### Sample run output

  ```text
  wall time used = 2.7582 sec
  total number of 8-tuples checked = 160000000
  approximate minimal cost = 1.9693
  approximate optimal centers : 89 97 99 46 50 9 76 32
  ```
The Python visualization script (`plot_cities.py`) produced the following image of the optimal centers (red markers) and all data points (blue):

https://images/cities120_8.png

---

## 🔍 Conceptual Write‑Up

### 1. How many different sets of 8 centers must be checked in total? If the program can check 10 million sets per second, how many hours would it take to check all possible sets?

For n=120, k=8, the total number of combinations is 120 choose 8 ≈ 8.4 x 10^(11) (about 840 billion). At 10 million checks/sec, it would take 8.4 x 10^(11) / 10^(7) = 84,000 seconds ≈ **23.3 hours** to exhaustively search all possibilities. The randomized algorithm only explores a tiny fraction (e.g., 160 million total sets) in a few seconds.

### 2. What is the big advantage of using MPI over OpenMP?

MPI is designed for distributed memory systems (clusters). It allows programs to run on multiple nodes (machines) that do not share memory, enabling scalability far beyond a single node. OpenMP is limited to shared memory within one node. MPI also gives explicit control over communication, making it suitable for large‑scale parallel computing.

### 3. Describe three major differences between MPI and OpenMP

| Feature | MPI | OpenMP |
|---------|-----|--------|
| Memory model | Distributed memory (each process has its own address space) | Shared memory (threads share memory) |
| Communication | Explicit message passing (send/receive) | Implicit via shared variables (with synchronization) |
| Parallelism granularity | Coarse‑grained (process‑level) – developer controls data distribution | Fine‑grained (loop‑level) – easy to parallelize loops |

### 4. What do these lines in `mpi_kcenter_debug.sh` specify?
  ```bash
  #SBATCH --nodes=2
  #SBATCH --ntasks-per-node=4
  ```
  - `--nodes=2` : use 2 compute nodes
  - `--ntasks-per-node=4` : run 4 MPI processes (ranks) on each node. Total MPI ranks = 2 x 4 = 8

### 5. What does `-O3` in the `mpicc` compiler command specify? Why use it?

`-O3` enables aggressive compiler optimizations (e.g., loop unrolling, vectorization, inlining). It makes the code run faster, which is critical when searching for the optimal k‑center solution because we want to check as many random tuples as possible within a given time limit.

### 6. How many MPI ranks are used in the debug test script?

With `--nodes=2` and `--ntasks-per-node=4`, the total ranks = 8.

### 7. How many MPI ranks are used in the search test script (`mpi_kcenter_search.sh`)?

The script uses `--nodes=4` and `--ntasks-per-node=32`, so total ranks = 4 x 32 = 128.

### 8. How did you prevent each rank from printing their own optimal solution?

By adding `rank` to the random seed (`srandom(base_seed + rank)`), each rank tests a different set of tuples. Then only rank 0 collects and prints the global best. Without this, all ranks would test the same tuples (redundant) and could print identical or competing results.

### 9. Describe the message passing using `MPI_Send` and `MPI_Recv`. Which ranks are senders/receivers? What data is sent? How many messages does each rank send/receive?

- **Senders:** every rank except 0 (i.e., ranks 1 to `size-1`)
- **Receiver:** only rank 0
- **Data:** each sender sends its `min_cost_sq` (double) and its `optimal_centers` array of `k` integers
- **Number of messages:** each sender sends 2 messages (cost + centers). Rank 0 receives 2*(size-1) messages (one cost + one center array per sender)

---

## 🚀 How to Compile and Run on ARC

### 1. Copy the materials and load modules
  ```bash
  cp -r ~/cmda3634_materials/PSA03 .
  cd PSA03
  module purge
  module load gcc/12.2.0 openmpi
  ```

### 2. Compile (with optimizations)
  ```bash
  mpicc -O3 -o mpi_kcenter mpi_kcenter.c
  ```

### 3. Run the debug test (8 ranks, small dataset)
  ```bash
  sbatch mpi_kcenter_debug.sh
  ```
Output will be in `mpi_kcenter_debug.out`.

### 4. Run the search test (128 ranks, larger dataset `cities128.dat`)
  ```bash
  sbatch mpi_kcenter_search.sh
  ```

### 5. Visualize the results
After obtaining output, generate an image of the centers:
  ```bash
  python3 plot_cities.py cities128.dat < output_file > out.png
  ```

(Use the appropriate Python script from the materials repository.)

---

📁 Repository Contents
text
CMDA3634_PSA03/
- src/main.tex
- images/
  - cities120_8.png
  - cities128_8_464.png
  - cities128_8_3464.png
  - cities128_8_82464.png
- CMDA_3634_PSA03__MPI_k_center_Jason_BrunoTerceros.pdf
- README.md

---

## 🧠 Lessons Learned
- **Randomized approximation** + MPI allows us to explore billions of candidate solutions in seconds
- **Adding rank to the seed** is critical to avoid redundant work
- **Point‑to‑point communication** is simple and effective for a master‑worker pattern
- **Strong scaling** is excellent because the problem is embarrassingly parallel (each rank works independently)
- **Different random seeds** can produce different sets of optimal centers with the same minimal cost – the k‑center solution is not necessarily unique

---

## 👤 Author
**Jason Bruno Terceros** – GitHub Profile
> Course: CMDA 3634 – Comp Sci Foundations
> Virginia Tech
