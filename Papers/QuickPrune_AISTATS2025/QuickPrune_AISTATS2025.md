# QuickPrune: Theoretically Grounded Pruning of Large Ground Sets for Constrained, Discrete Optimization

**Authors:** Ankur Nath, Alan Kuhnle
**Venue:** AISTATS 2025 (28th International Conference on Artificial Intelligence and Statistics), PMLR Volume 258
**Code:** https://github.com/ankurnath/QuickPrune

---

## Problem Setting

The paper addresses the problem of pruning large ground sets for constrained discrete optimization. Given an objective function f defined on subsets of a ground set U, subject to a knapsack constraint (modular cost function c with budget kappa), the goal is to produce a much smaller pruned universe U' such that:

- |U'| is polylogarithmic in n = |U|
- For any budget tau in a range [kappa_min, kappa_max], the optimal value on U' retains a constant fraction of the optimal value on U

This is motivated by modern instances of combinatorial optimization where ground sets are billion-scale but optimal sets are small (e.g., viral marketing campaigns selecting a few products from millions).

### Formal Problem Definition (Pruning)

Given an objective function f: 2^U -> R+, modular cost function c: U -> R+, and budget range [kappa_min, kappa_max], produce U' subset of U such that:
- |U'| = O(F(kappa_max, kappa_min, c) polylog(n))
- There exists alpha in [0,1] such that for any tau in [kappa_min, kappa_max], f_tau(U') >= alpha * f_tau(U)

---

## Key Assumptions

1. **Gamma-weak submodularity:** The objective function f is gamma-submodular, meaning gamma is the maximum value in [0,1] such that for all S subset T subset U, for all x not in T, gamma * Delta(x|T) <= Delta(x|S). When gamma = 1, f is fully submodular. This is also called the diminishing-returns ratio.

2. **No Huge Items (NHI) Assumption:** Given an instance (f, c, kappa) with parameter eta > 0, there exists an optimal solution O such that for all o in O, c(o) <= kappa(1 - eta). This says no single element of an optimal solution consumes a very large fraction of the total budget. Satisfied when kappa >= 2 and eta <= 1/2 for size constraints.

3. **Monotonicity:** The objective function f is assumed to be monotone (f(S) <= f(T) for S subset T).

---

## Algorithm

### QuickPrune-Single (Algorithm 1) - Pruning for a Single Knapsack Constraint

For a single budget value kappa with parameters delta > 0 (size-control) and epsilon > 0 (deletion):

1. Initialize: A = empty, a* = empty, A_s = empty
2. For each element e in U:
   - Skip if c(e) > kappa (too expensive)
   - Add e to A if Delta(e|A) >= (delta * c(e) * f(A)) / kappa (marginal gain proportional to cost-to-budget ratio)
   - Track best single element a*
   - **Deletion step:** If f(A) > (n/epsilon) * f(A_s), delete old elements from A (prevents unbounded growth)
3. Return U' = A + a*

The deletion step is key: when the accumulated value grows by a large factor since the last checkpoint, old elements are discarded. Submodularity bounds the value lost from deletion.

### QuickPrune (Algorithm 2) - The Main Pruning Algorithm

Handles a range of budgets [kappa_min, kappa_max]:

1. Generate budget set B = {tau_i = kappa_max(1-eta)^i} for i such that (1-eta)*kappa_min <= tau_i <= kappa_max
2. Run log(kappa_max/kappa_min) copies of QuickPrune-Single in parallel, one for each budget tau in B
3. Each element e in U is passed to all copies
4. Return the union of all pruned sets from all copies

---

## Theoretical Results

### Theorem 1 (Main Result)

Let kappa_min < kappa_max, 0 < eta <= 1/2, f be gamma-submodular and c be modular. Suppose for all kappa' in [kappa_min, kappa_max], the NHI assumption holds with eta. Then for all kappa' in [kappa_min, kappa_max]:

- **Value retention:** f_kappa'(U') >= alpha * f_kappa'(U), where alpha = (delta * gamma^4 * (1 - epsilon * gamma^{-1})) / (6 * (delta * gamma^2 + 1) * (1 + gamma^{-1} * delta))
- **Size bound:** |U'| = O(log(kappa_max/kappa_min) * (1 + kappa_max/(delta * c_min)) * log(n/epsilon))

When gamma = 1 (submodular) and delta = 1, the value retention becomes approximately (1/24 - epsilon).

### Theorem 2 (Single Budget)

For a single budget instance with Algorithm 1:
- |U'| < 2(1 + kappa/(delta * c_min)) * log(n/epsilon) + 3
- There exists A' subset U' with c(A') <= kappa and f(A') >= (delta * gamma^4) / (2(delta * gamma^2 + 1)) * f(A_m)

### Key Properties
- **Query complexity:** O(log(kappa_max/kappa_min)) queries to f per element
- **Single-pass:** Each element is processed once and never reconsidered
- **Pruned set size:** Logarithmic in n (ground set size)

---

## Experimental Evaluation

### Applications (4 graph CO problems)

1. **Maximum Cover (MaxCover):** Find S subset V maximizing f(S) = |{v | v in S or exists (u,v) in E, u in S}| subject to sum c(v) <= k
2. **Maximum Cut (MaxCut):** Find S subset V maximizing f(S) = |{(u,v) in E : v in S, u in V\S}| subject to sum c(v) <= k
3. **Influence Maximization (IM):** Under independent cascade model with p=0.01, find S maximizing expected spread f(S) = E[sigma(S)] subject to sum c(v) <= k
4. **Information Retrieval:** Image and video retrieval using graph cut function with similarity kernel

### Datasets

8 real-world graphs from SNAP (Stanford Large Network Dataset Collection):

| Graph | Vertices (Train) | Edges (Train) | Vertices (Test) | Edges (Test) |
|-------|-----------------|---------------|-----------------|-------------|
| Facebook | 3,847 | 26,470 (30%) | 4,002 | 61,764 |
| Wiki | 4,891 | 30,228 (30%) | 6,358 | 70,534 |
| Deezer | 48,870 | 149,460 (30%) | 53,511 | 348,742 |
| SlashDot | 47,546 | 140,566 (30%) | 67,640 | 327,988 |
| Twitter | 55,827 | 134,229 (10%) | 80,712 | 1,208,067 |
| DBLP | 63,004 | 41,994 (10%) | 315,305 | 1,007,872 |
| YouTube | 185,193 | 179,257 (6%) | 1,098,104 | 2,808,367 |
| Skitter | 147,604 | 110,952 (1%) | 1,694,318 | 10,984,346 |

### Evaluation Metrics

- **Pruning approximation ratio (P_r):** f(H(U')) / f(H(U)) — ratio of heuristic solution quality on pruned vs. original ground set (higher is better)
- **Pruned fraction (P_g):** |U - U'| / |U| — fraction of ground set pruned (higher is better)
- **Combined metric (C):** C = P_r * P_g — balances both metrics

### Baselines

- **Classical:** Submodular Sparsification (SS) [Zhou et al., 2017]
- **Learning-based:** GCOMB-P [Manchanda et al., 2020], LeNSE [Ireland and Montana, 2022], COMBHelper [Tian et al., 2024]
- **Proposed baseline:** GNNPruner (simplified GCN-based version of COMBHelper with random features)

### Key Results

#### Size Constraint (budget k = 100)
- QuickPrune achieves highest combined metric on majority of instances
- Prunes ground set by orders of magnitude more than other methods while maintaining similar approximation ratios
- SS requires 30x more oracle queries than QuickPrune
- GNNPruner (proposed simple baseline) outperforms all other ML methods

#### Knapsack Constraint
- Cost model: c(v) = (beta/|V|)(|N(v)| - alpha) with alpha = 1/20, beta normalized so c(v) >= 1
- QuickPrune achieves highest combined metric on 45% of instances
- Often outperforms Top-k by 20% in retaining objective value for MaxCover and MaxCut

#### Runtime and Resources
- QuickPrune uses only CPU resources (no GPU needed)
- Less memory than LeNSE's CPU usage alone
- Runtime on MaxCover: 0.064s (Facebook) to 10.317s (Skitter) vs. SS: 3.178s to 2238s

### Hyperparameters
- delta = 0.1, epsilon = 0.1, eta = 0.5 for graph problems
- delta = 0.5 for knapsack-constrained MaxCover
- delta = 0.05, epsilon = 0.1, eta = 0.5 for image/video retrieval

---

## GNNPruner Baseline

A modified version of COMBHelper proposed as a stronger baseline:
- Uses GCN (not SAGEConv) with fewer layers
- Random numbers as node features (no domain knowledge)
- Eliminates need for hand-crafted features and extensive feature engineering
- Outperforms GCOMB-P, LeNSE, and COMBHelper on ML-based pruning
- Architecture: 2-layer GCN, 16 hidden channels, ReLU activation, Adam optimizer (lr=0.001, weight decay 5e-4)

---

## Future Directions

1. Improving the theoretical ratio alpha (currently constant but small)
2. Upper bounds on the pruning problem to understand its complexity
3. Extending the framework to handle more than one objective function
4. Handling deletions to the ground set (dynamic setting)