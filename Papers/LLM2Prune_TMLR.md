# LLM2Prune: Using LLMs as Domain Experts for Search Space Reduction

**Authors:** Anonymous (double-blind review)
**Venue:** Under review at Transactions on Machine Learning Research (TMLR)
**Status:** Under review

---

## Problem Setting

Combinatorial optimization problems on graphs involve large discrete search spaces where many candidates contribute little due to redundancy or low value. The paper addresses the pruning problem: constructing a much smaller ground set U' with |U'| << |U| such that solving the reduced instance yields nearly the same objective value:

f(OPT(U')) / f(OPT(U)) = 1 - epsilon

### Limitations of Existing Approaches

- **Classical approaches (QuickPrune, SS):** Limited by scalability issues due to sequential processing; do not scale well to large graphs
- **Learning-based approaches (GCOMB-P, LeNSE, COMBHelper):** Rely on handcrafted, problem-specific features; lack of problem-agnostic design; weak performance on general graph optimization problems; poor adaptability to different constraints

### Motivating Question

*Is it possible to design a scalable pruning framework that avoids handcrafted features, delivers strong performance across general CO problems over graphs, and adapts easily to constraints?*

---

## Method: LLM2Prune

### Core Idea

Use LLMs to automatically generate node-level features for a lightweight GCN classifier that predicts which nodes belong in the pruned ground set. The feature discovery process is guided by an LLM-directed beam search over the feature space.

### Framework as a Decision Tree

The feature search is formalized as a decision tree T where each node n_{d,i} (depth d, index i) is represented as a tuple:

n_{d,i} = (F_{d,i}, M_{d,i}, g_{theta_{d,i}}, H_{d,i})

- **F_{d,i}:** Proposed feature set
- **M_{d,i}:** Performance metric of the trained classifier (combined metric C = P_r * P_g on held-out synthetic validation graph)
- **g_{theta_{d,i}}:** Trained classifier on this feature set
- **H_{d,i}:** Cumulative history (all past feature sets, performance metrics, feature-importance vectors, and discarded features along the root-to-node path)

### Two Phases at Each Depth

#### 1. Expansion Phase

For each node n_{d,i} in the beam B_d:
1. **Summary generation:** LLM summarizes the cumulative history H_{d,i} using meta-prompt Z_summary into key points and insights
2. **Feature proposal:** LLM generates gamma child feature sets using meta-prompt Z_feature, conditioned on:
   - Task description with constraints
   - History summary
   - Refinement rules: keep high-importance features, avoid redundant/failed ones, add features inspired by important patterns
3. **Code generation:** LLM generates executable Python code for each proposed feature using meta-prompt Z_code
   - Features must return a NumPy array with one value per node
   - Timeout threshold tau applied; failures recorded in history
4. **Classifier training:** Train a 2-layer GCN (g_theta) on each feature set with semi-supervised learning
5. **Evaluation:** Compute performance metric M and feature-importance vectors (via GNNExplainer)

The history conditioning on the full root-to-node path (not just the parent) is crucial: it prevents the LLM from re-proposing features explored and discarded at earlier depths.

#### 2. Pruning Phase

- Evaluate all candidate child nodes C_{d+1} by their performance metric M
- Select top-beta nodes to form the next beam B_{d+1}
- Continue until maximum depth D_max

### Final Output

The best node across the entire tree is selected:
n^{best} = argmax_{n_{d,i} in T} M_{d,i}

The classifier g_{theta_best} is fine-tuned on the actual target training graph and applied to prune U.

### Semi-Supervised Learning for Label Noise

Traditional approaches assume all non-solution nodes are equally unimportant, introducing label noise. LLM2Prune instead:
1. Train classifier g_theta on features F_{d+1,j}
2. After warm-up, retain high-confidence nodes: V_conf = {v in V : max_y p_theta(y|v) >= delta_hat}
3. Assign soft pseudo-labels: y_tilde_v = p_theta(y|v)
4. Continue training with pseudo-labels + ground-truth supervised labels

### Synthetic Graph Proxy

Beam search is performed on synthetic graphs (Holme-Kim random graph model, n=10000, m=4, p=0.01) that match the structural distribution of the target domain (scale-free degree distribution + high clustering). This avoids running expensive beam search on each real-world instance.

---

## Algorithm (Algorithm 1: LLM2Prune)

**Require:** Task description, ground set U, beam width beta, expansion factor gamma, max depth D_max, timeout tau, confidence threshold delta

1. Initialize root node n_{0,1} with empty feature set and history H_{0,1} = empty
2. B_0 = {n_{0,1}}
3. For d = 0, ..., D_max - 1:
   - C_{d+1} = empty
   - For each n_{d,i} in B_d:
     - summary = LLM(Z_summary, H_{d,i})
     - For j = 1, ..., gamma:
       - F_{d+1,j} ~ LLM(Z_feature, Task, summary)
       - Generate and execute code for each f in F_{d+1,j}; discard features exceeding tau
       - Train g_{theta_{d+1,j}} with semi-supervised learning
       - Compute M_{d+1,j} and Imp_{d+1,j}; update H_{d+1,j}
       - C_{d+1} = C_{d+1} union {n_{d+1,j}}
   - B_{d+1} = top-beta nodes in C_{d+1} by M
4. n^{best} = argmax M_{d,i} over all nodes in T
5. Fine-tune g_{theta_best} on target training graph
6. Return top-ranked nodes of U under g_{theta_best} as U'

---

## Experimental Evaluation

### Applications (3 graph CO problems)

1. **Maximum Coverage (MaxCov):** f(S) = |{v | v in S or exists (u,v) in E, u in S}|, subject to sum c(v) <= k
2. **Maximum Cut (MaxCut):** f(S) = |{(u,v) in E : v in S, u in V\S}|, subject to sum c(v) <= k
3. **Influence Maximization (IM):** f(S) = E[sigma(S)] under independent cascade model (p=0.01), subject to sum c(v) <= k

### Datasets

Same 8 SNAP datasets as QuickPrune (Facebook, Wiki, Deezer, Slashdot, Twitter, DBLP, YouTube, Skitter), with same train-test splits following Ireland and Montana (2022).

### Baselines

- **Classical:** QuickPrune [Nath & Kuhnle, 2025], Submodular Sparsification (SS) [Zhou et al., 2017]
- **Learning-based:** GCOMB-P [Manchanda et al., 2020], LeNSE [Ireland & Montana, 2022], COMBHelper [Tian et al., 2024]

### Heuristics Used

- MaxCover and MaxCut: Standard Greedy [Nemhauser et al., 1978]
- IM: IMM [Tang et al., 2015]
- Knapsack: Khuller et al. (1999) for IM and MaxCover; Pham et al. (2023) for MaxCut

### Evaluation Metrics

- **P_r (Pruning approximation ratio):** f(H(U')) / f(H(U))
- **P_g (Pruned fraction):** proportion of ground set removed
- **C (Combined metric):** C = P_r * P_g

### Key Results

#### Size Constraint
- LLM2Prune performs competitively with classical methods (QuickPrune, SS) on combined metric
- 1-4 orders of magnitude faster than classical approaches on large graphs (YouTube, Skitter)
- Among learning-based methods, consistently outperforms GCOMB-P, LeNSE, and COMBHelper across nearly all instances
- Achieves comparable or better performance without relying on hand-crafted features

#### Knapsack Constraint
- Only QuickPrune generalizes to knapsack among prior methods; LLM2Prune adapts by simply modifying the task description
- LLM2Prune outperforms Top-k across all applications
- Outperforms QuickPrune on MaxCov and IM; comparable on MaxCut

#### Runtime Analysis (Inference, Size Constraint, IM)

| Algorithm | Facebook | Wiki | Deezer | Slashdot | Twitter | DBLP | YouTube | Skitter |
|-----------|----------|------|--------|----------|---------|------|---------|---------|
| LLM2Prune | 0.194 | 0.203 | 0.235 | 0.257 | 0.314 | 0.458 | 1.118 | 2.420 |
| QuickPrune | 1.860 | 41.960 | 1.100 | 226.000 | 3219.040 | 222.760 | 658.320 | 5109.000 |
| SS | 24.960 | 86.424 | 11.371 | 102.259 | 1242.836 | 118.537 | 3652.793 | 10433.744 |
| LeNSE | 31.453 | 36.476 | 42.164 | 44.193 | 81.610 | 49.286 | 238.906 | 1607.605 |

LLM2Prune completes inference in under 3 seconds even on Skitter (1.7M nodes).

### Discovered Features (Non-trivial, Problem-Grounded)

A key strength is that LLM2Prune discovers features that are not generic graph statistics but are grounded in the problem objective:

- **MaxCut (size):** degree + random cut expectation (expected contribution to cut under random partition)
- **MaxCut (knapsack):** degree, weight, degree-to-weight ratio, normalized degree, neighbor weight sum, clustering coefficient
- **IM (size):** degree + average incoming activation probability (mean probability a node is activated by neighbors)
- **IM (knapsack):** degree-to-weight ratio + in-out degree difference (directional propagation asymmetry)
- **MaxCov (knapsack):** coverage per cost

These features would require significant domain expertise to design manually, yet LLM2Prune discovers them automatically from the task description alone.

### Multi-Budget Generalization

The classifier trained for budget k=100 produces pruned ground sets that remain optimal for budgets lower than 100 in nearly all instances, demonstrating budget generalization.

---

## Hyperparameters

### Beam Search
- Beam width beta = 3
- Expansion factor gamma = 2
- Max depth D = 5
- Selected via grid search; fixed across all experiments

### Training Data Generation
- Holme-Kim random graphs: n = 10,000 nodes, m = 4, p = 0.01

### LLM Settings
- GPT-5-nano and LLaMA-3.3-70B-Instruct (both achieve comparable performance)
- Default parameters from documentation

### GNN Architecture
- 2-layer GCN, 16 hidden channels, ReLU activation
- Semi-supervised: confidence threshold delta = 0.90, top k=100 most confident predictions per iteration
- 500 epochs warm-up

### Testing Settings
- Progressive expansion strategy: start with top 500 nodes, expand to 1000, 2000, 5000
- Re-evaluate heuristic after each expansion; stop when improvement < 1%

---

## Ablation Study

Four variants compared (same number of API calls):

1. **Single-shot prompt:** One LLM query, no refinement
2. **Simple feedback loop with performance:** Iterative refinement based on performance metrics only
3. **Simple feedback loop with performance + feature importance:** Adds feature-importance scores
4. **Beam search (full LLM2Prune):** Multiple candidate feature sets with beam expansion/pruning

Results show beam search consistently achieves higher and more stable performance (median values close to 1.0, minimal variance), while simpler variants suffer from error propagation in early iterations.

---

## Limitations and Future Directions

1. **Computational cost of beam search:** More expensive than a simple feedback loop; future direction is to use reasoning tokens or similar mechanisms to improve single-path reliability
2. **Sensitivity to synthetic graph proxy:** When the proxy is structurally mismatched to the target graph family (e.g., Erdos-Renyi graphs lack clustering/community structure), discovered features don't transfer well. LLM2Prune generalizes within a structural family but degrades under cross-family distribution shift.
