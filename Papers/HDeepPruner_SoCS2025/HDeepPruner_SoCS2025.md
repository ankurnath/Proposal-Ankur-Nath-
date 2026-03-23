# Hierarchical DeepPruner: A Novel Framework for Search Space Reduction

**Authors:** Ankur Nath, Alan Kuhnle
**Venue:** Proceedings of the 18th International Symposium on Combinatorial Search (SoCS 2025)
**Code:** https://github.com/ankurnath/Hierarchical-DeepPruner

---

## Problem Setting

Combinatorial optimization (CO) problems on graphs arise in diverse domains (social networks, transportation, telecommunications, scheduling) and are typically NP-hard. Heuristics provide near-optimal solutions in polynomial time but struggle to scale with problem size. The paper addresses search space reduction: given a ground set U and an exact method OPT, find a significantly smaller U' with |U'| << |U| while preserving objective value:

f(OPT(U')) / f(OPT(U)) = 1

The key insight is that many candidates are poor choices for selection (e.g., individuals with limited social connections in influence maximization), so focusing on a smaller pool of promising candidates enables efficient application of heuristics or exact methods.

### Motivating Question

*Is it possible to develop a pruning algorithm that does not rely on domain-specific knowledge or feature engineering, yet remains practical and competitive with existing heuristics?*

---

## Method: Hierarchical DeepPruner (H-DeepPruner)

A two-stage adaptive framework combining supervised learning (GNN) and reinforcement learning (Neural MCTS):

### Stage 1: Pre-Pruning with GNN

**Purpose:** Quickly eliminate nodes unlikely to be part of the solution, focusing NMCTS computational resources on promising nodes.

**Architecture:** 2-layer Graph Convolutional Network (GCN) [Kipf and Welling, 2016]:

h_{v_i}^{(k)} = f_2^{(k)}({h_{v_i}^{(k-1)}, f_1^{(k)}({h_u^{(k-1)} : u in N(v_i)})})

- f_1^{(k)} aggregates neighboring node features
- f_2^{(k)} combines with the node's own features
- 2 layers, 16 hidden channels, ReLU activation

**Key Design Choices:**
- **Random node initialization:** h_{v_i}^{(0)} ~ U(0,1) instead of hand-crafted features. The hypothesis is that vertices in the optimal solution exhibit similar local structures, identifiable from local neighborhood features alone.
- **No problem-specific features:** Unlike COMBHelper (eigenvector centrality, Fourier Feature Mapping) and LeNSE (domain-specific features), H-DeepPruner uses no domain knowledge.
- **GCN over GraphSAGE:** Empirically, GCN (which retains full neighborhood information) outperforms GraphSAGE (which samples a fixed-size subset of neighbors).

**Training:**
- Training data: Erdos-Renyi (ER) random graphs with 2000 vertices, edge probability p = 0.1
- Budget k = 100; solve with heuristic to obtain optimal ground set
- Weighted cross-entropy loss: prevents non-solution vertices from dominating training
- Balanced sampling from solution and non-solution sets each epoch

**Inference:** Simply discard vertices the GNN predicts are unlikely to be in the solution in a single pass. The GNN can remove a large number of elements without compromising objective value.

### Stage 2: Fine-Pruning with Neural Monte Carlo Tree Search (NMCTS)

**Purpose:** Further refine the GNN-pruned ground set by considering element interdependencies. The GNN predicts membership independently per node, potentially retaining redundant elements. NMCTS constructs a solution from scratch, considering dependencies.

**MDP Formulation:**
- **State s:** The current pruned ground set (instance) + candidates for addition
- **Action:** Adding an element to the pruned ground set
- **Transition:** State updated with newly expanded pruned ground set
- **Reward:** Objective value of the heuristic solution on the pruned ground set
- **Terminal condition:** Fixed number of elements added

**MCTS with PUCT Selection:**

PUCT = Q(s,a) + c_{puct} * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))

Where:
- Q(s,a): Average reward for taking action a in state s
- P(s,a): Prior probability from the policy network
- N(s,a): Visit count for state-action pair
- c_{puct}: Exploration-exploitation tradeoff hyperparameter

**Four MCTS Phases:**
1. **Selection:** Traverse tree from root, selecting child with highest PUCT until reaching a leaf
2. **Expansion:** Add new child node with highest prior probability from policy network
3. **Simulation:** Use value network to predict expected reward (or true reward at terminal)
4. **Backpropagation:** Propagate simulation outcome up the tree, updating statistics

**Policy and Value Networks:** Share the same GCN backbone architecture as the pre-pruning GNN (2-layer GCN, 16 hidden channels).

### Progressive Widening

For large action spaces, standard MCTS becomes intractable (too many possible elements to add). Progressive widening maintains a finite list of child chance nodes, incrementally adding new children based on visitation counts:

|Number of visits(N_p)^alpha| >= |CHILDREN(N_p)|

Where alpha = 0.1 controls the growth rate. This starts with a narrow focus and expands as more information is gained.

### Progressive Deepening

An adaptive search strategy that iteratively expands the search tree based on improvements in objective value:

1. Start at root with empty subset, explore solutions up to fixed depth of 50
2. After each depth-limited search, check if objective value improved by at least 1% over previous search
3. If improved: restart search from root with current solution as new root, effectively deepening
4. If not improved: terminate (potential plateau reached)

This prevents the search tree from growing uncontrollably and focuses expansion only when progress is being made.

---

## Experimental Evaluation

### Applications (3 budget-constrained CO problems on graphs)

1. **Maximum Cover (MaxCover):** Given budget k, find S subset V maximizing f(S) = |{v | v in S or exists (u,v) in E, u in S}| where |S| <= k
2. **Maximum Cut (MaxCut):** Given budget k, find S subset V maximizing f(S) = |{(u,v) in E : v in S, u in V\S}| where |S| <= k
3. **Influence Maximization (IM):** Given budget k, probabilities p(u,v) on edges, cascade model C, find S subset V maximizing f(S) = E[sigma(S)] where |S| <= k. Uses independent cascade model (p = 0.01).

### Datasets

8 real-world graphs from SNAP (Stanford Large Network Dataset Collection), same as QuickPrune paper. Train-test edge splits follow Ireland and Montana (2022). H-DeepPruner is trained on ER graphs (not the SNAP training splits), unlike baselines which train on the actual graph data.

### Heuristics

- MaxCover and MaxCut: Standard Greedy [Nemhauser, Wolsey, and Fisher 1978]
- IM: IMM [Tang, Shi, and Xiao 2015]

### Baselines

- **GCOMB** [Manchanda et al., 2020]: Probabilistic greedy + weighted degree heuristic + Q-learning
- **COMBHelper** [Tian, Medya, and Ye 2024]: GNN with knowledge distillation + problem-specific boosting
- **LeNSE** [Ireland and Montana 2022]: Supervised GNN encoder + RL agent for subgraph navigation

### Evaluation Metrics

- **P_r (Pruning approximation ratio):** f(H(U')) / f(H(U))
- **P_g (Pruned fraction):** |U - U'| / |U|
- **C (Combined metric):** C = P_r * P_g
- **Speed-up S:** time_U / (time_{U'} + time_{prune}) — ratio of heuristic runtime on original vs. pruned ground set + pruning time

### Key Results

#### Combined Metric Performance (Table 1, Size Constraint, k=100)

H-DeepPruner achieves the **highest combined metric in 88% of all tested instances** across all three problems and all eight datasets.

**Maximum Cover highlights:**
- Facebook: C = 0.9357 (H-DeepPruner) vs 0.0649 (GCOMB) vs 0.3840 (COMBHelper) vs 0.0676 (LeNSE)
- Skitter: C = 0.9590 vs -- (GCOMB failed) vs -- (COMBHelper failed) vs 0.6832 (LeNSE)

**Maximum Cut highlights:**
- Facebook: C = 0.8789 vs 0.7723 (GCOMB) vs 0.1538 (COMBHelper) vs 0.0700 (LeNSE)
- YouTube: C = 0.9811 vs 0.5306 (GCOMB) vs 0.9705 (COMBHelper) vs 0.7797 (LeNSE)

**Influence Maximization highlights:**
- Facebook: C = 0.8744 vs 0.6942 (GCOMB) vs 0.3248 (COMBHelper) vs 0.0881 (LeNSE)
- Skitter: C = 0.9330 vs 0.8742 (GCOMB) vs -- (COMBHelper) vs 0.7667 (LeNSE)

#### Speed-up (Figure 4, MaxCover)

H-DeepPruner accelerates heuristics by up to **40x** by pruning a large fraction of the ground set:
- Slashdot: ~38x speed-up
- YouTube: ~30x speed-up
- Largest speed-ups on larger graphs where pruning has the most impact

#### Comparison with LeNSE (Leading Competitor)

H-DeepPruner achieves a **54% improvement** in combined metric over LeNSE, quantified as:
(C_{H-DeepPruner} - C_{LeNSE}) / C_{LeNSE}

Key advantages over LeNSE:
- No eigenvector centrality computation (costly for large graphs)
- No domain expertise for subgraph categorization
- No hyperparameter tuning per dataset/problem (subgraph size, embedding dimensionality)

#### Why H-DeepPruner Outperforms Others

- **vs GCOMB:** GCOMB's initial pruning relies on a handcrafted degree heuristic that doesn't generalize (prunes too little for MaxCover, too much for MaxCut)
- **vs COMBHelper:** Performs well on problems where pruned set can be large (Max Independent Set, Min Vertex Cover) but poorly when it must be small (MaxCover, IM)
- **vs LeNSE:** More balanced performance but requires expensive per-dataset tuning

### Multi-Budget Analysis (Table 2)

Ground set provided by H-DeepPruner for budget k=100 is also close-to-optimal for lower budgets (1, 10, 25, 50, 75):

| Budget | Facebook | Wiki | Deezer | Slashdot | Twitter | DBLP | YouTube | Skitter |
|--------|----------|------|--------|----------|---------|------|---------|---------|
| 1 | 1.0842 | 1.0630 | 0.9476 | 1.0573 | 1.0244 | 0.6147 | 1.0343 | 0.9800 |
| 10 | 0.9902 | 1.0142 | 0.9275 | 0.9609 | 0.9895 | 0.6587 | 1.0045 | 1.0424 |
| 50 | 0.9782 | 1.0010 | 0.9668 | 0.9756 | 0.9497 | 0.7629 | 1.0221 | 0.9620 |
| 100 | 0.9319 | 0.9681 | 0.9853 | 1.0192 | 0.9821 | 0.8517 | 0.9566 | 0.9494 |

(Values for IM problem; P_r ratios where >1 means pruned set yields better solution)

### Ablation Study (Table 3)

Both GNN and NMCTS components are essential:

| Component | MaxCover (Facebook C) | MaxCut (Facebook C) | IM (Facebook C) |
|-----------|-----------------------|---------------------|-----------------|
| GNN only | 0.7768 | 0.7651 | 0.8744 |
| H-DeepPruner (GNN+NMCTS) | 0.9357 | 0.8789 | 0.8744 |

- Removing GNN has limited impact on pruning approximation ratio but significantly improves NMCTS scalability
- NMCTS tries to prune the ground set further without sacrificing the pruning approximation ratio
- The GNN's utility is more pronounced on larger instances where only a smaller fraction can be explored

### Resource Usage (Figure 5)

- H-DeepPruner: Moderate CPU + GPU usage
- GCOMB: Lowest memory (handcrafted features, no GNN training)
- LeNSE: Minimal memory (modifies only a subgraph)
- COMBHelper: Highest combined usage (Fourier Feature Mapping + boosting)

---

## Experimental Setup

- **Hardware:** Linux server with NVIDIA RTX A6000 GPU and AMD EPYC 7713 CPU
- **Software:** PyTorch 2.4.1, Python 3.12.7

### Network Architecture
- Pre-pruning GNN: 2-layer GCN, 16 hidden channels, ReLU
- Policy and value networks of NMCTS: Same backbone as GNN

### Training
- GNN trained on Erdos-Renyi graphs (2000 vertices, p=0.1)
- Budget k = 100
- NMCTS: Progressive widening alpha = 0.1, depth limit = 50, progressive deepening threshold = 1%

---

## Key Contributions Summary

1. **Adaptive, problem-agnostic hierarchical framework** combining supervised (GNN) and reinforcement learning (NMCTS) for search space pruning on graphs
2. **First approach that requires no problem-specific feature engineering** for learning-based pruning
3. **Training on small random graphs scales to large out-of-distribution test graphs** without performance degradation
4. **Progressive widening + progressive deepening** for efficient MCTS on large action spaces
5. **Best combined metric in 88% of instances** across 3 CO problems and 8 datasets
6. **54% improvement over leading competitor (LeNSE)** in combined metric
7. **Up to 40x speed-up** of downstream heuristics through effective pruning
8. **Multi-budget generalization:** pruned ground sets remain optimal for budgets lower than training budget