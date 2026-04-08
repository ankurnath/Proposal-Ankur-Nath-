## Feedback 1

Define the pruning problem parameterized by \epsilon. I have told you this many times. Do not write ~1. We need a precise definition.
[5:41 PM]
you related work section needs to be expanded a lot. We need to trust as a committee that you have done a serious survey of what’s out there and are not just reinventing the wheel.
[5:41 PM]There’s a whole line of work on ML for pruning
[5:42 PM]That you didn’t mention
[5:42 PM]It also gives the problem legitimacy (other people are interested)
[5:42 PM]I have also given you this comment before
[5:42 PM]Please fix the slides before the presentation tomorrow

## Feedback 2

Hi Ankur,

I took a look at your prelim slides. The content is strong — two
published papers and a clear proposed direction put you in a good
position. Here are my notes, roughly in order of importance.

Structural issues:

You need a thesis statement. Right now the three papers are presented
as separate contributions that share a topic, but there's no
overarching claim the dissertation is proving. Something like: "Ground
set pruning is a principled, general framework for scaling
combinatorial optimization, with provable guarantees, learned
adaptability, and automated generality." One sentence that ties the
whole arc together. Without it, the committee will ask, and you need a
crisp answer.

The formal problem definition on slide 5 needs to be actually formal.
"f(H(U'))/f(H(U)) approximately 1" — what does "approximately" mean?
State a proper optimization problem with an explicit approximation
parameter alpha and a clear tradeoff. Your committee (especially Choe
and Chakrabortty) will press on this.

When you state Theorems 1 and 2 (slide 13), you need at least one
proof sketch slide. The committee wants to see that you understand why
the results hold, not just what they say. The checkpoint/deletion
mechanism is the key insight — surface it.

Your references are too thin for a prelim. You're missing foundational
work: Nemhauser, Wolsey, and Fisher (1978); Feige (1998); Vondrak
(2008); Mirzasoleiman et al. (2015) on lazy/stochastic greedy; and the
Cappart et al. survey on GNNs for combinatorial optimization. The
committee expects you to demonstrate command of the field.

Technical gaps:

Gamma-weak submodularity: Theorem 1's guarantees depend on gamma, but
you never discuss what gamma looks like in practice. What is gamma for
influence maximization? For MaxCut? If gamma is very small, your
bounds become vacuous. Show typical values for the problems you
experiment on, or discuss when the assumption is tight versus loose.

The NHI assumption — "in large graphs, individual nodes rarely
dominate" — is hand-waving. What about hub nodes in power-law or
scale-free graphs? Either justify this more carefully or acknowledge
the limitation explicitly.

The H-DeepPruner reward function (slide 21) says "heuristic objective
value on pruned set" but doesn't say which heuristic or how it's
computed. This is the signal driving the entire MCTS — it can't be a
black box.

LLM2Prune section:

This is the weakest part and will get the most pushback. A few specific issues:

The claim that "LLMs encode broad domain knowledge about graph theory,
optimization, and network science" is strong and unjustified. You need
empirical evidence or an honest discussion of when this fails.

There's no convergence analysis or even intuition for why LLM-guided
search should reach good features. Even empirical runtime curves would
help.

You show downstream speedups (slide 23) but never discuss the cost of
pruning itself. How much do the GPT-4.1 API calls cost per run? How
does that compare to just running the solver on the full ground set?
The committee will ask.

The decision tree formulation (slide 29) — why a tree structure
specifically? The formalism looks precise but the motivation for that
particular structure is missing.

You're also missing complexity analysis for both H-DeepPruner and
LLM2Prune. QuickPrune has clean O(n) complexity. What's the MCTS
overhead? The LLM beam search overhead?

Minor:

The radar charts on slide 31 are hard to read, especially projected. A
simple table or grouped bar chart would be clearer.

Your timeline has the defense in August 2026 with knapsack experiments
and open-source LLM evaluation still pending. The committee will ask
whether that's realistic. Either tighten the remaining scope or have a
clear contingency plan.

Overall, you're in good shape — the hard part (doing the research) is
done. These are presentation and framing issues, and they're all
fixable. Focus on the thesis statement, the formal definition, and
shoring up the LLM2Prune section first.

Best,
Alan



Folder that contains the papers:  Papers
Use this folders as your source of information

Presentation Slide: CleanEasy.tex