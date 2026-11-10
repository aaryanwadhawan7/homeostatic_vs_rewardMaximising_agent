# Homeostatic vs Reward-Maximising Agents (Gridworld)

Small study: do agents driven by internal needs behave more adaptively and robustly than agents trained on hand-designed rewards?

## Setup
- 8x8 grid: one food cell, one water cell, four toxin cells in the centre.
- Internal variables: energy and water (0-10). Both decay each step; the agent dies if either reaches 0.
- Tabular Q-learning (no deep learning), 12,000 training episodes, 3 seeds. Only the reward differs between agents.
- **Homeostatic:** reward = reduction in drive (distance of energy and water from their set point).
- **External:** +1 for eating or drinking, -1 for toxin.
- **Gated:** like external, but +1 only if the agent was actually hungry or thirsty (a fairer hand-designed baseline).
- Robustness test: agents trained in one world are evaluated, without retraining, in (a) the same world, (b) 1.5x faster resource decay, (c) food moved to a new cell.

## Run
    pip install numpy matplotlib
    python homeostatic_agent.py

## Results (mean survival steps ± std over 3 seeds, cap 300)
| Agent | Same world | Faster decay | Food moved |
|---|---|---|---|
| Homeostatic | 298.3 ± 2.4 | 298.2 ± 2.5 | 62.9 ± 1.6 |
| External | 59.7 ± 0.5 | 39.9 ± 0.3 | 59.1 ± 0.4 |
| Gated | 258.0 ± 59.4 | 219.9 ± 113.3 | 63.5 ± 3.0 |

- The homeostatic agent learns reliably and stays robust to faster decay.
- The external agent exploits its reward (it repeatedly re-enters the water cell) and starves.
- The gated baseline learns, but is much less stable across seeds (high variance).
- All agents fail when the food moves: a tabular agent memorises locations, so this is a limitation of the method, not evidence for either reward.

## Limitations
- Toy environment, 3 seeds, tabular agents, no neural analysis.
- The external reward is simple; the gated one is only one of many possible designs.
- Toxins are avoided by all agents, so they are not a distinguishing result here.

## Possible extensions
Neural-network agents that can generalise, multiple competing drives, richer physics-based bodies, ablation of network components.
