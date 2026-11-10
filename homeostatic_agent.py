"""
Homeostatic vs reward-maximising agents in a small gridworld.

Question: do agents driven by internal needs (energy, water) behave more
adaptively, and more robustly, than agents trained on hand-designed rewards?

Three agents (same learning algorithm, only the reward differs):
  homeostatic : reward = reduction in "drive" (distance of needs from set point)
  external    : +1 for eating/drinking, -1 for toxin (simple, exploitable)
  gated       : like external, but +1 only if the agent was actually hungry/thirsty
                (a fairer, smarter hand-designed baseline)

Run:  python homeostatic_agent.py       
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Settings ----------------
SIZE = 8
FOOD, WATER = (1, 1), (6, 6)
TOXINS = {(3, 3), (4, 4), (3, 4), (4, 3)}
MAX_LEVEL, MAX_STEPS = 10.0, 300
ENERGY_DECAY, WATER_DECAY = 0.10, 0.15
FOOD_GAIN, WATER_GAIN, TOXIN_DMG = 6.0, 6.0, 3.0
DEATH_PENALTY = -5.0
HUNGRY_BELOW = 6.0                      # used only by the "gated" reward
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]   # up, down, left, right, stay
REWARDS = ["homeostatic", "external", "gated"]


# ---------------- Environment ----------------
class GridWorld:
    def __init__(self, reward_type, rng, food=FOOD, water=WATER, decay_mult=1.0):
        self.reward_type, self.rng = reward_type, rng
        self.food, self.water_pos, self.decay_mult = food, water, decay_mult

    def reset(self):
        self.pos = (int(self.rng.integers(SIZE)), int(self.rng.integers(SIZE)))
        self.energy, self.water, self.t = 7.0, 7.0, 0
        return self.state()

    def state(self):
        # Internal levels are cut into 6 buckets (0..5) so a lookup table can store them.
        return (self.pos[0], self.pos[1],
                min(int(self.energy // 2), 5), min(int(self.water // 2), 5))

    @staticmethod
    def drive(e, w):
        """How far the body is from the ideal state (both levels = 10). 0 = perfect."""
        return ((MAX_LEVEL - e) / MAX_LEVEL) ** 2 + ((MAX_LEVEL - w) / MAX_LEVEL) ** 2

    def step(self, a):
        d_before = self.drive(self.energy, self.water)
        was_hungry, was_thirsty = self.energy < HUNGRY_BELOW, self.water < HUNGRY_BELOW
        dx, dy = ACTIONS[a]
        self.pos = (min(max(self.pos[0] + dx, 0), SIZE - 1),
                    min(max(self.pos[1] + dy, 0), SIZE - 1))
        self.energy -= ENERGY_DECAY * self.decay_mult      # bodies burn resources
        self.water -= WATER_DECAY * self.decay_mult
        ate = drank = poisoned = False
        if self.pos == self.food and a != 4:               # arriving on a cell (not "stay") uses it
            self.energy, ate = min(self.energy + FOOD_GAIN, MAX_LEVEL), True
        if self.pos == self.water_pos and a != 4:
            self.water, drank = min(self.water + WATER_GAIN, MAX_LEVEL), True
        if self.pos in TOXINS:
            self.energy, poisoned = self.energy - TOXIN_DMG, True
        self.t += 1
        dead = self.energy <= 0 or self.water <= 0
        done = dead or self.t >= MAX_STEPS

        if self.reward_type == "homeostatic":
            r = d_before - self.drive(max(self.energy, 0), max(self.water, 0))
        elif self.reward_type == "external":
            r = float(ate) + float(drank) - float(poisoned)
        else:  # gated
            r = float(ate and was_hungry) + float(drank and was_thirsty) - float(poisoned)
        if dead:
            r += DEATH_PENALTY
        return self.state(), r, done, dict(poisoned=poisoned, dead=dead)


# ---------------- Learning (tabular Q-learning) ----------------
def train(reward_type, seed, episodes=12000, alpha=0.1, gamma=0.97):
    rng = np.random.default_rng(seed)
    env = GridWorld(reward_type, rng)
    Q = np.zeros((SIZE, SIZE, 6, 6, len(ACTIONS)))        # Q[state][action] = expected future reward
    curve = []
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.6 * episodes))      # explore a lot early, little later
        s = env.reset()
        while True:
            a = int(rng.integers(len(ACTIONS))) if rng.random() < eps else int(np.argmax(Q[s]))
            s2, r, done, info = env.step(a)
            target = r if info["dead"] else r + gamma * np.max(Q[s2]) * (not done)
            Q[s][a] += alpha * (target - Q[s][a])         # nudge estimate towards what happened
            s = s2
            if done:
                break
        curve.append(env.t)
    return Q, np.array(curve)


def evaluate(Q, seed, episodes=200, record=False, **env_kwargs):
    """Run the trained agent (greedy, no exploration). Always measured with 'survival steps'."""
    rng = np.random.default_rng(seed + 999)
    env = GridWorld("external", rng, **env_kwargs)        # reward type is irrelevant when evaluating
    surv, traj = [], None
    for i in range(episodes):
        s, hist = env.reset(), []
        while True:
            s, _, done, _ = env.step(int(np.argmax(Q[s])))
            hist.append((env.energy, env.water))
            if done:
                break
        surv.append(env.t)
        if record and i == 0:
            traj = np.array(hist)
    return np.mean(surv), traj


def smooth(x, k=200):
    return np.convolve(x, np.ones(k) / k, mode="valid")


# ---------------- Experiment ----------------
if __name__ == "__main__":
    SEEDS = [0, 1, 2]
    TESTS = {                                             # worlds the agents were NOT trained in
        "Same world": {},
        "Faster decay (x1.5)": {"decay_mult": 1.5},
        "Food moved": {"food": (1, 6)},
    }
    scores = {rt: {t: [] for t in TESTS} for rt in REWARDS}
    curves, trajs = {}, {}
    for rt in REWARDS:
        cs = []
        for sd in SEEDS:
            Q, c = train(rt, sd)
            cs.append(c)
            for t, kw in TESTS.items():
                m, traj = evaluate(Q, sd, record=(sd == 0 and t == "Same world"), **kw)
                scores[rt][t].append(m)
                if traj is not None:
                    trajs[rt] = traj
        curves[rt] = np.mean(cs, axis=0)

    print(f"{'agent':<13}" + "".join(f"{t:>22}" for t in TESTS))
    for rt in REWARDS:
        print(f"{rt:<13}" + "".join(f"{np.mean(scores[rt][t]):>14.1f}±{np.std(scores[rt][t]):<5.1f}" for t in TESTS))

    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    for rt in REWARDS:
        ax[0, 0].plot(smooth(curves[rt]), label=rt)
    ax[0, 0].set(title="Learning curve", xlabel="Episode", ylabel="Steps survived"); ax[0, 0].legend()
    w = 0.25
    for i, rt in enumerate(REWARDS):
        ax[0, 1].bar(np.arange(len(TESTS)) + i * w, [np.mean(scores[rt][t]) for t in TESTS], w,
                     yerr=[np.std(scores[rt][t]) for t in TESTS], label=rt)
    ax[0, 1].set_xticks(np.arange(len(TESTS)) + w); ax[0, 1].set_xticklabels(list(TESTS), fontsize=8)
    ax[0, 1].set(title="Robustness: survival in changed worlds", ylabel="Steps survived"); ax[0, 1].legend()
    ax[0, 2].axis("off")
    for k, rt in enumerate(REWARDS):
        a = ax[1, k]
        a.plot(trajs[rt][:, 0], label="energy"); a.plot(trajs[rt][:, 1], label="water")
        a.set(title=f"Internal states: {rt}", xlabel="Step", ylabel="Level", ylim=(0, 10.5)); a.legend()
    plt.tight_layout(); plt.savefig("results.png", dpi=150)
    print("Saved results.png")