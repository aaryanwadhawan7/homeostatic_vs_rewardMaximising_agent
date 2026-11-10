# Code walkthrough (homeostatic_agent.py)

## Big picture
A creature lives on a grid. Energy and water fall every step. It must walk to food and water. We train three creatures with the same learning method but different "reward" rules, then compare how long each survives.

## 1. Settings (top of file)
- SIZE = 8: the grid is 8x8.
- FOOD, WATER: grid cells (x, y) where eating and drinking happen.
- TOXINS: four cells in the middle that hurt (energy -3).
- MAX_LEVEL = 10: the maximum energy or water.
- MAX_STEPS = 300: an episode ends after 300 steps (a success).
- ENERGY_DECAY, WATER_DECAY: how much each need drops per step.
- FOOD_GAIN, WATER_GAIN: how much a meal or drink restores.
- DEATH_PENALTY: extra negative reward when a need hits 0.
- HUNGRY_BELOW = 6: "hungry" threshold, used only by the gated reward.
- ACTIONS: the 5 moves (up, down, left, right, stay) as (dx, dy).
- REWARDS: names of the three agents.

## 2. GridWorld class (the world)
- __init__: stores which reward rule to use, a random generator, and the food/water positions.
- reset(): starts a new episode at a random cell with energy = water = 7.
- state(): what the agent "sees": its x, y, and its energy and water cut into 6 buckets (levels 0 to 5). Buckets keep the lookup table small.
- drive(e, w): how far the body is from perfect (both = 10). 0 means perfect; larger means worse.
- step(a), line by line:
  1. d_before: the drive before moving.
  2. was_hungry / was_thirsty: needed only by the gated reward.
  3. Move by the chosen action, staying inside the grid.
  4. Energy and water decay.
  5. If on food (and not "stay"), energy goes up. Same for water.
  6. If on a toxin cell, energy goes down.
  7. Count the step. dead = a need reached 0. done = dead or 300 steps.
  8. Reward, depending on the rule:
     - homeostatic: drive before minus drive after (positive if needs improved);
     - external: +1 for eating, +1 for drinking, -1 for toxin;
     - gated: same, but +1 only if it was hungry or thirsty.
  9. Add the death penalty if dead. Return the new state, reward, done flag, and info.

## 3. train() (the learning)
- Q is a table: for each state and action, a number saying "how good is this action here?". It starts at 0.
- Each episode: eps (exploration rate) starts at 1 (random moves) and falls to 0.05.
- Each step: with probability eps pick a random action, otherwise the best action in the table.
- After acting, compute target = reward + 0.97 * best future value (just reward if dead).
- Q[s][a] moves 10% (alpha = 0.1) of the way towards the target. This is Q-learning.
- curve records how many steps the agent survived in each episode.

## 4. evaluate()
Runs the trained agent greedily (no random moves) for 200 episodes and returns the average survival time. It can also record energy and water over time for plotting. It accepts changed-world settings (decay_mult, food) for the robustness test.

## 5. Experiment (bottom of file)
- Trains each of the 3 agents with 3 random seeds.
- Tests each in 3 worlds: same, faster decay, food moved.
- Prints a table and saves results.png (learning curves, robustness bars, internal-state traces).

## How to read results.png
- Top left: learning curves. Higher means better survival.
- Top middle: survival in changed worlds.
- Bottom row: energy and water over one episode. A healthy agent shows regular sawtooth refills of both.
