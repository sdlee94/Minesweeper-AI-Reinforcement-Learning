## I trained an A.I. to beat Minesweeper.. without teaching it any rules!

Of course, since this is a Reinforcement Learning project, the above statement should be a given. After all, Reinforcement Learning is all about having a machine learning model improve through trial-and-error. Below is a comparison of a model playing Minesweeper before training and after training on ___ games!

*Insert GIF*

## Table of Contents
1. [Introduction to Minesweeper](#intro)
2. [Reinforcement Learning](#RL)
3. [Deep Q Learning](#DQN)

### Introduction: The Game of Minesweeper <a name='intro'></a>

Minesweeper is a classic game of logic, dating back to 1989. The objective - click on all tiles except the ones containing mines. By clicking on tiles you reveal numbers which indicate how many mines are in the tiles around them. You progress through the game by revealing numbers and deducing where it is safe to click next.

*Insert GIF*

**Can a computer learn to beat Minesweeper?**

Given the logical rules of the game, Minesweeper can actually be solved using brute force algorithms such as a combination of if-else statements. This means that a computer solver can be obtained by explicitly programming it to take specific actions from specific states. But can a computer *learn* to play Minesweeper without these clear-cut commands? In other words, can a computer learn the logic of Minesweeper without it being aware of the rules firsthand?

This is where Reinforcement Learning comes in!

### What is Reinforcement Learning?

Reinforcement Learning (RL) is an area of machine learning that aims to train a computer to accomplish a task. The following are the key components of RL:


- **The Reward Structure**: Rather than explicit rules, we indicate to the computer what is beneficial or detrimental to performing a task by assigning rewards and/or penalties on specific conditions.
- **The Agent**: This is essentially the computer, which takes actions on the **environment** based on what it thinks will result in the highest reward / lowest penalty.
- **The Environment**: This is the game. Its state is updated every time the **agent** takes an action.
Each action is assigned a reward based on our **reward structure**. The environment's current state, action, reward and new state are collectively called a **transition**. The current state and reward are fed back to the agent so that it can learn from these experiences. By accumulating experience, the agent develops a better **policy** (*i.e.* behaviour) in performing the task at hand.

So the goal of RL is for the **Agent** to learn an optimal **policy** by pursuing actions that return the greatest reward. There are several different types of RL algorithms. In this project, I used a **Deep Q-learning Network** (DQN).

### What is a Deep Q-learning Network?

First, let's define Q-learning. In Q-learning, an agent selects an action (\\alpha)



## Neural Network Architechture
