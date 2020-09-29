## I trained an A.I. to beat Minesweeper.. without teaching it any rules!

Of course, since this is a Reinforcement Learning project, the above statement should be a given. After all, Reinforcement Learning is all about having a machine learning model improve through trial-and-error. Below is a comparison of a model playing Minesweeper before training and after training on ~half a million games!

<p align="center">
  <img src="https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning/blob/master/before_after_train.gif"/>
</p>  

## Table of Contents
1. [Introduction to Minesweeper](#intro)
2. [Reinforcement Learning](#RL)
3. [Deep Q Learning](#DQN)

### Introduction: The Game of Minesweeper <a name='intro'></a>

Minesweeper is a classic game of logic, dating back to 1989. The objective - click on all tiles except the ones containing mines. By clicking on tiles you reveal numbers which indicate how many mines are in the tiles around them. You progress through the game by revealing numbers and deducing where it is safe to click next.

**Can a computer learn to beat Minesweeper?**

Given the logical rules of the game, Minesweeper can actually be solved using brute force algorithms such as a combination of if-else statements. This means that a computer solver can be obtained by explicitly programming it to take specific actions from specific states. But can a computer *learn* to play Minesweeper without these clear-cut commands? In other words, can a computer learn the logic of Minesweeper without it being aware of the rules firsthand?

This is where Reinforcement Learning comes in!

### What is Reinforcement Learning? <a name='RL'></a>

Reinforcement Learning (RL) is an area of machine learning that aims to train a computer to accomplish a task. The following are the key components of RL:


- **The Reward Structure**: Rather than explicit rules, we indicate to the computer what is beneficial or detrimental to performing a task by assigning rewards and/or penalties on specific conditions.
- **The Agent**: This is essentially the computer, which takes actions on the **environment** based on what it thinks will result the highest reward / lowest penalty.
- **The Environment**: This is the game. Its state is updated every time the **agent** takes an action. Each action is assigned a reward based on our **reward structure**. The environment's current state, action, reward and new state are collectively called a **transition**. The current state and reward are fed back to the agent so that it can learn from these experiences. By accumulating experience, the agent develops a better **policy** (*i.e.* behaviour) in performing the task at hand.

So the goal of RL is for the **Agent** to learn an optimal **policy** by pursuing actions that return the greatest reward. There are several different types of RL algorithms. In this project, I used a **Deep Q-learning Network** (DQN).

### What is a Deep Q-learning Network?

First, let's define Q-learning. Q-learning involves having a reference table of Q-values with all possible states as rows and all possible actions as columns. Actions are chosen based on the maximum quality-value ( **Q** ) for all possible actions in a given state ( **s** ). Q-values are initialized randomly (typically at 0) and are updated as the agent plays the game and observes rewards for its actions. Note that Q-values have no inherent meaning outside of the context of a specific Q-learning problem - they simply serve to compare the value of actions relative to each other.

However, Q-learning becomes unfeasible if the number of states and actions are large - the Q-table would quickly hit a memory limit! Bypassing this limitation is a combination of Deep Learning and Q-learning (aptly named Deep Q-learning) which uses a neural network to approximate the Q-function.

Speaking of which, the Q-function is the core algorithm of Q-learning, and is derived from the **Bellman Equation**:


<p align='center'>
  <img src='bellman.png' width='500'/>
</p>


Put simply, the updated Q-value is the immediate reward (r) plus the highest possible Q-value of the next state, multiplied by a **Discount Factor** ( **γ** ). The **Discount Factor** ranges from 0 to 1 and indicates how much weight is given to future rewards: a value closer to 1 places more weight to future rewards while a value closer to 0 places less weight (more discount). In other words, **Discount Factor** is a hyperparameter that controls how much your agent pursues immediate rewards vs. future rewards. Say a game has the option to fight Wario with a prize of 10,829 coins (a very high reward) for beating him. However, fighting Wario requires you to take damage (which gives negative reward). With **γ** set close to 0, your agent may learn to avoid fighting Wario altogether, as the reward for beating him is heavily discounted and thus not worth the damage it would take to fight him. With **γ** close to 1, your agent may opt to fight Wario since it values the reward for beating him very highly despite the damage required to do so.

> Sidenote: In Minesweeper, the discount factor is not so important since every action that does not reveal a mine, be it sooner or later, has equal value in progressing towards the end goal: solving the board. In fact, [Hansen and others (2017)](https://github.com/jakejhansen/minesweeper_solver/blob/master/article.pdf) showed that γ=0 resulted in higher win-rates than with γ=0.99 for their Q-learning Minesweeper implementation.

But wait! If an agent starts with no experience, and is always choosing the action that returns the highest reward, it would never learn to beat Wario in the first place, right? Right! Since the agent begins without the experience of beating Wario, it does not know about the juicy 10,829 coins and will thus learn to avoid the negative reward taken from fighting him. For an agent to find new valuable policies, it needs to explore new actions. This is where the second hyperparameter: epsilon ( **ε** ) comes in.

**ε** is the probability of exploring (acting randomly) vs. exploiting (acting based on maximum Q). If **ε** is 0.9, your agent will act randomly 90% of the time and exploit prior knowledge (use maximum Q) 10% of the time. Typically, **ε** is set to be high (>=0.9) at the start and decayed to a lower value as training progresses. This allows your agent to sufficiently explore and generate experience before gradually transitioning to a policy based on exploitation.

```python
epsilon = 0.9
rand = np.random.random() # random value b/w 0 & 1

if rand < epsilon:
    move = np.random.choice(actions)
else:
    moves = model.predict(state) # model is your neural Network
    move = np.argmax(moves)
```

Let's watch Q-learning at work with the example below. We start with a Q-table with all Q-values initialized at 0. We can see that

At a given time **t**, the agent selects an action ( **α<sub>t</sub>** ), gets a reward ( **r<sub>t</sub>** ) and the state is updated ( **s<sub>t</sub>** --> **s<sub>t+1</sub>** ).

You could think of the boxed portion of the equation as the target variable.

```python
# model is your neural Network,
# done is a boolean that is True if the game is at a terminal state
discount = 0.9
if not done:
  new_q = reward + discount * np.max(model.predict(state))
else:
  new_q = reward
```


## Neural Network Architechture
