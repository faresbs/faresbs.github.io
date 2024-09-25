---
layout: post
title: Hands on RL algorithms
subtitle: 
cover-img: assets/img/posts/2024-07-11/cover.JPG
thumbnail-img: 
share-img: assets/img/posts/2024-07-11/cover.JPG
tags: [AI, ML, Reinforcement Learning]
author: Fares Ben Slimane
date:   2024-07-11
---

This notebook is a tutorial to explain and showcase how to use RL algorithms like Q-learning (model-free version and the DQN version), Sarsa, MC, and how they differ, using PyTorch and the OpenAI Gymnasium library. This notebook will give you a straightforward overview of how RL algorithms work with real examples. I deliberately made the code redundant to showcase the differences and similarities of the different algorithms.

##### Requirements :

*   Pytorch: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
*   Gymnasium: [https://gymnasium.farama.org/](https://gymnasium.farama.org/)

## Sections

1. **Algorithm overview**

2. **Q learning**

3. **Double Q Learning**

4. **Monte Carlo**

5. **Sarsa**

6. **Deep Q Network**

7. **Experimenting with other Games from the library**


## Cliff walking:  Task Definition 

<center>
   
<img src="/assets/img/posts/2024-07-11/clifwalking.gif">

</center>
   
The agent has to decide between 4 actions: Right, Left, Bottom, or Up.

As the agent observes the current state of the environment and chooses an action, the environment transitions to a new state and returns a reward that indicates the consequences of the action.

In this task, each time step incurs a -1 reward, unless the player steps into the cliff, which incurs a -100 reward. The goal is to get to the target point with more rewards.
The space is represented by a 12 x 4 grid (48 observations) with 3 x 12 + 1 possible states. The agent observes the position from the environment and chooses an action from 4 possible actions.

You can learn more about it here:
https://gymnasium.farama.org/environments/toy_text/cliff_walking/

In this Notebook, we are going to experiment with multiple RL algorithms:

1) Q-learning: (A) model-free Q-learning (the classical one) which uses a Q-table to predict the expected value, (B) Double Q-learning to fix the problem of maximization bias, and (C) Deep Q Network, which uses a network to predict the expected value for each action, given the input state. The action with the highest expected value is then chosen. 

2) Monte Carlo

3) Sarsa

## Q-learning (Tabular method): Off-policy TD Algorithm

Q-learning is a model-free, value (Reward) based, and off-policy RL algorithm. The “Q” stands for <i>quality</i>, representing how valuable an action is in maximizing future rewards for a given state. It aims to maximize the value function Q, which estimates the expected cumulative reward.

Iteratively, the agent adjusts its strategy over time to make optimal decisions in various situations. The algorithm uses a Q-table to find the best action for each state, maximizing expected rewards.

Q-learning computes the difference between the current Q-value and the maximum Q-value of the time step over all possible actions:

$$ Q(s_t, a_t) = Q(s_t, a_t) + \alpha \left( R_{t+1} + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \right) $$

Q-learning updates the Q-value using the maximum Q-value over all possible actions for the next state, which helps it focus on the best possible action every step.

It explores aggressively, even if the current policy is different, aiming to learn the optimal policy <i>independently of the agent’s actions</i>. And this is why it is an <b>off-policy algorithm</b>.

### How Does it Work?

At each step:
You’re in a specific state (S) (a maze cell).
You choose an action (A) (e.g., move left, right, up, or down).
Based on that action, you receive a reward (cheese or poison).
You end up in the next state (S1) (a neighbouring cell).
You then decide on another action (A1) for the next step.

### Algorithm 

Parameters: step size $\alpha = (0, 1]$, small greedy $\epsilon > 0$
Initialize Q(s, a), for s representing all states and a representing all possible actions, arbitrarily except that Q(terminal, ) = max reward value.

1. **Initialize** Q-values for all state-action pairs (Q-table).
2. **Loop over episodes**:
   - Initialize the current state: $S = S_0$.
   - **Loop over each step of the current episode**:
     - Choose an action $a$ from state $S$ using a policy derived from Q (e.g., $\epsilon-greedy$ strategy).
     - Take action $a$, observe the reward $R_t$ and the next state $S_{t+1}$.
     - Update the Q-table with the observed reward and next state values.
     - Compute the **TD error**:
       
       $$ error = (R_{t+1} + \gamma \max_a Q(S_{t+1}, a)) - Q(S_t, a_t) $$
       
     - Update the Q-value using the step size $\alpha$ and the calculated TD error:
       
      $$ Q(S_t, a_t) = Q(S_t, a_t) + \alpha \cdot error $$
     
     - Transition to the next step: $S_t \leftarrow S_{t+1}$.
   - **Repeat until the current state is terminal** (i.e., $S_t = S_{\text{end}}$).


### Q-learning Python Code

```python

alpha = 0.1  # Learning rate (make sure to choose a high lr)
gamma = 0.98  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995  # Decay rate for exploration rate
min_epsilon = 0.01  # Minimum exploration rate
episodes = 600  # Number of episodes
#saving_episodes = 100

# Initialize Q-table
#Shape = (32, 4)
Q_table = np.zeros((n_observations, n_actions))

episode_durations = []

for episode in range(0, episodes):

    state, _ = env.reset()  # Get initial state
    total_reward = 0
    
    #for step in range(max_steps):
    for t in count():

        # Epsilon-greedy action selection for an initial state
        # Note: make sure to do this exploit/explore inside the time step loop to make sure we are doing this on every step
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(0, n_actions)
        else:
            action = np.argmax(Q_table[state])
            # or action = best_next_action

        # Take action and observe the result
        next_state, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        
        #Always pick the next best action using the current policy (and not the target policy)
        best_next_action = np.argmax(Q_table[next_state])

        # Update Q-value using the Bellman equation
        Q_table[state][action] += alpha * (reward + gamma * Q_table[next_state][best_next_action] - Q_table[state][action])
        
        state = next_state  # Move to the next state

        if terminated:
            #We can also use t for total duration instead of total reward
            episode_durations.append(t+1)
            #episode_durations.append(total_reward)
            if t % 10 == 0:
                plot_durations()
            break

    # Decay epsilon for exploration-exploitation balance
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

print("Training finished.\n")

env.close()
```

## Maximization Bias problem

Because of the max operation in the update equation, Q-learning sometimes tends to overestimate Q-values, leading to suboptimal policies by following a biased estimation. You can think of it as the agent always starting by turning right first. After many episodes, it can start to learn to overcome this biased approach and convert to a different and better strategy.
This overestimation problem is known as the <b>maximization bias</b>.


### Solution: Double Q-learning

Double Q-learning introduces a subtle improvement to mitigate the maximization bias.
Instead of relying on a single Q-value estimate, it maintains two separate Q-value functions: Q1 and Q2.
During updates, one function (e.g., Q1) selects the best action, while the other (Q2) evaluates that action.
The update equation alternates between Q1 and Q2, reducing the overestimation effect.
The final policy is derived from the average or sum of Q1 and Q2.

### Advantages

Double Q-learning provides faster transient performance compared to standard Q-learning.
It reduces the maximization bias, leading to better policies over a shorter period of training time.

<center>

![Q-learning camparison](/assets/img/posts/2024-07-11/q_learning_vs_double.png)

</center>

### Monte Carlo Python Code

Unlike Q-learning, through the Monte Carlo method, you can observe every state in an episode and calculate the total reward received from the current state to the end.
In the Monte Carlo method, we focus on episodes to calculate the value of each state.

This approach is distinct from Q-learning, which updates Q-values at each time step. The Monte Carlo method relies on sampling episodes to derive the expected returns.

```python
    # Calculate returns and update Q-table
    G = 0
    cum_rewards = 0
    cum_count = 0

    for t in reversed(range(len(reward_trajectory))):

        reward = reward_trajectory[t]
        action = action_trajectory[t]
        state = state_trajectory[t]

        G = gamma * G + reward

        # first-visit MC method
        #You can use to average the total cum rewards for better smoother update
        Q_table[state][action] += alpha * (G - Q_table[state][action])
        
    # Decay epsilon for exploration-exploitation balance
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

```
Monte Carlo methods and Q-learning are two fundamental approaches for estimating the value of actions (Q-values) in reinforcement learning.

## Sarsa: On-Policy TD Algorithm

Sarsa (stands for State-Action-Reward-State-Action) is another popular RL algorithm that is <b>on-policy</b>, which is used to learn a new policy for better decision-making. Unlike Q-learning, SARSA considers the action taken in the next state when updating Q-values. It updates its policy based on the actual actions taken by the agent, and that's why it is an on-policy algorithm.

$$ Q(s_{t+1}, a_{t+1}) = Q(s_{t+1}, a) + \alpha \left( R_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s, a) \right) $$

SARSA updates the Q-value for the current state-action pair based on the reward, the next state, and the action taken by the current policy.

### Q-learning vs Sarsa

The key difference between Q-learning and SARSA is that SARSA is an on-policy algorithm, meaning it updates the Q-values using the action actually taken by the policy, whereas Q-learning is an off-policy algorithm that updates the Q-values using the action that maximizes the Q-value.
In summary, SARSA is cautious, balancing exploration and exploitation, considering the current policy, while Q-learning boldly explores the best possible actions, for any given policy.

As can be seen, in the code difference, Q-learning: Uses epsilon-greedy for action selection at each step and updates Q-values based on the maximum Q-value for the next state.
SARSA: Uses epsilon-greedy for action selection at each step and updates Q-values based on the action taken in the next state.

### How Does it Work?

Imagine an agent navigating an environment (like a robot in a maze).
At each step:
It’s in a state (S).
It takes an action (A).
Receives a reward R.
Ends up in the next state (S1).
Chooses another action (A1) in the next state.
The tuple (S, A, R, S1, A1) represents SARSA.

### Algorithm 

Parameters: step size $\alpha = (0, 1]$, small greedy $\epsilon > 0$
Initialize Q(s, a), for s representing all states and a representing all possible actions, arbitrarily except that Q(terminal, ) = max reward value.

1. **Initialize** Q-values for all state-action pairs (Q-table).
2. **Loop over episodes**:
   - Initialize the current state: $S = S_0$.
   - Choose an action $a$ from state $S$ using a policy derived from Q (e.g., $\epsilon-greedy$ strategy).
   - **Loop over each step of the current episode**:
     - Take action $a$, observe the reward $R_t$ and the next state $S_{t+1}$.
     - Choose an action $a{t+1}$ from state $S_{t+1}$ using a policy derived from Q (e.g., $\epsilon-greedy$ strategy).
     - Update the Q-table with the observed reward and next state values.
     - Compute the **TD error**:
      $$  \text{error} = (R_{t+1} + \gamma \max_a Q(S_{t+1}, a)) - Q(S_t, a_t) $$
     - Update the Q-value using the step size $\alpha$ and the calculated TD error:
      $$  Q(S_t, a_t) = Q(S_t, a_t) + \alpha \cdot \text{error} $$
     - Transition to the next step: $S_t \leftarrow S_{t+1}$.
   - **Repeat until the current state is terminal** (i.e., $S_t = S_{\text{end}}$).

### Sarsa python Code

Instead of always picking the best next action with the max operation as we did in Q-learning, we apply the epsilon greedy approach to select the action for the next time step.

```python
 # Epsilon-greedy action selection for the next action
if np.random.uniform(0, 1) < epsilon:
   #Explore
   next_action = np.random.randint(0, n_actions)
else:
   #Exploit
   next_action = np.argmax(Q_table[next_state])
```
### Comparing the training curve of the three RL algorithms

<center>

![rl camparison](/assets/img/posts/2024-07-11/algo_comp.png)

</center>

### Monte Carlo Simulation

Instead of using fixed input values to test our models, Monte Carlo Simulation uses probability distributions for variables with inherent uncertainty. The simulation runs thousands of times, recalculating results each time. This yields a range of possible outcomes with associated probabilities. Then we can measure the rate of success or failure across all thousands of simulations, to have a better confidence about the state of our model. 

```python

#Testing with Monte Carlo Simulation

def eval_q(env, Q, n_sim=5):
    """Use Monte Carlo evaluation."""
    episode_rewards = np.zeros(n_sim)
    
    for episode_id in tqdm(range(n_sim)):
        state, _ = env.reset()
        done = False
        
        while not done:
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_rewards[episode_id]+=reward

    return episode_rewards

#Return average reward across all simulation
reward = np.mean(eval_q(env, Q_table, n_sim=10000))
print("Average Reward across all simulations is: "+ str(reward))
```

The average reward is -13 (which is the optimal result) across the 10,000 simulations, proving that our model gives the optimal solution in every case. 

### Display the results

The agent seems to follow the risky but optimal strategy of following the cliff to get to the target, which results in a total reward of -13. 

<center>

<img src="/assets/img/posts/2024-07-11/final_cliffwalking.gif">

</center>

# Deep Q learning

## Deep Q Network (DQN)

The main difference between the standard Q-learning algorithm and DQN (Deep Q Neural Networks) is that DQN uses a deep neural network to approximate the Q-values, while Q-learning relies on a Q-table to store the values.

```python
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine the next action or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
```

We first calculate the expected future rewards (Q values) based on the next state. If there is no next state, the expected value is just the immediate reward. It then gets the estimated value of the current action from the policy network. The Huber loss or L1loss is computed to measure the difference between the expected and estimated Q values. Finally, the model's parameters are adjusted to minimize this loss, improving the model's future predictions and this is done using a small learning rate.

```python
# Compute the expected Q values for the next time step
if next_state is None:
   expected_values = reward
else:
   next_state_values = policy_net(next_state).max(1)[0].detach()
   expected_values = (next_state_values * GAMMA) + reward

 # Get the estimated values from the policy network 
estimated_values = policy_net(state).gather(1, torch.tensor([[action]], device=device))

# Compute Huber loss
criterion = nn.SmoothL1Loss()
loss = criterion(estimated_values, expected_values)

# Optimize the model
optimizer.zero_grad()
loss.backward()
# In-place gradient clipping
torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
optimizer.step()
```

## Experimenting with other Games from the library

Experimenting with various environments can provide valuable insights into how well these algorithms generalize and perform across different tasks. The following are some popular environments available in OpenAI Gym:

CartPole: A classic control task where you balance a pole on a moving cart.
MountainCar: A car must drive up a steep hill.
Acrobot: A two-link pendulum must swing up and balance.
LunarLander: A lunar lander must land safely on the moon.
Atari Games: Various classic Atari games, such as Breakout, Pong, and Space Invaders, offer more complex and visually rich environments.
Each environment presents unique challenges and requires different strategies, making them excellent testbeds for exploring and refining reinforcement learning algorithms.

Let's pick Cartpole, for example. 

Link: https://gymnasium.farama.org/environments/classic_control/cart_pole/

The agent has to decide between two actions - moving the cart (1) left or (2) right - so that the pole attached to it stays upright.

As the agent observes the current state of the environment and chooses an action, the environment transitions to a new state. Also, it returns a reward that indicates the consequences of the action.

In this task, rewards are +1 for every incremental timestep and the environment terminates if the pole falls over too far or the cart moves more than 2.4 units away from the center. This means better-performing scenarios will run for a longer duration, accumulating larger rewards.

In the CartPole environment, the state space consists of four observations. These four observations represent the following physical properties of the cart-pole system:

- Cart Position: The position of the cart on the track.
- Cart Velocity: The velocity of the cart.
- Pole Angle: The angle of the pole concerning the vertical axis.
- Pole Angular Velocity: The angular velocity of the pole.


Here are the different observations and their ranges. 

| Num | Observation           | Min                  | Max                |
|-----|-----------------------|----------------------|--------------------|
| 0   | Cart Position         | -4.8*                 | 4.8*                |
| 1   | Cart Velocity         | -Inf                 | Inf                |
| 2   | Pole Angle            | ~ -0.418 rad (-24°)** | ~ 0.418 rad (24°)** |
| 3   | Pole Angular Velocity | -Inf                 | Inf                |

From: https://magalimorin18.github.io/reinforcement_learning/td2/discrete.html

The agent takes these 4 inputs from the environment without any scaling and (A) stores them in a Q-table or (B) passes them through a small fully connected network to output 2 classes, one for each action (left or right). 

### Discretization of the env

Unlike the previous example (Cliff Walking), in this environment (Cartpole), our observation space is represented by float values. Even though our DQN approach can deal with that. If we want to use a Q table, we need to discretize our environment by representing our observation space as a finite set of discrete values.

```python

# Discretize the env
DISCRET_NBR = [1, 1, 10, 6] # Number of values per dimension of the state

env_low = env.observation_space.low
env_high = env.observation_space.high
env_low[1], env_high[1] = -5, 5
env_low[3], env_high[3] = -math.radians(50), math.radians(50)
print(env_low)
print(env_high)
plt.plot(env_low)
plt.plot(env_high)
plt.title("Upper and Lower bounds")

#Use this function to map continuous values to their corresponding discrete bins
def convert_state_discrete(state):
    bucket_indice = []
    for state_idx in range(len(state)):
        if state[state_idx] <= env_low[state_idx]:
            bucket_index = 0
        elif state[state_idx] >= env_high[state_idx]:
            bucket_index = DISCRET_NBR[state_idx] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = env_high[state_idx] - env_low[state_idx]
            offset = (DISCRET_NBR[state_idx] - 1) * env_low[state_idx] / bound_width
            scaling = (DISCRET_NBR[state_idx] - 1) / bound_width
            bucket_index = int(round(scaling * state[state_idx] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)
```

And that's a wrap! I hope you enjoyed it and you got to learn something from it! 
Keep exploring, keep learning. Onward to the next subject!


You can find the entire Notebook here: https://github.com/faresbs/Machine-learning-Applications/blob/master/reinforcement_learning/hands-on-rl.ipynb
