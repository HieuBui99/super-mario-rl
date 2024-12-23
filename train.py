import torch
import numpy as np
from src.env import create_env
from src.model import Agent
from src.replay import ReplayBuffer




####Configurations####
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 1

#learning hyperparameters
lr = 1e-4
gamma = 0.9
tau = 1.0 #GAE
beta = 0.01 #entropy
eps = 0.2 #PPO clip

bs = 32
epochs = 10

train_steps = 2000000
max_episode_len = 512
update_steps = max_episode_len * 2
#######################


world, stage = 1, 1
env = create_env(world, stage)


# seed everything
if torch.cuda.is_available():
    torch.cuda.manual_seed(123)
else:
    torch.manual_seed(123)

model = Agent(env.observation_space.shape[0], env.action_space.n).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


step = 0
episode = 0

while step < train_steps:
    observation, _ = env.reset()
    episode_reward = 0

    #replay buffer
    actions = []
    observations = []
    logprobs = []
    rewards = []
    state_values = []
    is_terminals = []

    cur_obs = observation
    cur_obs = torch.tensor(np.array(observation), dtype=torch.float32).to(device)
    cur_obs = cur_obs.squeeze(-1).unsqueeze(0)
    for t in range(max_episode_len):
        #take a step in the environment
        with torch.no_grad():
            action, logprob, state_value = model.act(cur_obs)


        observation, reward, done, _, _ = env.step(action.detach().cpu().item())

        observation = torch.from_numpy(np.array(observation)).float().to(device)
        observation = observation.squeeze(-1).unsqueeze(0)
        cur_obs = observation

        actions.append(action.detach())
        observations.append(observation)
        logprobs.append(logprob.detach())
        rewards.append(reward)
        state_values.append(state_value.detach())
        is_terminals.append(torch.tensor(done).float().to(device))

        episode_reward += reward
        step += 1

        if step % update_steps == 0:
            R = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + gamma * discounted_reward
                R.insert(0, discounted_reward)

            R = torch.tensor(R).float().to(device)
            R = (R - R.mean()) / (R.std() + 1e-7)

            old_obs = torch.cat(observations, dim=0)
            old_actions = torch.cat(actions, dim=0)
            old_logprobs = torch.cat(logprobs, dim=0)
            old_state_values = torch.cat(state_values, dim=0)

            print(old_obs.shape, old_actions.shape, old_logprobs.shape, old_state_values.shape, R.shape)
        
        if done:
            break
    # break

