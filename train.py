import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from src.env import create_env
from src.model import Agent
from src.replay import ReplayBuffer

####Configurations####
device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 1

# learning hyperparameters
lr = 1e-4
gamma = 0.9
tau = 1.0  # GAE
beta = 0.01  # entropy
eps = 0.2  # PPO clip

bs = 32
epochs = 10

train_steps = 5000000
max_episode_len = 512
update_steps = max_episode_len * 2

print_freq = 10000
save_freq = 100000
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

replay_buffer = ReplayBuffer()

step = 0
episode = 0

running_reward = 0
running_episodes = 0


pb = tqdm(range(train_steps))
while step < train_steps:
    observation, _ = env.reset()
    episode_reward = 0

    cur_obs = observation
    cur_obs = torch.tensor(np.array(observation), dtype=torch.float32).to(device)
    cur_obs = cur_obs.squeeze(-1).unsqueeze(0)
    for t in range(max_episode_len):
        # take a step in the environment
        with torch.no_grad():
            action, logprob, state_value = model.act(cur_obs)

        observation, reward, done, _, _ = env.step(action.detach().cpu().item())

        observation = torch.from_numpy(np.array(observation)).float().to(device)
        observation = observation.squeeze(-1).unsqueeze(0)
        cur_obs = observation

        replay_buffer.actions.append(action.detach())
        replay_buffer.observations.append(observation)
        replay_buffer.logprobs.append(logprob.detach())
        replay_buffer.rewards.append(reward)
        replay_buffer.state_values.append(state_value.detach())
        replay_buffer.is_terminals.append(torch.tensor(done).float().to(device))

        episode_reward += reward
        step += 1

        if step % update_steps == 0:
            with torch.no_grad():
                _, next_value = model(cur_obs)

            rewards = []
            gae = 0
            for reward, value, is_terminal in list(
                zip(
                    replay_buffer.rewards,
                    replay_buffer.state_values,
                    replay_buffer.is_terminals,
                )
            )[::-1]:
                gae = gae * gamma * tau
                gae = (
                    gae
                    + reward
                    + gamma * next_value.detach() * (1 - is_terminal)
                    - value.detach()
                )
                next_value = value
                rewards.append(gae + value)
            rewards = rewards[::-1]

            rewards = torch.tensor(rewards).float().to(device)
            # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
            rewards = rewards.detach()

            old_obs = torch.cat(replay_buffer.observations, dim=0).detach()
            old_actions = torch.cat(replay_buffer.actions, dim=0).detach()
            old_logprobs = torch.cat(replay_buffer.logprobs, dim=0).detach()
            old_state_values = torch.cat(replay_buffer.state_values, dim=0).detach()

            advantages = rewards - old_state_values.squeeze(-1)
            for _ in range(epochs):
                indices = torch.randperm(len(replay_buffer.observations))

                for j in range(0, len(indices), bs):
                    batch_indices = indices[j : j + bs]

                    logprobs, state_values, dist_entropy = model.evaluate(
                        old_obs[batch_indices], old_actions[batch_indices]
                    )

                    ratio = torch.exp(logprobs - old_logprobs[batch_indices])
                    surr1 = ratio * advantages[batch_indices]
                    surr2 = (
                        torch.clamp(ratio, 1 - eps, 1 + eps) * advantages[batch_indices]
                    )
                    actor_loss = -torch.min(surr1, surr2).mean()

                    critic_loss = 0.5 * F.mse_loss(
                        state_values.squeeze(-1), rewards[batch_indices]
                    )
                    entropy_loss = -beta * dist_entropy.mean()

                    loss = actor_loss + critic_loss + entropy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    # loss = -torch.min(ratio * advantages[batch_indices], torch.clamp(ratio, 1-eps, 1+eps) * advantages[batch_indices]) + 0.5 * F.mse_loss(state_values, rewards[batch_indices]) - beta * dist_entropy
            replay_buffer.clear()

        if step % print_freq == 0:
            print(
                f"Step: {step}, Episode: {episode}, Reward: {running_reward / running_episodes}"
            )
            running_reward = 0
            running_episodes = 0

        if step % save_freq == 0:
            print(f"Saving model at step {step}")
            torch.save(model.state_dict(), f"model_{step}.pt")

        if done:
            break
        
        pb.update(1)

    running_reward += episode_reward
    running_episodes += 1
    episode += 1
    

torch.save(model.state_dict(), "model_final.pt")
