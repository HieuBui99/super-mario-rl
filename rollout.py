import numpy as np
import torch
from torch.distributions import Categorical
from pathlib import Path
from src.env import create_env
from src.model import Agent


weight_path = Path("weight/model_final.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"

world, stage = 1, 1
env = create_env(world, stage)

agent = Agent(env.observation_space.shape[0], env.action_space.n).to(device)
agent = agent.load_state_dict(torch.load(weight_path))
agent.eval()

obs, _ = env.reset()
obs = torch.tensor(np.array(obs), dtype=torch.float32).to(device)
obs = obs.squeeze(-1).unsqueeze(0)

max_step = 10000
step = 0
while True:
    with torch.no_grad():
        logits, _ = agent(obs)
        probs = torch.softmax(logits, dim=-1)
        action = torch.argmax(probs).item()

    obs, reward, done, _, info = env.step(action)

    obs = torch.from_numpy(np.array(obs)).float().to(device)
    obs = obs.squeeze(-1).unsqueeze(0)

    env.render()
    step += 1
    if done or step >= max_step or info["flag_get"]:
        break





