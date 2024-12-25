from pathlib import Path

import numpy as np
import torch

import imageio
from tqdm import tqdm
from src.env import *
from src.model import Agent

weight_path = Path("/home/aki/workspace/learning/super-mario-rl/weight/model_2700000.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"

world, stage = 1, 1
env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-v0", apply_api_compatibility=True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
monitor = MonitorEnv()
env = NoopResetEnv(env, noop_max=30, monitor=monitor)
env = SkipFrame(env, skip=4)
env = ResizeFrame(env, width=84, height=84, grayscale=True)
env = NormalizeFrame(env)
env = FrameStack(env, num_stack=4)

agent = Agent(env.observation_space.shape[0], env.action_space.n, ).to(device)
agent.load_state_dict(torch.load(weight_path))
agent.eval()

obs, _ = env.reset()
obs = torch.tensor(np.array(obs), dtype=torch.float32).to(device)
obs = obs.squeeze(-1).unsqueeze(0)

max_step = 10000
step = 0

pbar = tqdm(range(max_step))
while True:
    with torch.no_grad():
        logits, _ = agent(obs)
        probs = torch.softmax(logits, dim=-1)
        action = torch.argmax(probs).item()

    obs, reward, done, _, info = env.step(action)

    obs = torch.from_numpy(np.array(obs)).float().to(device)
    obs = obs.squeeze(-1).unsqueeze(0)


    step += 1
    if done or step > max_step:
        break
        # obs, _ = env.reset()
        # obs = torch.tensor(np.array(obs), dtype=torch.float32).to(device)
        # obs = obs.squeeze(-1).unsqueeze(0)
    if step > max_step:
        break
    pbar.update(1)

print(len(monitor.store))
# print(len(frames), done, info)
# save video
imageio.mimsave("output.mp4", monitor.store, fps=60)
