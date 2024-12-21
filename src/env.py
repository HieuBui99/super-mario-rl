
import cv2
import gym
import gym_super_mario_bros
import numpy as np
from gym import ObservationWrapper
from gym.spaces import Box
from gym.wrappers import FrameStack
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]."""
        obs, info = self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(
                1, self.noop_max + 1
            )  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _,  _ = self.env.step(self.noop_action)
            if done:
                obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, ac):
        return self.env.step(ac)


class NormalizeFrame(ObservationWrapper):
    def __init__(self, env):
        super(NormalizeFrame, self).__init__(env)
        self.observation_space = Box(
            low=0,
            high=1,
            shape=(self.observation_space.shape),
            dtype=np.float32
        )

    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0
    

class ResizeFrame(ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True):
        super(ResizeFrame, self).__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        if self._grayscale:
            num_channels = 1
        else:
            num_channels = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_channels),
            dtype=np.uint8,
        )
        original_space = self.observation_space
        self.observation_space = new_space

        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        frame = obs

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        return frame
    

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        super(SkipFrame, self).__init__(env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, truncate, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, truncate, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
def create_env(world, stage):
    env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-v0", apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = NoopResetEnv(env, noop_max=30)
    env = SkipFrame(env, skip=4)
    env = ResizeFrame(env, width=84, height=84, grayscale=True)
    env = NormalizeFrame(env)
    env = FrameStack(env, num_stack=4)
    return env