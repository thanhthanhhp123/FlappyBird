from collections import deque
from itertools import count
import random
import numpy as np
import gymnasium


class Memory():
    def __init__(self, memory_size = 10000):
        self.memory_size = memory_size
        self.buffer = deque(maxlen=memory_size)
    
    def add(self, exp) -> None:
        self.buffer.append(exp)
    
    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]
        
    def clear(self):
        self.buffer.clear()



class FrameStack(gymnasium.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        super().__init__(env)

        self.k = k
        self.frames = deque([], maxlen=k)

        low = np.repeat(self.observation_space.low[np.newaxis, ...], k, axis=0)
        high = np.repeat(self.observation_space.high[np.newaxis, ...], k, axis=0)
        self.observation_space = gymnasium.spaces.Box(
            low=low,
            high=high,
            dtype=self.observation_space.dtype,
        )

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self._set_ob(ob)
        return self._get_ob(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self._set_ob(ob)
        return self._get_ob(), reward, terminated, truncated, info

    def _set_ob(self, ob):
        self.frames.append(ob)

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.stack(list(self.frames), axis=0)  # stack along time axis




class FlappyBirdPresets:
    @staticmethod
    def easy():
        return {
            'PIPE_VEL_X': -3,           # Slower pipes
            'PLAYER_MAX_VEL_Y': 8,      # Slower falling
            'PLAYER_FLAP_ACC': -7,      # Gentler flapping
            'PLAYER_ACC_Y': 0.8,        # Slower gravity
            'PIPE_WIDTH': 60            # Wider pipes
        }
    
    @staticmethod
    def normal():
        return {
            'PIPE_VEL_X': -4,          # Default values
            'PLAYER_MAX_VEL_Y': 10,
            'PLAYER_FLAP_ACC': -9,
            'PLAYER_ACC_Y': 1,
            'PIPE_WIDTH': 52
        }
    
    @staticmethod
    def hard():
        return {
            'PIPE_VEL_X': -10,          # Faster pipes
            'PLAYER_MAX_VEL_Y': 12,    # Faster falling
            'PLAYER_FLAP_ACC': -10,    # Stronger flapping
            'PLAYER_ACC_Y': 2,       # Stronger gravity
            'PIPE_WIDTH': 1           # Narrower pipes
        }