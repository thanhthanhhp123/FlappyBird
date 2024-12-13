
import gymnasium
import flappy_bird_gymnasium



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

# Usage example:
env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar = False)
print(env.unwrapped.constants)


# Set difficulty to easy
env.unwrapped.update_constants(**FlappyBirdPresets.hard())

# Play game...
obs, info = env.reset()

while True:
    action = env.action_space.sample()
    
    obs, reward, done,  _, info = env.step(action)
    # print(f" Action: {action}, Reward: {reward}, Done: {done}, Info: {info}")
    if done:
        break
    