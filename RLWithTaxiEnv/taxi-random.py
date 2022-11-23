import gym
from IPython.display import clear_output
from time import sleep

def print_frames(frames):
    for i, frame in enumerate(frames):
        print(frame['frame'])
        print(f"Timestep: {i+1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)
        clear_output()


def main(): 
    env = gym.make("Taxi-v3", render_mode = "ansi")

    env.reset()

    # Solve the environment without reinforcement
    env.s = 328

    epochs, penalties, reward = 0, 0, 0

    frames = []

    done = False

    while not done:
        action = env.action_space.sample()
        state, reward, done, trunc, info = env.step(action)
        
        # Put each rendered frame into a dict for animation
        frames.append({
            'frame': env.render(),
            'state': state,
            'action': action,
            'reward': reward
            }
        )

    print_frames(frames)

if __name__ == "__main__":
    main()
