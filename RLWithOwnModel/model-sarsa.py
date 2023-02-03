import numpy as np
from time import sleep
from IPython.display import clear_output
from redistribution_env import RedistributionEnv

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
    env = RedistributionEnv(render_mode='ansi')
    env.reset()

    state_size = env.observation_space.n
    action_size = env.action_space.n

    frames = []

    epsilon = 0.9
    num_episodes = 5000
    max_steps = 99
    learning_rate = 0.85
    discount_rate = 0.95
    decay_rate = 0.005

    q_table = np.zeros([state_size, action_size])

    reward = 0

    for episode in range(num_episodes):
        state = env.reset()
        if np.random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state, :])       
        for step in range(max_steps):
            new_state, reward, done, trunc = env.step(action)
            if np.random.uniform(0,1) < epsilon:
                new_action = env.action_space.sample()
            else:
                new_action = np.argmax(q_table[new_state, :])  
            q_table[state, action] = q_table[state, action] + learning_rate * (reward + (discount_rate * q_table[new_state,new_action] - q_table[state, action]))

            state = new_state
            action = new_action

            if done:
                break
        epsilon = np.exp(-decay_rate*episode)
    
    print(f"Training completed over {num_episodes} episodes")
    input("Press Enter to observe the trained model...")

    state = env.reset()
    rewards = 0

    for step in range(max_steps):

        action = np.argmax(q_table[state,:])
        new_state, reward, done, trunc = env.step(action)
        rewards += reward
        print(f"score: {rewards}")
        state = new_state

        frames.append({
            'frame': env.render(),
            'state': state,
            'action': action,
            'reward': reward
        })

        if done:
            break

    print_frames(frames)

    env.close()

if __name__ == "__main__":
    main()
