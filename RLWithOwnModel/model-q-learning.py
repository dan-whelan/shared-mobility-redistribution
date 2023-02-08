from time import sleep
from IPython.display import clear_output 
import numpy as np
import random
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
    env = RedistributionEnv(render_mode="ansi")
    env.reset()

    state_size = env.observation_space.n
    action_size = env.action_space.n

    frames = []

    q_table = np.zeros([state_size, action_size])

    learning_rate = 0.9
    discount_rate = 0.8
    epsilon = 1
    decay_rate = 0.005

    num_episodes = 5000
    max_steps = 99

    for episode in range(num_episodes):
        state = env.reset()
            
        done = False

        for s in range(max_steps):
            if random.uniform(0,1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state,:])
            
            new_state, reward, done, _ = env.step(action)

            q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount_rate * np.max(q_table[new_state,:])-q_table[state, action])

            state = new_state

            if done == True:
                break
        
        epsilon = np.exp(-decay_rate*episode)

    print(f"Training Completed over {num_episodes} episodes")
    input("Press Enter to watch the trained agent...")

    state = env.reset()
    done = False
    rewards = 0

    for s in range(max_steps):

        print(f"TRAINED AGENT")
        print("Step {}".format(s+1))

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
            }
        )

        if done == True:
            break
    
    print_frames(frames)
    
    env.close()

if __name__ == "__main__":
    main()
