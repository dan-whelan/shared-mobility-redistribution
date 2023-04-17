import numpy as np
from time import sleep
from IPython.display import clear_output
import matplotlib.pyplot as plt
import gym

def print_frames(frames):
    for i, frame in enumerate(frames):
        print(frame['frame'])
        print(f"Timestep: {i+1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)
        clear_output()

def _moving_average(x, periods=5):
    if len(x) < periods:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    res = (cumsum[periods:] - cumsum[:-periods]) / periods
    return np.hstack([x[:periods-1], res])

def plot_durations(episode_durations, total_rewards, exploration_rate_vec):
    lines = []
    fig = plt.figure(1, figsize=(15, 7))
    plt.clf()
    axis1 = fig.add_subplot(111)

    plt.title('SARSA Training')
    axis1.set_xlabel('Episode')
    axis1.set_ylabel('Duration & Rewards')
    axis1.set_ylim(-3 * 100, 100 + 10)
    axis1.plot(episode_durations, color="C1", alpha=0.2)
    axis1.plot(total_rewards, color="C2", alpha=0.2)
    mean_steps = _moving_average(episode_durations, periods=5)
    mean_reward = _moving_average(total_rewards, periods=5)
    lines.append(axis1.plot(mean_steps, label="steps", color="C1")[0])
    lines.append(axis1.plot(mean_reward, label="rewards", color="C2")[0])


    axis2 = axis1.twinx()
    axis2.set_ylabel('Epsilon')
    lines.append(axis2.plot(exploration_rate_vec, label="epsilon", color="C3")[0])
    labs = [l.get_label() for l in lines]
    axis1.legend(lines, labs, loc=3)
    plt.show()

    plt.pause(0.001)

def main():
    env = gym.make("Taxi-v3", render_mode="ansi")
    env.reset()

    state_size = env.observation_space.n
    action_size = env.action_space.n

    frames = []

    epsilon = 0.9
    num_episodes = 1800
    max_steps = 99
    learning_rate = 0.85
    discount_rate = 0.95
    decay_rate = 0.005

    q_table = np.zeros([state_size, action_size])

    reward = 0

    episode_durations = []
    total_rewards = []
    exploration_rate_vec = []

    for episode in range(num_episodes):
        this_state = env.reset()
        state = this_state[0]
        rewards_in_episode = 0
        if np.random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state, :])       
        for step in range(max_steps):
            new_state, reward, done, trunc, info = env.step(action)
            if np.random.uniform(0,1) < epsilon:
                new_action = env.action_space.sample()
            else:
                new_action = np.argmax(q_table[new_state, :])  
            q_table[state, action] = q_table[state, action] + learning_rate * (reward + (discount_rate * q_table[new_state,new_action] - q_table[state, action]))
            action = new_action
            state = new_state
            rewards_in_episode += reward
            if done:
                episode_durations.append(step)
                total_rewards.append(rewards_in_episode)
                exploration_rate_vec.append(epsilon)
                plot_durations(episode_durations, total_rewards, exploration_rate_vec)
                break

        epsilon = np.exp(-decay_rate*episode)
    
    print(f"Training completed over {num_episodes} episodes")
    input("Press Enter to observe the trained model...")

    this_state = env.reset()
    state = this_state[0]
    rewards = 0

    for step in range(max_steps):
        print(f"TRAINED AGENT")
        print("Step {}".format(step+1))

        action = np.argmax(q_table[state,:])
        new_state, reward, done, trunc, info = env.step(action)
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
