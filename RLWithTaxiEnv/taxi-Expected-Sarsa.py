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

def main():
    env = gym.make("Taxi-v3", render_mode="ansi")

    num_episodes = 1000
    max_steps_per_episode = 99
    episode_durations = []
    total_rewards = []
    exploration_rate_vec = []

    alpha = 0.85
    gamma = 0.95
    epsilon = 0.9
    min_epsilon = 0.01
    epsilon_decay = 0.005

    frames = []
    num_actions = env.action_space.n

    state_size = env.observation_space.n
    action_size = env.action_space.n
   
    Q =  np.zeros([state_size, action_size]) 

    def _moving_average(x, periods=5):
        if len(x) < periods:
            return x
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        res = (cumsum[periods:] - cumsum[:-periods]) / periods
        return np.hstack([x[:periods-1], res])

    def plot_durations():
        lines = []
        fig = plt.figure(1, figsize=(15, 7))
        plt.clf()
        axis1 = fig.add_subplot(111)

        plt.title('Q-Learning Training')
        axis1.set_xlabel('Episode')
        axis1.set_ylabel('Duration & Rewards')
        axis1.set_ylim(-3 * max_steps_per_episode, max_steps_per_episode + 10)
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
    
    try:
        for episode in range(1, num_episodes + 1):
            this_state = env.reset()
            state = this_state[0]
            action = 0
            step = 0
            rewards_in_episode = 0
            done = False
            for timestep in range(max_steps_per_episode):
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(Q[state])
                next_state, reward, done, trunc, info = env.step(action)

                policy = np.ones(num_actions) * epsilon / num_actions
                best_action = np.argmax(Q[next_state])
                policy[best_action] += 1 - epsilon
                expected_value = np.dot(Q[next_state], policy)
                Q[state, action] += alpha * (reward + gamma * expected_value - Q[state, action])

                state = next_state
                step += 1
                rewards_in_episode += reward

                # if timestep == max_steps_per_episode:
                #     done = True

                if done:
                    episode_durations.append(step)
                    total_rewards.append(rewards_in_episode)
                    exploration_rate_vec.append(epsilon)
                    plot_durations()
                    break
            
            if epsilon > min_epsilon:
                epsilon = np.exp(-epsilon_decay*episode)
    except KeyboardInterrupt:
        plot_durations()
        print("Training has been interrupted")

    print("Average reward over 100 test episodes: {}".format(np.mean(total_rewards)))
    input("Press Enter to watch the trained agent...")

    this_state = env.reset()
    state = this_state[0]
    done = False
    rewards = 0

    for s in range(max_steps_per_episode):

        print("TRAINED AGENT")
        print("Step {}".format(s+1))
        action = np.argmax(Q[state])
        new_state, reward, done, trunc, info = env.step(action)
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
