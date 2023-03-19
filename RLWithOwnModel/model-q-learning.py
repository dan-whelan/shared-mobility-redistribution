from time import sleep
from IPython.display import clear_output 
import numpy as np
from redistribution_env import RedistributionEnv2
import matplotlib.pyplot as plt
from config import Config

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
    env = RedistributionEnv2()
    config_file = "RLWithOwnModel/config/config.yaml"
    config = Config(config_file)

    num_episodes = config.training.num_episodes
    max_steps_per_episode = config.rl.max_steps_per_episode
    episode_durations = []
    total_rewards = []
    exploration_rate_vec = []

    alpha = config.training.alpha
    gamma = config.rl.gamma
    epsilon = config.epsilon.max_epsilon
    min_epsilon = config.epsilon.min_epsilon
    epsilon_decay = config.epsilon.decay_epsilon

    frames = []

    q_table_shape = (
        env.observation_space["truck_position"].n,
        *env.observation_space["bike_states"].nvec,
        env.observation_space["bikes_on_truck"].n,
        env.action_space.n
    )
    Q = np.zeros(q_table_shape)

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

        plt.title('Training...')
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
        for episode in range(num_episodes):
            state = env.reset()
            done = False   
            step = 0
            rewards_in_episode = 0

            while not done:
                state_tuple = (state["truck_position"], *state["bike_states"], state["bikes_on_truck"])
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(Q[state_tuple])
                
                new_state, reward, done = env.step(action)

                new_state_tuple = (new_state["truck_position"], *new_state["bike_states"], new_state["bikes_on_truck"])
                Q[state_tuple + (action,)] = ((1-alpha)*Q[state_tuple + (action,)]) + (alpha * (reward + gamma * np.max(Q[new_state_tuple] - Q[state_tuple + (action,)])))

                state = new_state
                rewards_in_episode += reward
                step += 1

                if done:
                    episode_durations.append(step)
                    total_rewards.append(rewards_in_episode)
                    exploration_rate_vec.append(epsilon)
                    plot_durations()
            if (episode + 1) % 1000 == 0:
                print("Episode {}/{}".format(episode+1, num_episodes))
            
            if epsilon > min_epsilon:
                epsilon = np.exp(-epsilon_decay*episode)

    except KeyboardInterrupt:
        plot_durations()
        print("Training has been interrupted")

    print(f"Training Completed over {num_episodes} episodes")
    
    total_rewards = []
    for i in range(100):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            state_tuple = (state["truck_position"], *state["bike_states"], state["bikes_on_truck"])
            action = np.argmax(Q[state_tuple,:])
            state, reward, done = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)

    print("Average reward over 100 test episodes: {}".format(np.mean(total_rewards)))
    input("Press Enter to watch the trained agent...")

    state = env.reset()
    done = False
    rewards = 0

    for s in range(max_steps_per_episode):

        print("TRAINED AGENT")
        print("Step {}".format(s+1))
        state_tuple = (state["truck_position"], *state["bike_states"], state["bikes_on_truck"])
        action = np.argmax(Q[state_tuple])
        new_state, reward, done = env.step(action)
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
