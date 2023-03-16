import os
import gym
from redistribution_env import RedistributionEnv
from config import Config
from agent import QAgent
from model import DQN

def test_agent():
    env = RedistributionEnv()
    config = "RLWithOwnModel/config/config_DQN.yaml"

    agent = QAgent(env=env, config=config, model_class=DQN)
    agent.compile()
    agent.fit()
    input("Press Enter to end training")
    agent.play()

if __name__ == '__main__':
    test_agent()