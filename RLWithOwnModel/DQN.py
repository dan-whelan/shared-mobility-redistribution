from redistribution_env import RedistributionEnv2
from agent import QAgent
from model import DQN

def test_agent():
    env = RedistributionEnv2()
    config = "RLWithOwnModel/config/config_DQN.yaml"

    agent = QAgent(env=env, config=config, model_class=DQN)
    agent.compile()
    agent.fit()
    input("Press Enter to end training")
    agent.play()

if __name__ == '__main__':
    test_agent()