from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from utils import fqi_agent

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

ProjectAgent = fqi_agent
    # def act(self, observation, use_random=False):
    #     return self.agent.greedy_action(observation)

    # def save(self, path):
    #     self.agent.save_last_Qfunction(path)

    # def load(self):
    #     self.agent.load_Qfunction()

    # def __init__(self):
    #     self.agent = fqi_agent()

if __name__ == '__main__':
    config = {'actions': env.action_space.n,
              'states' : 8,
          'gamma': 0.97, # Def: 0.95
          'monitoring_nb_trials': 0}

    agent = fqi_agent(config)
    agent.train(env)