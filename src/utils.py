import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
import random
import os
from pickle import dump
from pickle import load as load_
from decimal import Decimal
from copy import deepcopy
import xgboost as xgb
from gymnasium.wrappers import TimeLimit
from fast_env import FastHIVPatient
from fast_evaluate import evaluate_HIV, evaluate_HIV_population

env_fixed = TimeLimit(
    env=FastHIVPatient(domain_randomization=False), max_episode_steps=200
)


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        # print(batch)
        # print(list(zip(*batch)))
        # for x in list(zip(*batch)):
        #     print(x)
        #     print(np.array(x))
        return [torch.Tensor(np.array(x)).to(self.device) for x in list(zip(*batch))]
    def __len__(self):
        return len(self.data)


def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()



class dqn_agent:
    def __init__(self, config, model):
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size,device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model 
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        self.monitoring_nb_trials = config['monitoring_nb_trials'] if 'monitoring_nb_trials' in config.keys() else 0
        self.step = 0
        self.ep_step = 0.0
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def transform_state(self, state):
        T1 = state[0]  # healthy type 1 cells concentration (cells per mL)
        T1star = state[1]  # infected type 1 cells concentration (cells per mL)
        T2 = state[2]  # healthy type 2 cells concentration (cells per mL)
        T2star = state[3]  # infected type 2 cells concentration (cells per mL)
        ratio1 = T1/(T1+T1star)
        ratio2 = T2/(T2+T2star)
        new_state = np.append(state, [ratio1, ratio2])
        return np.append(state, [self.ep_step/200])# state# new_state
    
    def train(self, env, max_episode):
        episode_return = []
        scores = []
        episode = 0
        self.ep_step = 0.0
        episode_cum_reward = 0
        state, _ = env.reset()
        state = self.transform_state(state)
        epsilon = self.epsilon_max
        while episode < max_episode:
            # update epsilon
            if self.step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            next_state = self.transform_state(next_state)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.step % self.update_target_freq == 0: 
                self.target_model.load_state_dict(self.model.state_dict())
            # next transition
            self.step += 1
            self.ep_step += 1

            if done or trunc:
                # Evaluate model.
                seed_everything(seed=42)
                if episode%5==0:
                    score_agent = evaluate_HIV(agent=self, nb_episode=5)
                    print("Real score agent: ", '%.2E' % Decimal(score_agent))
                    scores.append(score_agent)
                    if score_agent > 1e9:
                        self.save('./models/DQN1e9.pth')
                        print("Model saved ", score_agent)
                    if score_agent > 1e10:
                        self.save('./models/DQN1e10.pth')
                        print("Model saved ", score_agent)

                episode += 1
                # Monitoring
                episode_return.append(episode_cum_reward)
                print("Episode ", '{:2d}'.format(episode), 
                        ", epsilon ", '{:6.2f}'.format(epsilon), 
                        ", batch size ", '{:4d}'.format(len(self.memory)), 
                        ", ep return ", '%.2E' % Decimal(episode_cum_reward), 
                        sep='')
                state, _ = env.reset()
                state = self.transform_state(state)
                episode_cum_reward = 0
                self.ep_step = 0.0
            else:
                state = next_state
        return episode_return, scores

    def act(self, observation, use_random=False):
        self.ep_step = self.ep_step+1 % 200
        if len(observation) == 6:
            observation = self.transform_state(observation)
        with torch.no_grad():
            if use_random:
                return random.randint(0, 3)
            else:
                return greedy_action(self.model, observation)
        
    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))  # Ensure the directory exists

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()}
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load(self):
        checkpoint = torch.load('./models/DQN.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")


class fqi_agent:
    def __init__(self, config = {}):
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.98
        self.n_actions = config['actions'] if 'actions' in config.keys() else 4
        self.dim_state = config['states'] if 'states' in config.keys() else 8
        self.Qfunctions = []
        self.samples = []
        self._sa_buffer = np.zeros((self.n_actions, self.dim_state + 1))
        for a in range(self.n_actions):
            self._sa_buffer[a, self.dim_state] = a
    
    def greedy_action(self, state):
        s = self.  transform_state(state)
        if len(self.Qfunctions) == 0:
            return np.random.randint(4)
        else:
            Q = self.Qfunctions[-1]
            for a in range(self.n_actions):
                self._sa_buffer[a, :len(s)] = s
            return np.argmax(Q.predict(self._sa_buffer))


    def collect_samples(self, env:gym.Env, n_traj_unhealthy=30, n_traj_healthy=0, n_traj_uninfected=0, eps=.15, disable_tqdm=False):
        print("Collecting samples.")
        for mode, n_traj in zip(['unhealthy', 'healthy', 'uninfected'], [n_traj_unhealthy, n_traj_healthy, n_traj_uninfected]):
            for _ in tqdm(range(n_traj), disable=disable_tqdm):
                timestep = 0
                s, _ = env.reset(options={'mode':mode})
                while True:
                    if np.random.rand() < eps:
                        a = env.action_space.sample()
                    else:
                        a = self.greedy_action(s)
                    s2, r, done, trunc, _ = env.step(a)
                    self.samples.append((s, a, r, s2, done or trunc, timestep))
                    if done or trunc:
                        break
                    else:
                        s = s2
                        timestep += 1

    def get_samples(self):
        samp = [x for x in zip(*self.samples)]
        S = np.array([self.transform_state(s) for s in samp[0]]) 
        A = np.array(samp[1]) 
        R = np.array(samp[2]) 
        S2 = np.array([self.transform_state(s) for s in samp[3]]) 
        D = np.array(samp[4]) 

        return S, A, R, S2, D
    
    def transform_state(self, state):
        T1 = state[0]  # healthy type 1 cells concentration (cells per mL)
        T1star = state[1]  # infected type 1 cells concentration (cells per mL)
        T2 = state[2]  # healthy type 2 cells concentration (cells per mL)
        T2star = state[3]  # infected type 2 cells concentration (cells per mL)
        ratio1 = T1/(T1+T1star) # two ratios that help the model to learn the dynamics of the system
        ratio2 = T2/(T2+T2star)
        new_state = np.append(state, [ratio1, ratio2])
        return new_state# np.append(state, [self.ep_step/200])# state# new_state

    def train(self, env:gym.Env, num_epochs=150, iterations_per_epoch=200):
        best_score = 0
        best_score_random = 0
        epoch = 0
        while epoch < num_epochs:
            eps = max(0.02, eps - 0.01*epoch)
            self.collect_samples(env, eps=eps)
            S, A, R, S2, D = self.get_samples()
            print("Starting epoch", epoch + 1)
            for i in tqdm(range(iterations_per_epoch)):
                if len(self.Qfunctions) == 0:
                    target = R.copy()
                else:
                    Q2 = np.empty(shape=(self.n_actions, len(self.samples)))
                    for a2 in range(self.n_actions):
                        S2A2 = np.append(S2, np.ones(shape=(len(self.samples), 1)), axis=1)
                        Q2[a2] = self.Qfunctions[-1].predict(S2A2)
                    target = R + self.gamma*(1-D)*np.max(Q2, axis=0)
                    
                Q = xgb.XGBRegressor(n_estimators=50)
                Q.fit(np.append(S, A, axis=1), target)
                self.Qfunctions.append(Q)
                
                score_agent = evaluate_HIV(agent=self, nb_episode=1)
                if score_agent > best_score:
                    best_score = score_agent
                    print("New best score:", '%.2E' % Decimal(best_score))
                    self.save("./models/best_fixed_patient_model.pkl")
                    
                if score_agent > 2e10:
                    score_agent_random = evaluate_HIV_population(agent=self, nb_episode=20) 
                    if score_agent_random > best_score_random:
                        best_score_random = score_agent_random
                        self.save()
                        best_score_random = score_agent_random
                    print("New best population score:", '%.2E' % Decimal(best_score_random))
            epoch += 1


    def act(self, observation, use_random=False):
        if use_random:
            return random.randint(0, 3)
        else:
            return self.greedy_action(observation)

    def save(self, path="./models/FQI.pkl"):
        if len(self.Qfunctions) == 0:
            print("Error: Qfunctions list is empty")
            return
        with open(path, "wb") as f:
            dump(self.Qfunctions[-1], f, protocol=5)

    def load(self, path="./models/FQI.pkl"):
        with open(path, "rb") as f:
            Q = load_(f)
            self.Qfunctions = [Q]