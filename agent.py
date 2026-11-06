import gymnasium as gym
import numpy as np

import matplotlib
# 'Agg': 将 plots 保存为图片而不是渲染到屏幕
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import random
import yaml

from dqn import DQN
from experience_replay import ReplayMemory

from datetime import datetime, timedelta
import argparse
import itertools

import torch
from torch import nn

import flappy_bird_gymnasium
import os

DATE_FORMAT = '%m-%d %H:%M:%S'
RUNS_DIR = 'runs'
os.makedirs(RUNS_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

class Agent:
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yaml', 'r', encoding='utf-8') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            print(hyperparameters)
        
        self.hyperparameter_set = hyperparameter_set
        
        self.env_id             = hyperparameters['env_id']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size    = hyperparameters['mini_batch_size']
        self.epsilon_init       = hyperparameters['epsilon_init']
        self.epsilon_decay      = hyperparameters['epsilon_decay']
        self.epsilon_min        = hyperparameters['epsilon_min']
        self.network_sync_rate  = hyperparameters['network_sync_rate']
        self.learning_rate_a    = hyperparameters['learning_rate_a']
        self.discount_factor_g  = hyperparameters['discount_factor_g']
        self.stop_on_reward     = hyperparameters['stop_on_reward']
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.env_make_params    = hyperparameters.get('env_make_params',{})
        
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{hyperparameter_set}.png')
    
    def run(self, is_training=True, render=False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time
            
            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'a') as file:
                file.write(log_message + '\n')
            
        # 只需要用到鸟和管道的位置，不需要lidar
        # env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        env = gym.make(self.env_id, render_mode="human" if render else None, **self.env_make_params)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        
        reward_per_episode = []
        
        # main network: 需要优化的网络
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
        
        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            
            epsilon = self.epsilon_init
            
            # target network: 用来估计 q-value
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
            
            epsilon_history = []
            step_count = 0
            
            best_reward = -99999
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device, weights_only=True))
            policy_dqn.eval()
            

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            
            terminated = False
            episode_reward = 0.0
            
            # 等到回合结束或者 reward 非常大的时候就结束
            while (not terminated and episode_reward < self.stop_on_reward):
                # ε-greedy
                if is_training and random.random() < epsilon:
                    # 随机的 action
                    # (feed the observation to your agent here)
                    '''
                    Action space:
                        0 - do nothing
                        1 - flap
                    '''
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.long, device=device)
                else:
                    # 只是采样，不需要计算梯度
                    with torch.no_grad():
                        # tensor([1, 2, 3, ...]) ==> tensor([[1, 2, 3, ...]])
                        # 添加一个 batch 维度
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
                
                # Processing:
                '''
                new_state:
                    the last pipe's horizontal position           最近的 pipe 的水平位置
                    the last top pipe's vertical position         最近的顶部 pipe 的垂直位置
                    the last bottom pipe's vertical position      最近的底部 pipe 的垂直位置
                    the next pipe's horizontal position           下一个 pipe 的水平位置
                    the next top pipe's vertical position         下一个顶部 pipe 的垂直位置
                    the next bottom pipe's vertical position      下一个底部 pipe 的垂直位置
                    the next next pipe's horizontal position      下下一个 pipe 的水平位置
                    the next next top pipe's vertical position    下下一个顶部 pipe 的垂直位置
                    the next next bottom pipe's vertical position 下下一个底部 pipe 的垂直位置
                    player's vertical position                    小鸟的垂直位置
                    player's vertical velocity                    小鸟的垂直速度
                    player's rotation                             小鸟的旋转角度
                Rewards:
                    +0.1 - every frame it stays alive
                    +1.0 - successfully passing a pipe
                    -1.0 - dying
                    −0.5 - touch the top of the screen
                '''
                new_state, reward, terminated, truncated, info = env.step(action.item())
                
                episode_reward += reward
                
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                
                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                    step_count += 1
                
                state = new_state
            
            reward_per_episode.append(episode_reward)
            
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward - best_reward)/best_reward*100:0.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')
                    
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward
                
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(reward_per_episode, epsilon_history)
                    last_graph_update_time = current_time
            
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)
                    
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)
                    
                    if step_count > self.network_sync_rate:
                        # 同步 main network 和 target network
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0
    
    def save_graph(self, rewards_per_episode, epsilon_history):
        fig = plt.figure(1)
        
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121)
        plt.ylabel("Mean Rewards")
        plt.plot(mean_rewards)
        
        plt.subplot(122)
        plt.ylabel("Epsilon Decay")
        plt.plot(epsilon_history)
        
        plt.subplots_adjust(wspace=1.0, hspace=1.0)
        
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)
    
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # transpose the list of experiences and seperate each element
        states, actions, new_states, rewards, terminations = zip(*mini_batch)
        
        # stack tensors to create batch tensors
        # tensor([[1, 2, 3]])
        states = torch.stack(states)          # batch x num_state
        actions = torch.stack(actions)        # batch x num_action
        new_states = torch.stack(new_states)  # batch x num_state
        rewards = torch.stack(rewards)        # batch x 1
        terminations = torch.tensor(terminations).float().to(device)
        
        with torch.no_grad():
            '''
            target_dqn(new_states) ==> tensor([[1, 2, 3], [4, 5, 6]])
            .max(dim=1)            ==> torch.return_types.max(values=tensor([3,6]), indices=tensor([3, 0, 0, 1]))
            .[0]                   ==> tensor([3, 6])
            '''
            target_q = rewards + ( 1 - terminations ) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
            
        '''
        policy_dqn(states)       ==> 
                                    tensor([[ 0.3238,  0.1208],
                                            [ 0.7442,  0.3304],
                                            [ 0.1455,  0.0016], ...], device='cuda:0', grad_fn=<AddmmBackward0>)  32x2
        actions ==> tensor([1., 1., 0., 1., 0., 1., 1., 0., 1., 0., 0., 1., 1., 1., 0., 0., 1., 1., 1., 0., 1., 1., 0., 0., 1., 0., 0., 1., 1., 0., 1., 0.], device='cuda:0') 32
        actions.unsqueeze(dim=1) ==>
                                    tensor([[1.],
                                            [1.],
                                            [0.], ...], device='cuda:0')  32x1
        policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)) ==>
                                    tensor([[ 0.1208],
                                            [ 0.3304],
                                            [ 0.1455], ...], device='cuda:0', grad_fn=<GatherBackward0>) 32x1
        .squeeze() ==> 32
        '''
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze() # 32
        
        # Compute loss for the whole minibatch
        loss = self.loss_fn(current_q, target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()  # clear gradients
        loss.backward()             # compute gradients
        self.optimizer.step()       # update network parameters (i.e. weights and biases)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test model.")
    parser.add_argument("hyperparameters", help='')
    parser.add_argument("--train", help='Training mode', action='store_true')
    args = parser.parse_args()
    
    dql = Agent(hyperparameter_set=args.hyperparameters)
    
    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)
