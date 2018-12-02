# adapted from https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20%28%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay%29.ipynb

from SumTree import SumTree
from collections import namedtuple
import numpy as np
import random

import torch
from torch import nn,optim
import torch.nn.functional as F

from model import QNetwork

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lr = 0.001
tau = 1e-3
UPDATE_EVERY = 4
gamma = 0.99
batch_size = 64
capacity = int(1e5)

class Agent:
    
    def __init__(self,state_size,action_size,seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self.qnetwork_local = QNetwork(state_size,action_size,seed).to(device)
        self.qnetwork_target = QNetwork(state_size,action_size,seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=lr)
        
        self.memory = PRIOREPLAYBUFFER(action_size,capacity,batch_size,seed)
        self.t_step = 0
    
    def step(self,state,action,reward,next_state,done):
        self.memory.add(state,action,reward,next_state,done)
        
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if self.memory.memory.n_entries > batch_size:
                idxs,weights,experiences = self.memory.sample()
                self.learn(experiences,weights,idxs,gamma)
    
    def act(self,state,eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def learn(self,experiences,weights,idxs,gamma):
        states,actions,rewards,next_states,dones = experiences
        
        act_idxs = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        Q_tar_vals = self.qnetwork_target(next_states).detach().gather(1,act_idxs)
        
        Q_tar = rewards + (gamma*(1-dones)*Q_tar_vals)
        
        Q_est = self.qnetwork_local(states).gather(1,actions)
        
        errors = F.l1_loss(Q_tar,Q_est,reduce=False)
        
        weights = torch.from_numpy(weights).float().to(device)
        loss = (weights * F.mse_loss(Q_tar,Q_est,reduce=False)).mean()
        
        
        self.memory.batch_updates(idxs,errors.cpu().data.numpy())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.soft_update(self.qnetwork_local,self.qnetwork_target,tau)
    
    def soft_update(self,local_net,target_net,tau):
        for lp,tp in zip(local_net.parameters(),target_net.parameters()):
            tp.data.copy_(tau * lp.data + (1-tau) * tp.data)
            
class PRIOREPLAYBUFFER:
    
    alpha = 0.6
    beta = 0.4
    beta_update = 0.001
    e = 0.01
    
    def __init__(self,action_size,capacity,batch_size,seed):
        self.action_size = action_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory = SumTree(capacity)
        self.experience = namedtuple('Experience',field_names=['state','action','reward','next_state','done'])
    
    def add(self,state,action,reward,next_state,done):
        e = self.experience(state,action,reward,next_state,done)
        
        max_priority = np.max(self.memory.tree[-self.memory.capacity:])
        if max_priority == 0:
            max_priority = 1
        self.memory.add(max_priority,e)
    
    def sample(self):
        idxs = []
        priorities = []
        experiences = []
        
        seg = self.memory.total_priority() / self.batch_size
        self.beta = np.min([1,self.beta + self.beta_update])
        
        for i in range(self.batch_size):
            a = seg * i
            b = seg * (i+1)
            val = np.random.uniform(a,b)
            
            idx,priority,exp = self.memory.get_value(val)
            
            priorities.append(priority)
            idxs.append(idx)
            experiences.append(exp)
          
        prob = np.array(priority) / self.memory.total_priority()
        weights = np.power(prob*self.memory.n_entries,-self.beta)
        weights /= weights.max()
           
        states = torch.from_numpy(np.vstack(e.state for e in experiences if e is not None)).float().to(device)
        actions = torch.from_numpy(np.vstack(e.action for e in experiences if e is not None)).long().to(device)
        rewards = torch.from_numpy(np.vstack(e.reward for e in experiences if e is not None)).float().to(device)
        next_states = torch.from_numpy(np.vstack(e.next_state for e in experiences if e is not None)).float().to(device)
        dones = torch.from_numpy(np.vstack(e.done for e in experiences if e is not None).astype(np.uint8)).float().to(device)
        
        return np.array(idxs),np.array(weights,dtype=np.float32),(states,actions,rewards,next_states,dones)
    
    def batch_updates(self,idxs,errors):
        for idx,error in zip(idxs,errors):
            error = min(1,error + self.e)
            ps = np.power(error,self.alpha)
            self.memory.update(idx,ps)