# https://medium.freecodecamp.org/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682

import numpy as np
import random
import torch

class SumTree:
    
    data_pointer = 0
    
    def __init__(self,capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity-1)
        self.data = np.zeros(capacity,dtype=object)
        self.n_entries = 0
    
    def add(self,priority,data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        
        self.update(tree_idx,priority)
        
        self.data_pointer += 1
        
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self,tree_idx,priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
    
    def get_value(self,v):
        
        parent = 0
        
        while True:
            left = 2*parent + 1
            right = left + 1
            
            if left >= len(self.tree):
                idx = parent
                break
                
            else:
                if v <= self.tree[left]:
                    parent = left
                else:
                    v -= self.tree[left]
                    parent = right
       
        data_idx = idx - self.capacity + 1
        
        return idx,self.tree[idx],self.data[data_idx]
    
    def total_priority(self):
        return self.tree[0]
    
