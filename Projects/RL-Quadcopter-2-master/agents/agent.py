import numpy as np
from task import Task

class agent():
    def __init__(self,task,start_alpha = 0.3, start_gamma = 0.9, start_epsilon = 0.5):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low
        self.epsilon = start_epsilon
        self.gamma = start_gamma
        self.alpha = start_alpha
        self.q = np.zeros(shape = (self.state_size,self.action_size))
        # Set up policy pi, init as equiprobable random policy
        self.pi = np.zeros_like(self.q)
        for i in range(self.pi.shape[0]): 
            for a in range(self.action_size): 
                self.pi[i,a] = 1/self.action_size
                
                
    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state
    
    
    def step(self, action, state ,next_state, reward, done):
        # Save experience / reward
        self.count += 1
        self.q[state,action] = self.q[state,action] + self.alpha * (reward + self.gamma* np.dot(self.pi[next_state,:],self.q[next_state,:])-self.q[state,action])
        
    def act(self, state):
        
        action = np.argmax(np.cumsum(self.pi[state,:]) > np.random.random())
        return action   
    def learn(self,state):
        best_action = np.random.choice(np.where(self.q[state] == max(self.q[state]))[0])
        for i in range(self.action_size): 
            if i == best_action:      self.pi[state,i] = 1 - (self.action_size-1)*(self.epsilon / self.action_size)
            else:                self.pi[state,i] = self.epsilon / self.action_size
                
    