
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

import random

import torch
import torch.nn as nn
from tqdm import tqdm

import gc


df_ratings = pd.read_csv('/content/drive/MyDrive/IDL_Project/data/ratings.dat', sep='::', names = ['user_id', 'movie_id', 'rating', 'timestamp'], encoding='latin-1', engine='python')
df_ratings.head()

df_movies = pd.read_csv('/content/drive/MyDrive/IDL_Project/data/movies.dat', sep='::', names = ['movie_id', 'name', 'genres'], encoding='latin-1', engine='python')
df_movies.head()

df_users = pd.read_csv('/content/drive/MyDrive/IDL_Project/data/users.dat', sep='::', names = ['user_id', 'gender', 'age', 'occupation', 'zipcode'], encoding='latin-1', engine='python')
df_users.head()

sparse_dataframe = df_ratings.pivot(index = 'user_id', columns ='movie_id', values = 'rating').fillna(0) # Covert to sparse representation
sparse_dataframe.head()

sparse_matrix = sparse_dataframe.values # User ID, Movie ID -> value is rating

train_records = {}

#sorting the movies acc. to ratings in descending order, also shifting start of indices from 0 to 1
indices = np.flip(sparse_dataframe.values.argsort(axis = 1), axis = 1) + 1

for i in range(sparse_dataframe.shape[0]):
  user_ratings = sparse_dataframe.values[i]
  #taking only 3+ ratings
  temp_records = (indices[i][sparse_matrix[i] >= 3]).tolist()
  #taking only those datapoints that have more than 50 ratings
  if len(temp_records)>=50:
    train_records[i+1] = temp_records

"""### DQN Implementation"""

class Embedding(nn.Module):
  def __init__(self, num_users, num_movies, embed_dim):
    
    super(Embedding, self).__init__()

    #creating embedding objects with dimensions 1 + num_users (because of shift of start of indices from 0 to 1) and embed_dim
    self.user_embed = nn.Embedding(1 + num_users, embed_dim)
    self.movie_embed = nn.Embedding(1 + num_movies, embed_dim)
    self.action_embed = nn.Embedding(1 + num_movies, embed_dim)


  def forward(self, user_ids, movie_history, actions): # (u, s, a)
        embedded_user = self.user_embed(user_ids).squeeze(1)     
        embedded_movie = torch.flatten(self.movie_embed(movie_history), start_dim = -2)
        embedded_action = self.action_embed(actions).squeeze(1)    
        # print(embedded_user.shape, embedded_movie.shape, embedded_action.shape)
        final_embedding = torch.cat([embedded_user, embedded_movie, embedded_action], dim = -1) # combine user, movie and action embeddings
        return final_embedding #state

class DQN(nn.Module):
  def __init__(self, embed_size):

    super(DQN, self).__init__()

    self.state_size = embed_size * 7 # user id + 5 movies + action

    self.model = nn.Sequential(
        nn.Linear(self.state_size, 128), # (action_size, embed*7) user id, 5 movies, action
        nn.ReLU(),
        nn.Dropout(p = 0.1),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Dropout(p = 0.1),
        nn.Linear(128, 1)
    )

  def forward(self, state):
    '''
    Input : Model state i.e [user_embedding, movie_embeddings]
    Output : Recommended Action
    '''

    return self.model(state) # (action_size, 1)

class Agent():

  def __init__(self, num_users, num_movies, action_size, embed_dim=256 , lr=0.001, epsilon=0.9, gamma=0.9, memory_capacity=30):
      
      super(Agent, self).__init__() 
      self.embed_dim = embed_dim
      self.lr = lr
      self.epsilon = epsilon
      self.gamma = gamma
      self.action_size = action_size
      self.memory_capacity = memory_capacity
      self.n_states = 5 
      self.iteration = 5 
      self.batch_size = 5

      #state size is embed_dim * (num_states+2) because ???
      self.state_size = embed_dim * (self.n_states+2)
      self.get_Embedding = Embedding(num_users, num_movies, embed_dim)

      self.eval_net = DQN(embed_dim)
      self.target_net = DQN(embed_dim)

      self.replay_memory = np.zeros((memory_capacity, self.n_states * 2 + self.action_size + 3)) # (new state + state) + action_space + (action + user + reward)
      # print(self.replay_memory.shape)
      self.replay_memory_pointer = 0

      self.candidate_actions = []
      self.action_space = []

      if torch.cuda.is_available():
          self.eval_net.cuda()
          self.target_net.cuda()
          self.loss_func = nn.MSELoss().cuda()
      else:
          self.loss_func = nn.MSELoss()

      self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)

      self.learn_step_counter = 0

  def create_action_space(self, list_of_movies):
    self.action_space = list_of_movies[ :self.action_size]
    self.candidate_actions = list_of_movies[self.action_size: ]

  def update_action_space(self, action):
    self.action_space.remove(action)
    add = self.candidate_actions.pop(0)
    self.action_space.append(add)


  def get_action(self, user, state, train = True): # agent.get_action(user, state)
    if train:
        if np.random.uniform() < self.epsilon:
            action = self.action_space
            u = torch.full([len(self.action_space), 1], user)
            s = torch.tensor(state).view(-1, self.n_states).repeat(self.action_size, 1)
            a = torch.tensor(action).view(self.action_size, -1)
            # Get embedding
            final_embed = self.get_Embedding(u, s, a)
            res = self.eval_net(final_embed)
            _, indices = torch.max(res, 0)
            return action[indices]
        else:
            return random.choice(self.action_space)
    else:
        action = self.action_space
        u = torch.full([len(self.action_space), 1], user)
        s = torch.tensor(state).view(-1, self.n_states).repeat(self.action_size, 1)
        a = torch.tensor(action).view(self.action_size, -1)
        final_embed = self.get_Embedding(u, s, a)
        res = self.eval_net(final_embed)
        _, indices = torch.max(res, 0)
        return action[indices]


  def add_to_replay_memory(self, user, state, action, reward, newstate, current_action_space):
    u = np.array(user)
    s = np.array(state)
    s_ = np.array(newstate)
    a_s = np.array(current_action_space)
    transition = np.hstack((u, s, action, reward, s_, a_s))   
    index = self.replay_memory_pointer % self.memory_capacity
    # print(self.replay_memory.shape, transition.shape)
    self.replay_memory[index, :] = transition
    self.replay_memory_pointer += 1 



  def update(self):

    if self.learn_step_counter % self.iteration == 0:
        self.target_net.load_state_dict(self.eval_net.state_dict()) # Dueling

    self.learn_step_counter += 1

    random_replay = np.random.choice(self.memory_capacity, self.batch_size) 
    b_memory = self.replay_memory[random_replay, :]
    b_u = b_memory[:, :1].astype(int)
    b_s = torch.LongTensor(b_memory[:, 1:self.n_states+1])
    b_a = torch.LongTensor(b_memory[:, self.n_states+1: self.n_states + 2])
    b_r = torch.LongTensor(b_memory[:, self.n_states + 2: self.n_states + 3])
    b_s_ = torch.LongTensor(b_memory[:, self.n_states + 3:self.n_states + 3+self.n_states])
    b_a_sp = torch.LongTensor(b_memory[:, -self.action_size:])   # action space with state b_s_
    
    eval_embed = self.get_Embedding(torch.tensor(b_u), b_s, b_a)
    q_eval = self.eval_net(eval_embed)
    q_next = []
    
    for i in range(self.batch_size):
        u = b_u[i][0] # (1,1)
        u = torch.full([self.action_size, 1], u) # (20, 1)
        s_ = b_s_[i].view(-1, self.n_states)
        s_ = s_.repeat(self.action_size, 1) # (20, 5)
        a_ = b_a_sp[i].view(-1, self.action_size)
        a_ = a_.t() # (20, 1)
        final_embed = self.get_Embedding(u, s_, a_)
        res = self.target_net(final_embed).detach()
        value, _ = torch.max(res, 0)
        q_next.append(value)

    q_next = torch.tensor(q_next)
    q_target = b_r + self.gamma * q_next.view(self.batch_size, 1)

    loss = self.loss_func(q_eval, q_target)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    del b_a_sp
    gc.collect()

class Environment():
  
  def get_reward(self, rec_list, action, train_records):
    if action in train_records:
        rel = 1
        r = np.subtract(np.power(2, rel), 1) / np.log2(len(rec_list) + 1)
    else:
        rel = 0
        r = 0
    return r

  def get_next_state(self, state, action, train_records): # state -> list of 5 movie ids, action -> recommended movie id
    if action in train_records:
        state.pop(0)
        state.append(action)
        s_next = state
    else:
        s_next = state
    return s_next


train_data = sparse_dataframe.iloc[:4228, :]
test_data = sparse_dataframe.iloc[4228:, :]

train_user_set = set(train_records.keys()).intersection(set(train_data.index))
test_user_set = set(train_records.keys()).intersection(set(test_data.index))

movie_set = set(df_ratings["movie_id"].unique().tolist())
rewards = []

predicted_movies = {}

u_action_space_train = {}
for user in train_user_set:
  movies_watched = train_records[user][5:].copy()
  np.random.shuffle(movies_watched)
  k_movies = movies_watched[:50]
  u_action_space_train[user] = k_movies

agent = Agent(num_users = len(df_ratings["user_id"].unique().tolist()), num_movies = max(movie_set), action_size = 20)
env = Environment()

epoch = 0
total_episode_reward = 0
#epoch
agent.replay_memory_pointer = 0

for i in range(3):
  for idx, user in tqdm(enumerate(train_user_set)):

    if len(train_records[user]) < 5:
      continue 

    episode_reward = 0
    recommended_movie = []
    np.random.shuffle(train_records[user])
    state = train_records[user][0 : 5]
    agent.candidate_actions = []
    agent.action_space = []
    agent.create_action_space(u_action_space_train[user])
    print("epoch: ", i + 1, ", user : ", user)
    for t in range(20):
      action = agent.get_action(user, state, True)
      recommended_movie.append(action)
      agent.update_action_space(action)
      next_state = env.get_next_state(state, action, train_records[user])
      reward = env.get_reward(recommended_movie, action, train_records[user])
      agent.add_to_replay_memory(user, state, action, reward, next_state, agent.action_space)
      episode_reward += reward
      if agent.replay_memory_pointer > agent.memory_capacity:
          agent.update()
      state = next_state

      total_episode_reward += episode_reward

      if i % 10 == 0:
          total_episode_reward = 0

def dcg_at_k(r, k):
    '''
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
    Returns:
        Discounted cumulative gain
    '''
    assert k >= 1
    r = np.asfarray(r)[:k] != 0
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.0


def ndcg_at_k(r, k):
    '''
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
    Returns:
        Normalized discounted cumulative gain
    '''
    assert k >= 1
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg

preds = {}

u_action_space_test = {}
for user in test_user_set:
  k_movies = list(np.random.choice(list(movie_set), 50, replace=False))
  u_action_space_test[user] = k_movies

for user in tqdm(test_user_set):

  recommended_movies = []
  state = train_records[user][-5:]
  agent.create_action_space(u_action_space_test[user])

  for t in range(20):
    action = agent.get_action(user, state, False)
    recommended_movies.append(action)
    agent.update_action_space(action)
    next_state = env.get_next_state(state, action, train_records[user])
    state = next_state

  preds[user] = recommended_movies.copy()


u_binary = []
u_result = []
res = preds.copy()
record = {}
u_record = {}
u_test = []
for u in test_user_set:
    u_test.append(u)
    u_record[u] = [u] + res[u]
    u_result.append(u_record[u])
    preds[u] = [1 if i in train_records[u] else 0 for i in preds[u]]
    record[u] = [u] + preds[u]
    u_binary.append(record[u])


for k in [1, 5, 10, 20]:
    tmp_preds = preds.copy()        
    tmp_preds = {key: rank_list[:k] for key, rank_list in tmp_preds.items()}

    ndcg_k = np.mean([ndcg_at_k(r, k) for r in tmp_preds.values()])

    if k == 5:
        print(f'\nNDCG@{k}: {ndcg_k:.4f}')

    if k == 10:
        print(f'\nNDCG@{k}: {ndcg_k:.4f}')

    if k == 20:
        print(f'\nNDCG@{k}: {ndcg_k:.4f}')

# recommended_movies