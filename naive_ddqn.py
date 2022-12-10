

import queue
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from copy import deepcopy

config = {
    'ExperienceReplayCapacity': 1000000,
    'UserEmbedDim': 16, 
    'MovieEmbedDim':16,
    'NumMoviesInHistory':5,
    'SyncNetsAtIteration' : 100,
    'ActionSpaceSize': 3706,
    'episodes' : 20,
    'ndcg_k':20,
    'test_split_frac':0.3,
    'min_exploit': 5,
    'max_exploit':7
}

"""## Data handling"""

df_ratings = pd.read_csv('/content/drive/MyDrive/IDL_Project/data/ratings.dat', sep='::', names = ['user_id', 'movie_id', 'rating', 'timestamp'], encoding='latin-1', engine='python')
df_ratings.head()

def get_user_timestamp_df(df_ratings):
  # timesteps = config['ActionSpaceSize'] + config["NumMoviesInHistory"]
  # all_user_dfs = []
  # i = 0
  # for user in df_ratings['user_id'].unique():
  #   # print(len(df_ratings[df_ratings['user_id']==user]))
    
  #   user_df = df_ratings[df_ratings['user_id']==user].sort_values(['timestamp','movie_id'])
  #   # print(len(user_df))
  #   user_df['timestamp'] = range(len(user_df))
  #   all_user_dfs.append(user_df)

  # df_full = pd.concat(all_user_dfs, axis=0)
  df_full = df_ratings

  df_full_pivot = df_full.pivot(index = 'user_id', columns ='movie_id', values = 'rating')
  df_full_pivot = df_full_pivot.fillna(0)

  return df_full, df_full_pivot

def create_user_action_space_dict(df_pivot):
  mapping = dict()
  for idx, row in df_pivot.iterrows():
    mapping[idx] = list(row)[config['NumMoviesInHistory']:]

  return mapping

# pass df_full
def create_user_movie_to_rating_dict(df):
  mapping = dict()

  movie_ids = list(df.columns)
  for idx, row in df.iterrows():
    user = row['user_id']
    if user not in mapping:
      mapping[user] = dict()
    
    mapping[user][row['movie_id']] = row['rating']

  return mapping

# def create_user_movie_to_rating_dict(df):
#   mapping = dict()

#   movie_ids = list(df.columns)
#   for idx, row in df.iterrows():
#     user = row['user_id']
#     if user not in mapping:
#       mapping[user] = dict()
    
#     mapping[user][row['movie_id']] = row['rating']

#   return mapping

df_full, df_pivot = get_user_timestamp_df(df_ratings)

df_pivot.head(10)

"""## Experience Replay"""

class ExperienceReplay:
  def __init__(self, capacity):
    # Max length of 
    self.capacity = capacity
    # List of tuples
    self.memory_buffer = []


  def store_data(self,data):
    if len(self.memory_buffer)==self.capacity:
      self.memory_buffer.pop(0)
    self.memory_buffer.append(data)

  def get_data(self):
    idx = np.random.choice(len(self.memory_buffer),1, replace=False)[0]

    return self.memory_buffer[idx]

  def print_data(self):
    print(self.memory_buffer)

"""## DQN"""

def calc_size_after_conv(input_dim, out_channel, kernel_size, stride):
  _, width = input_dim

  out_width = ((width - kernel_size)//stride)+1

  return out_channel, out_width

class DQN(nn.Module):
  def __init__(self, input_dim, out_dim):
      super(DQN, self).__init__()

      self.conv1 = nn.Conv1d(1, 32, kernel_size=8, stride=4)
      out_channel, out_width = calc_size_after_conv((1,input_dim), 32, 8, 4)

      self.conv2 = nn.Conv1d(out_channel, 64, kernel_size=4, stride=2)
      out_channel, out_width = calc_size_after_conv((out_channel,out_width), 64, 4, 2)

      self.conv3 = nn.Conv1d(out_channel, 64, kernel_size=3, stride=1)
      out_channel, out_width = calc_size_after_conv((out_channel,out_width), 64, 3, 1)

      self.flatten = nn.Flatten()
      out_width = out_channel*out_width

      self.fc4 = nn.Linear(out_width, 512)
      # Cannot use batchnorm for SGD since mean(x) = x and norm will be 0. No learning will take place - https://stackoverflow.com/questions/48343857/whats-the-reason-of-the-error-valueerror-expected-more-than-1-value-per-channel
      # self.batchnorm = nn.BatchNorm1d(512)
      self.dropout = nn.Dropout(p=0.2)
      self.fc5 = nn.Linear(512, out_dim)

  def forward(self, x):
    x = F.tanh(self.conv1(x))
    x = self.dropout(x)
    x = F.tanh(self.conv2(x))
    x = self.dropout(x)
    x = F.tanh(self.conv3(x))
    x = self.dropout(x)
    x = self.flatten(x)
    x = self.fc4(x)
    x = F.tanh(x)
    x = self.dropout(x)

    result = self.fc5(x)
  
    return result

"""## Embedding"""

class Embedding(nn.Module):
  def __init__(self,num_users,user_embed_dim,num_movies,movie_embed_dim):
    super(Embedding, self).__init__()
    
    self.user_embedding = nn.Embedding(num_users,user_embed_dim)
    self.movie_embedding = nn.Embedding(num_movies,movie_embed_dim)

  def forward(self, user,movie_history):
    user_embed = self.user_embedding(user)
    movie_embed = self.movie_embedding(movie_history)
    movie_embed = torch.flatten(movie_embed).unsqueeze(0)
    
    state = torch.cat([user_embed, movie_embed], dim=1)
    return state

"""## Agent"""

class Agent:
  def __init__(self, memory_capacity, \
               num_users, num_movies, reward_dict,action_space_dict, df,\
               lr=10e-3, epsilon=0.1, gamma=0.9, min_reward = 1, max_reward = 5):
    super(Agent, self).__init__() 

    # Initialize memory
    self.replay_memory = ExperienceReplay(memory_capacity)

    # Variables for the run
    self.n_states = config['NumMoviesInHistory'] #number of movies in movie history per user
    self.sync_nets_at_iteration = config['SyncNetsAtIteration'] #number of iterations post which we sync the two DQNs
    self.gamma = gamma #discount factor
    self.epsilon = epsilon #epsilon greedy
    self.lr = lr #learning rate

    #Embedding initialization
    self.embedding_layer = Embedding(num_users, config['UserEmbedDim'], num_movies, config['MovieEmbedDim'])

    self.state_size = config['UserEmbedDim'] + config['MovieEmbedDim'] * config['NumMoviesInHistory']
    self.action_space_size = config['ActionSpaceSize']

    self.eval_net = DQN(self.state_size, self.action_space_size)
    self.target_net = DQN(self.state_size, self.action_space_size)
    self.update_target_net()

    self.criterion = nn.MSELoss()

    self.optimizer = torch.optim.AdamW(self.eval_net.parameters(), lr=self.lr)

    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',patience=4)

    self.reward_dict = deepcopy(reward_dict) # Maps (user,movie)->rewards

    self.action_space_dict = deepcopy(action_space_dict) # Maps user-> action_space

    self.num_users = len(df)
    
    #if all rewards are positive, the agent will take every action as favourable.
    self.zero_center_rewards(min_reward, max_reward)
    
    # Initialization
    self.init_step_count()
    self.init_replay_memory(df)

  def init_replay_memory(self,df):
    # df is a pivot table of userXtimesteps
    for idx, row in df.iterrows():
      user = idx
      
      movie_index = np.argsort(row)[:self.n_states]
      all_movies = list(df.columns)
      movies = [all_movies[i] for i in movie_index]

      # Get Q value for user, movies
      q_values = self.get_q_value(user,movies, net='eval')

      # Choose action; epsilon greedy
      _,_,action = self.select_action(user, q_values[0])

      # Execute action
      reward,new_movies = self.execute_action(user, movies, action)

      # Tuple for Experience Replay
      memory_tup = (user, movies,action, reward, new_movies)

      # Add to experience replay
      self.replay_memory.store_data(memory_tup)

  def init_step_count(self):
      self.step_count = {k:0 for k in range(1,self.num_users+1)}

  def get_q_value(self, user, movies, net='eval'):
    # This gives us a vector of size (1X96) 96 in this case can change
    # Pass user features as a list, since we use only index at first, we make a 1 length list
    state_embed = self.embedding_layer(torch.tensor([user]),torch.tensor(movies))

    # We need to add a dimension for batch size for the model
    if net=='eval':
      q_values = self.eval_net(state_embed.unsqueeze(0))
    elif net=='target':
      q_values = self.target_net(state_embed.unsqueeze(0))

    return q_values

  def select_action(self, user,q_values):
    rand_n = np.random.uniform()
    
    if rand_n > self.epsilon:
      action_idx = torch.argmax(q_values)
    else:
      action_idx = np.random.randint(0,self.action_space_size)

    q_value = q_values[action_idx]

    action = self.action_space_dict[user][action_idx]
    return q_value, action_idx, action
    
  def execute_action(self, user, movies, action):
    new_movies = movies[1:]
    new_movies.append(action)

    reward = min(1,self.reward_dict[user].get(action,0))

    self.step_count[user] += 1

    return reward, new_movies

  def zero_center_rewards(self, min_reward, max_reward):
    mean = (max_reward + min_reward)/2
    for user,map in self.reward_dict.items():
      for movie,reward in map.items():
        self.reward_dict[user][movie] = self.reward_dict[user][movie] - mean

  def zero_center_rewards_with_map(self, mapping):
    for user,map in self.reward_dict.items():
      for movie,reward in map.items():
        self.reward_dict[user][movie] = mapping[self.reward_dict[user][movie]]

  def get_data_from_replay(self):
    return self.replay_memory.get_data()

  def put_data_into_replay(self, user, movies, action, reward, new_movies):
    # Tuple for Experience Replay
    memory_tup = (user, movies,action, reward, new_movies)

    # Add to experience replay
    self.replay_memory.store_data(memory_tup)

  def update_target_net(self):
    #load the state of eval net and transfer to target net
    self.target_net.load_state_dict(self.eval_net.state_dict())

  def update_eval_net(self, memory_tuple):
    self.optimizer.zero_grad()

    #fetch new datapoint from experience replay
    user, movies, _, reward, new_movies = memory_tuple

    #pass current state to eval net and retrieve q_values
    q_values = self.get_q_value(user, movies, net="eval")[0]

    #select action and associated q value using epsilon greedy on the retrieved q_values
    q_value, action_idx, action = self.select_action(user, q_values)

    #pass new_movies through target net and get q_values
    next_q_values = self.get_q_value(user, new_movies, net="target")
    next_q_value = torch.max(next_q_values)

    #we don't call execute action here since no action is taken. We are only learning the q value for loss calculation
    #hence, no change to self.step_count

    if(self.step_count[user] == config["ActionSpaceSize"]):
      target_q_value = reward
    else:
      target_q_value = reward + (self.gamma * next_q_value)
    
    loss = self.criterion(q_value, target_q_value)
    loss.backward()

    return loss.item()

  def train_episode(self, steps=0):
    self.eval_net.train()
    self.target_net.eval()

    cumulative_loss = []
    self.init_step_count()

    while(min(self.step_count.values()) < config['min_exploit']):
      #sample memory tuple
      memory_tuple = self.get_data_from_replay()

      if self.step_count[memory_tuple[0]]==config['max_exploit']:
        continue

      #update eval
      cumulative_loss.append(self.update_eval_net(memory_tuple)) 
      
      # Unpack tuple
      (user, movies, action, reward, new_movies) = memory_tuple

      #set state to new state from memory tuple
      movies = new_movies

      #get new datapoint to populate experience replay by running eval_net
      q_values = self.get_q_value(user, movies, net='eval')[0]

      steps += 1

      # Get next state using eval
      self.put_data_into_replay(user, movies, action, reward, new_movies)
      q_value, action_idx, new_action = self.select_action(user, q_values)
      new_reward, new_movies = self.execute_action(user, movies, action)

      # Add new point in experience replay
      self.put_data_into_replay(user, movies, new_action,new_reward,new_movies)

      # self.sync_nets_at_iteration
      if steps%self.sync_nets_at_iteration==0:
        self.update_target_net()

    print('Loss:', np.mean(cumulative_loss))
    print('Min explored:', min(self.step_count.values()))
    print('Max explored:', max(self.step_count.values()))

  
    return np.mean(cumulative_loss), steps

  def predict(self, df_test, min_relevance, max_relevance):
    results5 = []
    results10 = []
    results20 = []

    for idx, row in df_test.iterrows():
      user = idx
      movie_index = np.argsort(row)[:self.n_states]
      all_movies = list(df_test.columns)
      movies = [all_movies[i] for i in movie_index]

      # Get Q value for user, movies
      q_values = self.get_q_value(user,movies, net='eval')[0]
      
      for index in movie_index:
        q_values[index] = 0

      sorted_movie_indices = torch.argsort(q_values).tolist()
      top_k_indices = sorted_movie_indices[:config['ndcg_k']]
      
      top_k_movie_ids = [self.action_space_dict[user][x] for x in top_k_indices]

      top_k_movie_rewards = [self.reward_dict[user].get(x,0) for x in top_k_movie_ids]

      sorted_top_k_movie_rewards = sorted(top_k_movie_rewards, reverse=True)

      ndcg5 = ndcg_at_k(top_k_movie_rewards,5)
      ndcg10 = ndcg_at_k(top_k_movie_rewards,10)
      ndcg20 = ndcg_at_k(top_k_movie_rewards,20)


      results5.append(ndcg5)
      results10.append(ndcg10)
      results20.append(ndcg20)

    print('5:',np.mean(results5), '10:',np.mean(results10), '20:',np.mean(results20))
    return np.mean(results5), np.mean(results10), np.mean(results20)

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
    # print(r)
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg

movie_list = df_pivot.columns
num_movies = max(df_pivot.columns)
print(num_movies, df_pivot.columns)

all_movies = list(df_pivot.columns)
action_space_dict = {k:all_movies for k in df_pivot.index}
user_movie_to_reward_dict = create_user_movie_to_rating_dict(df_full)

split = int((1-config['test_split_frac'])*len(df_pivot))
print(split)

train_df = df_pivot.iloc[:split]
val_df = df_pivot.iloc[split:]

a = Agent(config['ExperienceReplayCapacity'], \
          len(df_pivot)+1, num_movies+1, reward_dict=user_movie_to_reward_dict, df = train_df, \
          action_space_dict = action_space_dict,\
          lr=1e-3, epsilon=0.1, gamma=0.9)

steps = 0

for e in range(config['episodes']):
  print('Episode:', e)
  loss, steps = a.train_episode(steps)
  print('Val ndcg:')
  res_mean, res,_ = a.predict(val_df, 1,5)
  print('LR:', float(a.optimizer.param_groups[0]['lr']))
  a.scheduler.step(loss)
  print('\n\n\n')

