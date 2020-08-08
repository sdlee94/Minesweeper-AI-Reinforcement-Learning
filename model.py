from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np

# handles storage of the state-action reward and state-transition arrays
class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size # max size of memory
        self.input_shape = input_shape # input shape of the environment
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, new_state, action, reward, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)

        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, new_states, actions, rewards, terminal

def build_dqn(lr, n_actions, input_dims, fcl_dims, fc2_dims):
    model = Sequential([
                Dense(fc1_dims, input_shape=(input_dims, )),
                Activation('relu'),
                Dense(fc2_dims),
                Activation('relu'),
                Dense(n_actions)])

    model.compile(optimizer=Adam(lr=lr), loss='mse')

    return model

class Agent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_decay=0.996, min_epsilon=0.01,
                 mem_size=1_000_000, fname='dqn_model.h5')
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma # discount factor
        self.epsilon = epsilon # affects the probabilty of acting randomly (exploration)
        self.epsilon_dec - epsilon_decay
        self.epsilon_min = eps_end
        self.batch_size = batch_size
        self.model_file = fname

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval - build_dqn(alpha, n_actions, input_dims, 256, 256)

    def remember(self, state, action, reward, new_state, done)
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random() # random value b/w 0 & 1
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        action_values = np.array(self.action_space, dtype=np.float32)
        action_indices = np.dot(action, action_values)

        q_eval = self.q_eval.predict(state)
        q_next = self.q_eval.predict(new_state)

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, action_indices] = reward + \
            self.gamma*np.max(q_next, axis=1)*done

        _ = self.q_eval.fit(state, q_target, verbose=0)

        self.epsilon = self.epsilon*self.epsilon_decay \
            if self.epsilon > self.epsilon_min else self.epsilon_min

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)
