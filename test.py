import os
import pyautogui as pg
from PIL import Image

# CONSTANTS ====

# Directories
ROOT = os.getcwd()
IMGS = f'{ROOT}/pics'

# Training settings
MEM_SIZE = 50_000
MEM_SIZE_MIN = 1_000
LEARN_RATE = 0.001
BATCH_SIZE = 64
DISCOUNT = 0.99

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

# Environment settings
EPISODES = 20_000

TILES = {
    'U': f'{IMGS}/unsolved.png',
    '0': f'{IMGS}/zero.png',
    '1': f'{IMGS}/one.png',
    '2': f'{IMGS}/two.png',
    '3': f'{IMGS}/three.png',
    '4': f'{IMGS}/four.png',
    '5': f'{IMGS}/five.png',
    '6': f'{IMGS}/six.png',
    '7': f'{IMGS}/seven.png',
    '8': f'{IMGS}/eight.png',
    'M': f'{IMGS}/mine.png',
}
# ====

class Minesweeper:
    def __init__(self):
        pg.click((1,1))
        self.reset()

        # Minesweeper Parameters
        self.TILES = TILES
        self.loc = self.find_board()
        self.state = self.get_state(self.loc)
        self.nrows = sum(1 for i in self.state if i['coord'][0]==self.state[0]['coord'][0])
        self.ncols = sum(1 for i in self.state if i['coord'][1]==self.state[0]['coord'][1])
        self.ntiles = self.nrows*self.ncols
        self.scaled = self.scale_state(self.state)
        self.n_solved_ = 0

        # Deep Q-learning Parameters
        self.discount = DISCOUNT
        self.epsilon = epsilon
        self.model = create_dqn(LEARN_RATE, self.scaled.shape, self.ntiles)

        # target model - this is what we predict against every step
        self.target_model = self.create_dqn(LEARN_RATE, self.scaled.shape, self.ntiles)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=MEM_SIZE)
        self.target_update_counter = 0

    def reset(self):
        pg.press('f2')
        # restart game
        '''try:
            loc = pg.locateOnScreen(f'{IMGS}/oof.png')
        except:
            loc = pg.locateOnScreen(f'{IMGS}/reset.png')
        pg.click(loc, duration=0.3)'''

    def find_board(self):
        # obtain coordinates for Minesweeper board
        modes = ['beginner', 'intermediate', 'expert']
        boards = [pg.locateOnScreen(f'{IMGS}/{mode}.png') for mode in modes]

        assert boards != [None, None, None], 'Minesweeper board not detected on screen'

        for x in boards:
            if x != None:
                board = x

        return board

    def get_board(self, bbox):
        '''
        Gets the state of the board as a dictionary of coordinates and values,
        ordered from left to right, top to bottom
        '''
        all_tiles = [[t, list(pg.locateAllOnScreen(self.TILES[t], region=bbox))] for t in self.TILES]

        tiles = []
        for value, coords in all_tiles:
            for coord in coords:
                tiles.append({'coord': (coord[0], coord[1]), 'value': value})

        tiles = sorted(tiles, key=lambda x: (x['coord'][1], x['coord'][0]))

        return tiles

    def get_state(self, board):
        '''
        Gets the numeric image representation state of the board.
        This is what will be the input for the DQN.
        '''
        board_2d = [t['value'] for t in board]
        board_2d = np.reshape(board_2d, (self.nrows, self.ncols, 1))

        state = np.zeros((self.nrows, self.ncols, 1))
        state[board_2d=='U'] = -1
        state[board_2d=='0'] = 0

        num_tiles = ~np.logical_or(board_2d == "U", board_2d == "0")
        state[num_tiles] = board_2d[num_tiles].astype(int) / 8

        return state

    def step(self, action_index):

        done = False

        # number of solved tiles prior to move (initialized at 0)
        self.n_solved = self.n_solved_

        pg.click(self.board[action_index]['coord']) #, duration=0.1

        if pg.locateOnScreen(f'{IMGS}/oof.png', region=self.loc) != None: # if lose
            reward = -1
            done = True
            self.n_solved_ = 0

        elif pg.locateOnScreen(f'{IMGS}/gg.png', region=self.loc) != None: # if win
            reward = 1
            done = True
            self.n_solved_ = 0

        else:
            self.board = self.get_board(self.loc)
            self.state = self.get_state(self.board)

            # update number of solved tiles
            self.n_solved_ = self.ntiles - np.sum(self.state == -1)

            if self.n_solved_ > self.n_solved: # if progress
                reward = 0.9

            elif self.n_solved_ == self.n_solved: # if no progress
                reward = -0.9

        return self.state, reward, done

    def get_action(self, state):
        board = self.state.reshape(1, self.ntiles)
        unsolved = [i for i, x in enumerate(board[0]) if x==-1]

        rand = np.random.random() # random value b/w 0 & 1

        if rand < self.epsilon: # random move (explore)
            move = np.random.choice(unsolved)
        else:
            moves = self.model.predict(np.reshape(self.state, (1, self.nrows, self.ncols, 1)))
            moves[board!=-1] = 0
            move = np.argmax(moves)

        return move

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, done, episode):
        if len(self.replay_memory) < MEM_SIZE_MIN:
            return

        batch = random.sample(self.replay_memory, BATCH_SIZE)

        current_states = np.array([transition[0] for transition in batch])
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in batch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for i, (current_state, action, reward, new_current_state, done) in enumerate(batch):
            if not done:
                max_future_q = np.max(future_qs_list[i])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[i]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X), np.array(y), batch_size=BATCH_SIZE, shuffle=False)#,\
                       #verbose=0, shuffle=False, callbacks=[self.tensorboard]\
                       #if terminal_state else None)

        # updating to determine if we want to update target_model yet
        if done:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        # decay epsilon
        self.epsilon = max(EPSILON_MIN, self.epsilon*EPSILON_DECAY)
