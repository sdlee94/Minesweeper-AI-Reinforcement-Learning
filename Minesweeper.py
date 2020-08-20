class MinesweeperEnv(object):
    def __init__(self, width, height, n_mines):
        self.nrows, self.ncols = width, height
        self.n_mines = n_mines

    def populate_board(self):
        board = np.zeros((self.height, self.width), dtype='object')
        mines = self.n_mines

        while mines > 0:
            row, col = random.randint(0, self.nrows-1), random.randint(0, self.ncols-1)
            if board[row][col] != 'B':
                board[row][col] = 'B'
                mines -= 1

        return board

    def get_neighbors(self, coord):
        board_2d = [t['value'] for t in self.board]
        board_2d = np.reshape(board_2d, (self.nrows, self.ncols))

        tile = self.board[action_index]['index']
        x,y = tile[0], tile[1]

        neighbors = []
        for col in range(y-1, y+2):
            for row in range(x-1, x+2):
                if (-1 < x < self.nrows and
                    -1 < y < self.ncols and
                    (x != row or y != col) and
                    (0 <= col < self.ncols) and
                    (0 <= row < self.nrows)):
                    neighbors.append(board_2d[col,row])

        return neighbors
