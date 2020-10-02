import argparse
from tqdm import tqdm
from keras.models import load_model
from MinesweeperAgent import *

# intake MinesweeperEnv parameters, beginner mode by default
def parse_args():
    parser = argparse.ArgumentParser(description='Train a DQN to play Minesweeper')
    parser.add_argument('--width', type=int, default=9,
                        help='width of the board')
    parser.add_argument('--height', type=int, default=9,
                        help='height of the board')
    parser.add_argument('--n_mines', type=int, default=10,
                        help='Number of mines on the board')
    parser.add_argument('--episodes', type=int, default=100_000,
                        help='Number of episodes to train on')

    return parser.parse_args()

params = parse_args()

AGGREGATE_STATS_EVERY = 100 # calculate stats every 100 games for tensorboard
SAVE_MODEL_EVERY = 10_000 # save model and replay every 10,000 episodes

def main():
    agent = MinesweeperAgent(params.width, params.height, params.n_mines)

    progress_list, wins_list, ep_rewards = [], [], []
    n_clicks = 0

    for episode in tqdm(range(1, params.episodes+1), unit='episode'):
        agent.tensorboard.step = episode

        agent.reset()
        episode_reward = 0
        past_n_wins = agent.n_wins

        done = False
        while not done:
            current_state = agent.state_im

            action = agent.get_action(current_state)

            new_state, reward, done = agent.step(action)

            episode_reward += reward

            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done, episode)

            n_clicks += 1

        progress_list.append(agent.n_progress) # n of non-guess moves
        ep_rewards.append(episode_reward)

        if agent.n_wins > past_n_wins:
            wins_list.append(1)
        else:
            wins_list.append(0)

        if len(agent.replay_memory) < MEM_SIZE_MIN:
            continue

        if not episode % AGGREGATE_STATS_EVERY:
            med_progress = np.median(progress_list[-AGGREGATE_STATS_EVERY:])
            win_rate = np.sum(wins_list[-AGGREGATE_STATS_EVERY:]) / AGGREGATE_STATS_EVERY
            med_reward = np.median(ep_rewards[-AGGREGATE_STATS_EVERY:])

            agent.tensorboard.update_stats(
                progress_med = med_progress,
                winrate = win_rate,
                reward_med = med_reward,
                learn_rate = agent.learn_rate,
                epsilon = agent.epsilon)

        if not episode % SAVE_MODEL_EVERY:
            agent.model.save(f'{ROOT}/models/{MODEL_NAME}_{best}.h5')
            '''with open(f'replay/{MODEL_NAME}.pkl', 'wb') as output:
                pickle.dump(agent.replay_memory, output)'''

            print(f'Episode: {episode}, n_clicks: {n_clicks}, Median progress: {med_progress}, Median reward: {med_reward}, Max reward : {max_reward}')

if __name__ == "__main__":
    main()
