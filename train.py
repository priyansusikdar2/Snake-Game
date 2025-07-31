from agent import Agent
from game import SnakeGameAI
import matplotlib.pyplot as plt

def train():
    agent = Agent()
    game = SnakeGameAI()
    scores = []
    mean_scores = []
    total_score = 0

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            print(f'Game {agent.n_games} | Score: {score}')
            scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            mean_scores.append(mean_score)

train()
