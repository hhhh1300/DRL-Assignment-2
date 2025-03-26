from os import stat
import numpy as np
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt

COLOR_MAP = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
    4096: "#3c3a32", 8192: "#3c3a32", 16384: "#3c3a32", 32768: "#3c3a32"
}
TEXT_COLOR = {
    2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
    32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
    512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2", 4096: "#f9f6f2"
}

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True

        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def merge(self, row):
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        if np.any(self.board == 0):
            return False
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def act(self, action):
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved

        state_after = self.board.copy()
        score_after = self.score

        done = self.is_game_over()

        return state_after, score_after, moved, done, {}

    def render(self, mode="human", action=None):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")
        return not np.array_equal(self.board, temp_board)


import copy
import random
import math
import numpy as np
from collections import defaultdict


# -------------------------------
# TODO: Define transformation functions (rotation and reflection), i.e., rot90, rot180, ..., etc.
# -------------------------------
def rot90(coord, size):
    r, c = coord
    return (c, size - 1 - r)

def rot180(coord, size):
    r, c = coord
    return (size - 1 - r, size - 1 - c)

def rot270(coord, size):
    r, c = coord
    return (size - 1 - c, r)

def reflect_horizontal(coord, size):
    r, c = coord
    return (r, size - 1 - c)

def identity(coord, size):
    return coord



class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = [defaultdict(float) for _ in patterns]
        # Generate symmetrical transformations for each pattern
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            self.symmetry_patterns.append(syms)

    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        size = self.board_size
        transformations = [
            identity,
            rot90,
            rot180,
            rot270,
            reflect_horizontal,
        ]
        symmetries = []
        for transform in transformations:
            transformed_pattern = [transform(coord, size) for coord in pattern]
            symmetries.append(transformed_pattern)
            if transform == reflect_horizontal:
                transformed_90 = [rot90(coord, size) for coord in transformed_pattern]
                transformed_180 = [rot180(coord, size) for coord in transformed_pattern]
                transformed_270 = [rot270(coord, size) for coord in transformed_pattern]
                symmetries.append(transformed_90)
                symmetries.append(transformed_180)
                symmetries.append(transformed_270)
        return symmetries

    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        features = []
        for (r, c) in coords:
            tile_value = board[r][c]
            features.append(self.tile_to_index(tile_value))
        return tuple(features)

    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        total_val = 0.0
        for i, syms in enumerate(self.symmetry_patterns):
            for coords in syms:
                feat = self.get_feature(board, coords)
                total_val += self.weights[i][feat]
        return total_val / len(self.symmetry_patterns)
        # for i, pattern in enumerate(self.patterns):
        #       feat = self.get_feature(board, pattern)
        #       total_val += self.weights[i][feat]
        # return total_val / len(self.patterns)

    def update(self, board, delta, alpha):
        # TODO: Update weights based on the TD error.
        total_syms = 0
        for syms in self.symmetry_patterns:
            total_syms += len(syms)
        # assert total_syms == 32
        update_per_sym = alpha * delta / total_syms

        for i, syms in enumerate(self.symmetry_patterns):
            for coords in syms:
                feat = self.get_feature(board, coords)
                self.weights[i][feat] += update_per_sym
        # for i, pattern in enumerate(self.patterns):
        #     feat = self.get_feature(board, pattern)
        #     self.weights[i][feat] += alpha * delta


def td_learning(env, approximator, num_episodes=50000, alpha=0.01, gamma=0.99, epsilon=0.1):
    """
    Trains the 2048 agent using TD-Learning.

    Args:
        env: The 2048 game environment.
        approximator: NTupleApproximator instance.
        num_episodes: Number of training episodes.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Epsilon-greedy exploration rate.
    """

    def best_action(env, state, legal_moves):
        best_value = -1e9
        best_action = None
        for a in legal_moves:
            sim_env = copy.deepcopy(env)
            sim_env.board = state.copy()
            sim_env.score = previous_score

            state_after, score_after, moved, done, _ = sim_env.act(a) # compute the deterministic after state
            r = score_after - previous_score  # immediate reward
            v_after = approximator.value(state_after)
            if r + gamma * v_after > best_value:
                best_value = r + gamma * v_after
                best_action = a
        return best_action

    final_scores = []
    success_flags = []

    for episode in range(num_episodes):
        state = env.reset()
        trajectory = []  # Store trajectory data if needed
        previous_score = 0
        done = False
        max_tile = np.max(state)

        while not done:
            # print (state)
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break
            # TODO: action selection
            # Note: TD learning works fine on 2048 without explicit exploration, but you can still try some exploration methods.
            # if random.random() < epsilon:
            #     action = random.choice(legal_moves)
            # else:
            action = best_action(env, state, legal_moves)

            old_state = state.copy()
            state, score, moved, done, _ = env.act(action) # compute the deterministic after state
            state_after = state.copy()
            score_after = score
            if moved:
                env.add_random_tile()
            state_next = env.board.copy()
            score_next = env.score

            incremental_reward = score_next - previous_score
            previous_score = score_next
            max_tile = max(max_tile, np.max(state_next))

            # TODO: Store trajectory or just update depending on the implementation
            # v_s = approximator.value(old_state)
            # v_s_next = approximator.value(state_next)
            # td_error = incremental_reward + gamma * v_s_next - v_s


            # print ('----', old_state, state_after, state_next, sep='\n')

            state = state_next.copy()
            trajectory.append((old_state, action, score_after, score_next, state_after, state_next))
            max_tile = max(max_tile, np.max(state))

        # TODO: If you are storing the trajectory, consider updating it now depending on your implementation.
        for old_state, action, score_after, score_next, state_after, state_next in trajectory:
            sim_env = copy.deepcopy(env)
            sim_env.board = state_next.copy()
            sim_env.score = score_after
            a = best_action(sim_env, state_next, [a for a in range(4) if sim_env.is_move_legal(a)])
            # print (sim_env.board, a)
            if a is None:
                continue
            state_after_next, score_after_next, moved, _, _ = sim_env.act(a)

            incremental_reward = score_after_next - score_next
            # print (f'----{score_after_next}, {score_next}', state_after, state_next, state_after_next, sep='\n')

            v_s_after = approximator.value(state_after)
            v_s_after_next = approximator.value(state_after_next)
            td_error = incremental_reward + gamma * v_s_after_next - v_s_after

            approximator.update(state_after, td_error, alpha)


        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)

        if (episode + 1) % 20 == 0:
            print (np.sum([len(w.keys()) for w in approximator.weights]))
            # print (approximator.weights)
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.sum(success_flags[-100:]) / 100
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f}")

    return final_scores


# TODO: Define your own n-tuple patterns
patterns = [
    # [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)],
    # [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)],
    # [(0, 0), (1, 0), (2, 0), (3, 0), (2, 1), (3, 1)],
    # [(0, 1), (1, 1), (2, 1), (3, 1), (2, 2), (3, 2)],

    # [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
    # [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (0, 2), (0, 3)],
    [(1, 0), (1, 1), (1, 2), (1, 3)],
    # [(2, 0), (2, 1), (2, 2), (2, 3)],
    # [(3, 0), (3, 1), (3, 2), (3, 3)],

    [(0, 0), (1, 0), (2, 0), (3, 0)],
    [(0, 1), (1, 1), (2, 1), (3, 1)],
    # [(0, 2), (1, 2), (2, 2), (3, 2)],
    # [(0, 3), (1, 3), (2, 3), (3, 3)],

    # [(0, 0), (0, 1), (1, 0), (1, 1)],
    # [(0, 1), (0, 2), (1, 1), (1, 2)],
    # [(0, 2), (0, 3), (1, 2), (1, 3)],
    # [(1, 0), (1, 1), (2, 0), (2, 1)],
    # [(1, 1), (1, 2), (2, 1), (2, 2)],
    # [(1, 2), (1, 3), (2, 2), (2, 3)],
    # [(2, 0), (2, 1), (3, 0), (3, 1)],
    # [(2, 1), (2, 2), (3, 1), (3, 2)],
    # [(2, 2), (2, 3), (3, 2), (3, 3)],
]

import copy
import random
import math
import numpy as np

# Note: This MCTS implementation is almost identical to the previous one,
# except for the rollout phase, which now incorporates the approximator.

# Node for TD-MCTS using the TD-trained value approximator
class TD_MCTS_Node:
    def __init__(self, env, state, score, parent=None, action=None):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        # List of untried actions based on the current state's legal moves
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0


# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        # Create a deep copy of the environment with the given state and score.
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent.visits)/child.visits) to select the best child.
        best_value = float("-inf")
        best_child = None

        for child in node.children.values():
            q = child.total_reward / child.visits if child.visits > 0 else 0.0
            uct = q + self.c * math.sqrt(
                math.log(node.visits) / child.visits
            )
            if uct > best_value:
                best_value = uct
                best_child = child
            # print (q, self.c * math.sqrt(
            #     math.log(node.visits) / child.visits
            # ))
        return best_child


    def rollout(self, sim_env, depth):
        # TODO: Perform a random rollout until reaching the maximum depth or a terminal state.
        # TODO: Use the approximator to evaluate the final state.
        initial_score = self.approximator.value(sim_env.board)
        final_score = initial_score
        decay_factor = 0.95
        cnt = 1
        for _ in range(1):
            legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_moves:
                break
            # action = np.random.choice(legal_moves)
            for action in legal_moves:
                sim_env_copy = copy.deepcopy(sim_env)
                sim_env_copy.board = sim_env.board.copy()
                sim_env_copy.score = sim_env.score
                
                board, reward, moved, done, _ = sim_env_copy.act(action)
                final_score += self.approximator.value(board) * decay_factor
                
            decay_factor *= self.gamma
            cnt += 1
            if done:
                break
        # print (self.approximator.value(sim_env.board), initial_score)
        # print (self.approximator.value(sim_env.board))
        # return self.approximator.value(sim_env.board)
        # print (final_score, initial_score)
        return final_score


    def backpropagate(self, node, reward):
        # TODO: Propagate the obtained reward back up the tree.
        while node is not None:
            node.visits += 1
            node.total_reward += (reward - node.total_reward) / node.visits
            node = node.parent


    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        # TODO: Selection: Traverse the tree until reaching an unexpanded node.
        while node.fully_expanded() and node.children and not sim_env.is_game_over():
            node = self.select_child(node)
            _, _, done, _ = sim_env.step(node.action)
            if done:
                break


        # TODO: Expansion: If the node is not terminal, expand an untried action.
        if (not node.fully_expanded()) and (not sim_env.is_game_over()):
            action = node.untried_actions.pop()
            sim_env.step(action)

            child = TD_MCTS_Node(sim_env, state=sim_env.board.copy(), score=sim_env.score, parent=node, action=action)
            node.children[action] = child
            node = child

        # Rollout: Simulate a random game from the expanded node.
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        # Backpropagate the obtained reward.
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        # Compute the normalized visit count distribution for each child of the root.
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution