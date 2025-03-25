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

        return best_child


    def rollout(self, sim_env, depth):
        # TODO: Perform a random rollout until reaching the maximum depth or a terminal state.
        # TODO: Use the approximator to evaluate the final state.
        initial_score = self.approximator.value(sim_env.board)
        final_score = initial_score
        decay_factor = 0.95
        cnt = 1
        for _ in range(depth):
            legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_moves:
                break
            action = np.random.choice(legal_moves)
            board, reward, done, _ = sim_env.step(action)
            final_score += self.approximator.value(board) * decay_factor
            decay_factor *= self.gamma
            cnt += 1
            if done:
                break
        # print (self.approximator.value(sim_env.board), initial_score)
        # print (self.approximator.value(sim_env.board))
        # return self.approximator.value(sim_env.board)
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