import json
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd, adam


class Snake(object):

    def __init__(self, grid_size):
        self.grid_size = grid_size
        # Actual action coordinates
        self.possible_actions = np.asarray([[0, -1], [0, 1], [1, 0], [-1, 0]])  # L R U D

    def reset(self):
        # The target fruit (f1, f2) and head (h1, h2) are both random at reset:
        f1 = np.random.randint(1, self.grid_size - 2, size=1)
        f2 = np.random.randint(1, self.grid_size - 2, size=1)
        h1 = np.random.randint(1, self.grid_size - 2, size=1)
        h2 = np.random.randint(3, self.grid_size - 2, size=1)
        # Assume the snake is horizontal at reset so the tail can't be outside the grid
        self.t1 = np.asarray([h1, h2 - 1])
        self.t2 = np.asarray([h1, h2 - 2])
        # State:
        self.state = np.asarray([f1, f2, h1, h2])

    def _observe(self):
        # (Draw_state included)
        img_size = (self.grid_size,) * 2
        state = self.state
        # Fruit: - canvas at rand int coord
        canvas = np.zeros(img_size)
        canvas[state[0], state[1]] = 1
        # Snake head: - canvas at rand int coord
        canvas[state[2], state[3]] = 1
        # Snake tail:
        canvas[self.t1[0], self.t1[1]] = 1
        canvas[self.t2[0], self.t2[1]] = 1
        return canvas.reshape((1, -1))

    def observe(self):
        # (Draw_state included)
        img_size = (self.grid_size,) * 2
        state = self.state
        # Fruit: - canvas at rand int coord
        canvas = np.zeros(img_size)
        canvas[state[0], state[1]] = 1
        # Snake head: - canvas at rand int coord
        canvas[state[2], state[3]] = 1
        # Snake tail:
        canvas[self.t1[0], self.t1[1]] = 1
        canvas[self.t2[0], self.t2[1]] = 1
        #print(self.state.reshape((1, -1)))
        return self.state.reshape((1, -1))

    # Returns reshaped canvas

    def update_state(self, action):
        # Need To check whether the snake is trying to go towards its tail
        # Chooses random movement instead
        if self.t1[1] == self.state[3] - 1 and action == 0:  # Go left
            self.action_index = np.random.choice([1, 2, 3])
            # self.action_index = 1  # Go right
        elif self.t1[1] == self.state[3] + 1 and action == 1:  # Go right
            self.action_index = np.random.choice([0, 2, 3])
            # self.action_index = 0  # Go left
        elif self.t1[0] == self.state[2] + 1 and action == 2:  # Go up
            self.action_index = np.random.choice([0, 1, 3])
            # self.action_index = 3  # Go down
        elif self.t1[0] == self.state[2] - 1 and action == 3:  # Go down
            self.action_index = np.random.choice([0, 1, 2])
            # self.action_index = 2  # Go up
        else:
            self.action_index = action

        # Update tails first:
        self.t2[0] = self.t1[0]
        self.t2[1] = self.t1[1]
        self.t1[0] = self.state[2]
        self.t1[1] = self.state[3]
        # Update head with action:
        a = np.asarray(self.possible_actions[self.action_index]).reshape((2, 1))
        b = self.state[2] + a[0]  # Update for row
        c = self.state[3] + a[1]  # Update for column
        self.state[2] = b  # New coord for head row
        self.state[3] = c  # New coord for head column

    # Updates state and tails with action

    def reward(self):
        # If head coord = fruit coord
        if self.state[2] == self.state[0] and self.state[3] == self.state[1]:
            return 1
        # If bumps into wall
        elif self.state[2] == self.grid_size - 1 or self.state[3] == self.grid_size - 1 or self.state[2] == 0 or \
                self.state[3] == 0:
            return -1
        else:
            return 0

    # If catches fruit, or hits wall

    def game_over(self):
        if self.state[2] == self.grid_size - 1 or self.state[3] == self.grid_size - 1 or self.state[2] == 0 or \
                self.state[3] == 0:
            return True  # Bumped into wall
        elif self.state[2] == self.state[0] and self.state[3] == self.state[1]:
            return True  # Ate the fruit
        else:
            return False

    def act(self, action):
        self.update_state(action)
        return self.observe(), self.reward(), self.game_over()


class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.95):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        self.memory.append([states, game_over])
        # print(np.shape(self.memory[0]))
        # print(self.memory[0])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        # print("Length of memory: ", len_memory)
        num_actions = model.output_shape[-1]
        # print("Number of actions: ", num_actions)
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((4, env_dim))

        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]
            # print(i) = 0
            # state_t is the new calculated state
            # inputs[i: i+1] = inputs[0 : 1] = state_t

            #print(state_t, np.shape (state_t), np.shape(inputs))
            #print(inputs[i:i + 1])
            inputs[i:i + 1] = state_t
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            targets[i] = model.predict(state_t)[0]
            Q_sa = np.max(model.predict(state_tp1)[0])
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets


if __name__ == "__main__":
    # parameters
    # epsilon = .01  # exploration
    num_actions = 4
    epoch = 1000
    max_memory = 500
    hidden_size = 100
    batch_size = 50
    grid_size = 10
    # Define environment/game
    env = Snake(grid_size)

    model = Sequential()
    model.add(Dense(100, input_shape=(4,), activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(adam(lr=.00001), "mse")

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)
    win_cnt = 0
    loss_plot = []

    for e in range(epoch):
        loss = 0.
        env.reset()
        game_over = False
        # Get initial input : canvas reshaped
        input_t = env.observe()

        while not game_over:
            input_tm1 = input_t
            # get next action
            # if np.random.rand() <= epsilon:  # explore
            #     action = np.random.randint(0, num_actions, size=1)
            # else:  # or get action from network
            q = model.predict(input_tm1)
            action = np.argmax(q[0])

            # apply action, get rewards and new state
            # act calls update_state function
            input_t, reward, game_over = env.act(action)

            if reward == 1:
                win_cnt += 1

            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # adapt model
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            loss += model.train_on_batch(inputs, targets)

        print("Epoch {:03d}/{} | Loss {:.4f} | Win count {}".format(e, epoch, loss, win_cnt))
        loss_plot.append(loss)
    # Save trained model weights and architecture, this will be used by the visualization code
    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)
    # Plot loss
    plt.plot(loss_plot)
    plt.show()
