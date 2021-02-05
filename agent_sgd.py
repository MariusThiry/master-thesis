import random
import pickle
from sklearn.linear_model import SGDRegressor

import scheduling_environment

class Agent(object):
    def __init__(self, epsilon_start, epsilon_decay_rate, epsilon_min, gamma, alpha, seed):
        # constructor

        self.env = None
        self.epsilon = epsilon_start
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.alpha = alpha
        self.lastreward = 0
        self.current_schedule_feasible = 1
        self.learningBatch = [[],[]]
        self.minFeatureValue = 0
        self.maxFeatureValue = 0
        self.stateList = []
        self.sgd_list = []
        random.seed(seed)   # initialize random seed for all uses of random within an object of this class
        print('Seed: ' +str(seed))
        return

    # get-methods

    def get_q_value(self, state, action):
        # return the q-value of the given state and action, predicted by using the SGD-regressor
        return self.sgd_list[action].predict([state + [action]])[0]

    def get_last_reward(self):
        # return the last achieved reward; used for analysis of the development of the learning process
        return self.lastreward

    def get_is_feasible(self):
        # return the information, if the schedule, that the agent currently is building, is still feasible
        return self.is_feasible

    # set-methods

    def set_is_feasible(self, is_feas):
        # set the information, if the schedule, that the agent currently is building, is still feasible
        self.is_feasible = is_feas
        return

    def set_q_value(self, state, action, newQValue):
        # save the given q-value for the given state and action by performing a partial fit of the corresponing
        # SGD-regressor
        self.sgd_list[action].partial_fit([state + [action]], [newQValue])
        return

    def initialize_sgd(self):
        # initialize the two SGD-regressors (one per action) for the function approximation by performing a first
        # partial fit of feature vectors of 0 and target vectors of 0; save them in the list sgd_list
        sgdr_action_0 = SGDRegressor()
        sgdr_action_1 = SGDRegressor()
        self.sgd_list.append(sgdr_action_0)
        self.sgd_list.append(sgdr_action_1)
        self.sgd_list[0].partial_fit([[0, 0, 0]], [0])
        self.sgd_list[1].partial_fit([[0, 0, 0]], [0])
        return

    # method for building a schedule based on learning experience

    def build_schedule_greedy(self):
        # build a schedule based on learning experience using greedy action selection
        isTerminal = False
        while not isTerminal:
            state = self.env.get_vis_state()
            action = self.choose_action_greedy(state)
            nextState, isTerminal, reward, is_feasible = self.env.step(action)
            self.lastreward = reward

        return is_feasible

    # methods for training of the agent

    def create_environment(self, numHeats, rounds):
        # initialise new environment with the desired number of heats
        # comment: 'rounds' is only for testing and debugging reasons
        self.env = scheduling_environment.ScheduleEnvironment(numHeats, rounds)
        return

    def perform_learning_step(self):
        # perform one learning step: Chose an action out of the possible actions epsilon-greedily and perform a step
        # in the environment
        state = self.env.get_vis_state()
        action = self.choose_action_epsilon_greedy(state)
        nextState, isTerminal, reward, is_feasible = self.env.step(action)
        self.lastreward = reward
        self.perform_value_update(state, action, nextState, reward, isTerminal)
        if isTerminal:
            self.set_is_feasible(is_feasible)
            self.epsilon *= self.epsilon_decay_rate
            if self.epsilon <= self.epsilon_min:
                self.epsilon = self.epsilon_min

        return isTerminal

    def perform_value_update(self, state, action, nextState, reward, isTerminal):
        # performs the q-learning value update: Calculate new q-value und fit regressor to new value
        q_value_old = self.get_q_value(state, action)
        next_state_q_values_list = []
        for act in self.env.poss_actions:
            next_state_action = state + [act]
            next_q_value = self.sgd_list[act].predict([next_state_action])[0]
            next_state_q_values_list.append(next_q_value)

        if not isTerminal:
            q_value_new = q_value_old + self.alpha * (reward + self.gamma * max(next_state_q_values_list) - q_value_old)
        else:
            q_value_new = reward

        self.set_q_value(state, action, q_value_new)
        return

    # methods for selection of actions (greedy and epsilon-greedy)
    def choose_action_epsilon_greedy(self, state):
        # choose an action out of the list of possible actions following the epsilon-greedy approach
        self.print_step_info()
        print('next free heat:' +str(self.env.get_next_free_heat()))
        if random.random() < self.epsilon:
            action = random.choice(self.env.poss_actions)
        else:
            action = self.choose_action_greedy(state)

        return action

    def choose_action_greedy(self, state):
        # choose an action out of the list of possible actions following the greedy approach
        q_values_list = []
        action_list = []
        for act in self.env.poss_actions:
            q_value = self.get_q_value(state, act)
            q_values_list.append(q_value)
            action_list.append(act)

        chosen_action = action_list[q_values_list.index(max(q_values_list))]
        return chosen_action

    # methods for in- and output

    def print_step_info(self):
        # print information about the current step
        # just for testing and debugging reasons; can maybe be deleted later
        print('current visible state: ' + str(self.env.vis_state))
        print('possible actions: ' + str(self.env.poss_actions))
        return

    def save_regressors_to_file(self, filename):
        # save learning results in form of regressors with pickle to a file
        pickle.dump(self.sgd_list[0], open(filename + '_action_0', 'wb'))
        pickle.dump(self.sgd_list[1], open(filename + '_action_1', 'wb'))
        return

    def load_regressors_from_file(self, filename_action_0, filename_action_1):
        # load learning results from previous training in form of regressors from a file with pickle
        self.sgd_list.append(pickle.load(open(filename_action_0, 'rb')))
        self.sgd_list.append(pickle.load(open(filename_action_1, 'rb')))
        return