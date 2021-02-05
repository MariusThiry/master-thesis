#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import copy

class ScheduleEnvironment(object):
    # This class models the schedule-environment
    # it saves the times, machines and stages of each heat and simulates steps through the state space;
    # a step is the movement of the next free heat to the next stage (except on stage 4, where all heats are scheduled
    # in one step)
    # also, this class checks after each step, which actions can be taken from the current state,
    # so that the schedule is still feasible; only these actions can be chosen from the respective stage;
    # further, this class has a method for the graphical rendering of the schedule

    def __init__(self, num_heats, rounds):
        # constructor

        # variables for testing, output, analysis
        self.rounds = rounds    # only for testing reasons

        # save the start and end times of heats on each machine; necessary for rendering
        self.heat_seq_out = []
        self.mach_out = []
        self.start_times = []
        self.tau_out = []

        # visible state --> List, which contains for each machine on next stage , when it will be free again
        # uses relative times --> times are given from the perspective of the next free heat
        # the part of the state-information, which is visible to the agent
        self.vis_state = [0, 0]

        #actions
        self.all_actions = [0, 1]  # all actions, that can be taken in general
        self.poss_actions = [0, 1]  # actions, that can be taken without rendering the schedule infeasible

        # additional state information, invisible to agent

        # save for each stage and machine the sequence of heats
        self.heat_seq_stage_mach = [[[]], [[], []], [[], []], [[], []], [[], []]]
        # save for each heat which machine it is processed on on each stage; each entry in a heats-list represents
        # a stage; the lists are ordered in the sequence of the stages
        self.mach_of_heat_and_stage = [[0] for heat in range(num_heats)]
        # save for each heat on each stage the start time t_start and the processing time tau[t_start, tau]
        self.times_of_heat_stage = [[[0, 0]] for heat in range(num_heats)]

        # times_of_heat_stage - variants for actions left and right;
        # these lists are overwritten after each step for those actions, that lead to a feasible schedule
        # when an action is taken, the corresponding table is used to overwrite the main table times_of_heat_stage
        self.times_of_heat_stage_left = [[[0, 0]] for heat in range(num_heats)]
        self.times_of_heat_stage_right = [[[0, 0]] for heat in range(num_heats)]

        # Parameters

        # number of heats to be scheduled
        self.num_heats = num_heats
        # minimum processing times tau for each machine on each stage
        self.tau_low_stage_mach = [[0], [85, 85], [8, 8], [45, 45], [60, 60]]
        # maximum possible tau
        self.tau_max_stage_mach = [[0], [100, 100], [20, 20], [60, 60], [80, 80]]
        # setup times times for each machine on each stage
        self.t_s_stage_mach = [[0], [9, 9], [5, 5], [15, 5], [0, 0]]
        # clean-up times times for each machine on each stage
        self.t_cl_stage_mach = [[0], [9, 9], [5, 5], [15, 5], [0, 0]]
        # maximum transfer time for stage 0 --> is calculated as a number, that never is met;
        # because stage 0 ist the buffer in front of the first stage and therefore has unlimited transfer time
        self.t_tr_max_stage_0 = self.num_heats * (max(self.tau_max_stage_mach[1][0], self.tau_max_stage_mach[1][1])
                                                  + max(self.t_s_stage_mach[1][0], self.t_s_stage_mach[1][1])
                                                  + max(self.t_cl_stage_mach[1][0], self.t_cl_stage_mach[1][1]))
        # maximum transfer or hold-up time of a heat between consecutive stages
        self.t_tr_max_stage = [self.t_tr_max_stage_0, 30, 5, 20, 0]

        # Initialize first set of possible actions and times_of_heat_stage - variants; necessary for the first step
        self.set_poss_actions_stages_1_to_3()
        return

    # get-methods

    def get_t_free_total_stage_mach(self, stage, mach):
        # return the total time, at which the given machine on the given stage is free again for processing the next heat
        t_free_total = 0
        for heat, stages_list in enumerate(self.copy_times_of_heat_stage_complete()):
            # check for each heat
            if len(stages_list) > stage:
                # is heat processed on given stage?
                if (self.get_mach_of_heat_and_stage(heat, stage) == mach):
                    # is heat processed on given machine?
                    heat_stage_times = stages_list[stage]
                    t_mach_ready = heat_stage_times[0]+heat_stage_times[1] + self.get_t_cl_stage_mach(stage, mach)\
                             + self.get_t_s_stage_mach(stage, mach)
                    if t_free_total < t_mach_ready:
                        t_free_total = t_mach_ready

        return t_free_total

    def get_times_of_heat_stage(self, heat, stage):
        # return list with start time and processing time for one specific heat and one specific stage
        return self.times_of_heat_stage[heat][stage][:]

    def get_mach_of_heat_and_stage(self, heat, stage):
        # return the machine, which the given heat is processed on on the given stage
        return self.mach_of_heat_and_stage[heat][stage]

    def get_num_heats(self):
        # return the total number of heats, that are to be scheduled in this environment
        return len(self.times_of_heat_stage)

    def get_tau_low_stage_mach(self, stage, mach):
        # return for a given stage and machine the minimum processing time tau_low
        return self.tau_low_stage_mach[stage][mach]

    def get_t_s_stage_mach(self, stage, mach):
        # return for a given stage and machine the setup time t_s
        return self.t_s_stage_mach[stage][mach]

    def get_t_cl_stage_mach(self, stage, mach):
        # return for a given stage and machine the clean-up time t_cl
        return self.t_cl_stage_mach[stage][mach]

    def get_t_tr_max_stage(self, stage):
        # return for a given stage the maximum transfer time t_tr_max
        return self.t_tr_max_stage[stage]

    def get_tau_max(self, stage, mach):
        # return for a given stage and machine the maximum processing time tau_max
        return self.tau_max_stage_mach[stage][mach]

    def get_poss_actions(self):
        # return all actions, that can be taken next by the agent without rendering the schedule infeasible
        return self.poss_actions

    def get_heat_seq_stage_mach(self, stage, mach):
        # return the sequence of heats, that are processed on the given stage and machine
        return self.heat_seq_stage_mach[stage][mach]

    def get_t_free_total_heat(self, heat):
        # return the total time, at which the given heat is free again for being processed on the next stage
        t_free_total = self.get_times_of_heat_stage(heat, -1)[0] + self.get_times_of_heat_stage(heat, -1)[1]
        return t_free_total

    def get_curr_stage_of_heat(self, heat):
        # return the stage, on which the given heat is being processed currently
        return len(self.times_of_heat_stage[heat]) - 1

    def get_next_free_heat(self):
        # return the next free heat --> heat, the processing of which is finished next which thus is ready for being
        # scheduled next
        t_free_heat_stages_0_to_2 = [self.get_t_free_total_heat(heat) for heat in range(self.get_num_heats())\
                                      if self.get_curr_stage_of_heat(heat) < 3]
        heats_stages_0_to_2 = [heat for heat in range(self.get_num_heats()) if self.get_curr_stage_of_heat(heat) < 3]
        if not t_free_heat_stages_0_to_2:
            return None, True
        else:
            return heats_stages_0_to_2[t_free_heat_stages_0_to_2.index(min(t_free_heat_stages_0_to_2))], False

    def get_vis_state(self):
        # return the part of the current state, which is visible to the agent --> relative times from the view of the
        # next free heat, when each machine on the next stage (from the view of the next free heat) is free again for
        # processing
        return self.vis_state[:]

    def get_all_consec_heats_on_machine(self, heat, stage, mach):
        # return all consecutive heats after the given heat on the given stage and machine
        heat_seq = self.get_heat_seq_stage_mach(stage, mach)
        consec_heats = heat_seq[heat:]
        return consec_heats

    # set-methods

    def set_heat_seq_stage_mach(self, stage, mach, heat):
        # appends the given heat to the heat sequence n the given machine and stage
        self.heat_seq_stage_mach[stage][mach].append(heat)
        return

    def set_vis_state(self, next_free_heat):
        # set the visible state for the given next free heat
        # not used in following cases: Next state is terminal or infeasible; Next stage of next_free_heat is stage 4
        next_stage = self.get_curr_stage_of_heat(next_free_heat) + 1
        t_free_next_free_heat = self.get_t_free_total_heat(next_free_heat)
        self.vis_state[0] = max(self.get_t_free_total_stage_mach(next_stage, 0) - t_free_next_free_heat, 0)
        self.vis_state[1] = max(self.get_t_free_total_stage_mach(next_stage, 1) - t_free_next_free_heat, 0)
        return

    def set_vis_state_terminal(self):
        # set the visible state to the terminal stage [-1, -1]
        self.vis_state = [-1, -1]
        return

    def set_vis_state_stage_four(self):
        # set the visible state to [0, 0]
        # used, when the next stage to be scheduled is stage 4
        self.vis_state = [0, 0]
        return

    def set_vis_state_infeasible(self):
        # set the visible state to the infeasible state [-2, -2]
        self.vis_state = [-2, -2]
        return

    def set_times_of_heat_stage_complete(self, times):
        # overwrite ALL ACTUAL times of ALL heats on ALL stages with the values given in times
        self.times_of_heat_stage = copy.deepcopy(times)
        return

    def set_times_of_heat_stage_action_complete(self, times, action):
        # overwrite ALL HYPOTHETICAL times - if the given action would be taken - of ALL heats on ALL stages with
        # the values given in times
        if action == 0:
            self.times_of_heat_stage_left = copy.deepcopy(times)
        else:
            self.times_of_heat_stage_right = copy.deepcopy(times)
        return

    # supportive methods for performance of steps and choice of possible actions

    def copy_times_of_heat_stage_complete(self):
        # return complete copy of list times_of_heat_stage, which saves ACTUAL start times and processing times
        # for all heats and stages
        times_copy = copy.deepcopy(self.times_of_heat_stage)
        return times_copy

    def copy_times_of_heat_stage_action_complete(self, action):
        # return complete copy of list times_of_heat_stage_left or list times_of_heat_stage_right, which save for all
        # heats and stages HYPOTHETICAL start times and processing times in case action 0 or action 1 is taken next
        if action == 0:
            times_copy = copy.deepcopy(self.times_of_heat_stage_left)
        else:
            times_copy = copy.deepcopy(self.times_of_heat_stage_right)

        return times_copy

    def add_next_mach_of_heat(self, heat, mach):
        # save for the given heat the machine, on which it is processed on the next stage
        self.mach_of_heat_and_stage[heat].append(mach)
        return

    def add_stage_and_times_of_heat(self, times, heat):
        # save for the given heat the given times on the next stage
        self.times_of_heat_stage[heat].append(times)
        return

    # methods related with the determination of the processing and start times of heats in the context of
    # the fulfillment of transfer times

    def calc_tau_nec(self, heat, stage, times_of_heat_stage_copy):
        # calculate the necessary tau of the given heat on the given stage to fulfill t_tr
        # by the start_time on the next stage
        mach_given_stage = self.get_mach_of_heat_and_stage(heat, stage)
        start_time_next_stage = times_of_heat_stage_copy[heat][stage+1][0]
        start_time_given_stage = times_of_heat_stage_copy[heat][stage][0]
        t_tr_max_given_stage = self.get_t_tr_max_stage(stage)
        tau_low_mach_given_stage = self.get_tau_low_stage_mach(stage, mach_given_stage)
        tau_nec = start_time_next_stage - start_time_given_stage - t_tr_max_given_stage
        return tau_nec

    def calc_new_start_times_after_tau_changed(self, start_heat, stage, times_of_heat_stage_copy):
        # calculate new start times of all heats on a given stage, after a tau of a heat on this stage
        # has been changed
        mach = self.get_mach_of_heat_and_stage(start_heat, stage)
        heat_seq_complete = self.get_heat_seq_stage_mach(stage, mach)
        heat_seq = heat_seq_complete[heat_seq_complete.index(start_heat):]
        # Anmerkung: hier kann ich auch die Methode get_consec_heats verwenden!
        for i in range(len(heat_seq)-1):
            start_time_new = max(times_of_heat_stage_copy[heat_seq[i+1]][stage][0],
                                  times_of_heat_stage_copy[heat_seq[i]][stage][0]
                                  + times_of_heat_stage_copy[heat_seq[i]][stage][1]
                                  + self.get_t_cl_stage_mach(stage, mach) + self.get_t_s_stage_mach(stage, mach))
            times_of_heat_stage_copy[heat_seq[i+1]][stage][0] = start_time_new

        return times_of_heat_stage_copy

    def calc_and_apply_tau_nec(self, heat, stage, times_of_heat_stage_copy):
        # calculate for the given heat on the given stage the necessary processing time tau of the heat to fulfill
        # the transfer time t_tr of the stage by the start time on the next stage; then apply the necessary tau and
        # adjust all start times of consecutive heats the stage
        times_of_heat_stage_copy[heat][stage][1] = self.calc_tau_nec(heat, stage, times_of_heat_stage_copy)
        times_of_heat_stage_copy = self.calc_new_start_times_after_tau_changed(heat, stage, times_of_heat_stage_copy)

        return times_of_heat_stage_copy

    def check_t_tr_fulfilled(self, heat, stage, times_of_heat_stage_copy):
        # check, if the transfer time t_tr of the given stage is fulfilled for the given heat by its start time of
        # the consecutive stage
        is_fulfilled = True
        t_start_next_stage = times_of_heat_stage_copy[heat][stage+1][0]
        end_time_given_stage = times_of_heat_stage_copy[heat][stage][0]\
                            + times_of_heat_stage_copy[heat][stage][1]

        if end_time_given_stage + self.get_t_tr_max_stage(stage) < t_start_next_stage:
            is_fulfilled = False

        return is_fulfilled

    def check_change_tau_t_tr_fulfillable(self, heat, stage, times_of_heat_stage_copy):
        # should be called after check_t_tr_fulfilled;
        # check, if the transfer time t_tr of the given heat on the given stage is fulfillable by the start time on
        # the consecutive stage by changing tau of the heat on the given stage

        is_fulfillable = True
        mach_given_stage = self.get_mach_of_heat_and_stage(heat, stage)
        t_start_next_stage = times_of_heat_stage_copy[heat][stage+1][0]
        if (times_of_heat_stage_copy[heat][stage][0] + self.get_tau_max(stage, mach_given_stage)
                + self.get_t_tr_max_stage(stage) < t_start_next_stage):
            is_fulfillable = False

        return is_fulfillable

    def change_tau_and_check_all_t_tr_fulfillable(self, start_heat, start_stage, times_of_heat_stage_copy):
        # is used, if t_tr on a stage for a heat is not fulfilled, but fulfillable;
        # calculate and apply the new processing time tau for the start heat on the start stage and check if
        # the transfert time t_tr of the start stage is fulfillable by the start time of the heat on the next stage;
        # then check, if the schedule is still feasible regarding t_tr for ALL heats on ALL stages

        t_tr_fulfillable_for_all_heats = True
        start_mach = self.get_mach_of_heat_and_stage(start_heat, start_stage)

        # calculate and apply necessary tau to first heat in sequence on start stage
        times_of_heat_stage_copy = self.calc_and_apply_tau_nec(start_heat, start_stage, times_of_heat_stage_copy)

        # check for each consecutive heat on stage, if t_tr of PREVIOUS stage is still fulfillable after
        # changing of start_times
        # workflow: check, if t_tr is fulfilled; if not: check, if it is fulfillable; if not:
        # apply new tau, adjust all start_times influenced by new tau; check again, if fulfilled;
        # repeat for all heats on all stages
        heat_seq = self.get_heat_seq_stage_mach(start_stage, start_mach)[start_heat:]
        for heat in heat_seq:
            if not self.check_t_tr_fulfilled(heat, start_stage-1, times_of_heat_stage_copy):
                if not self.check_change_tau_t_tr_fulfillable(heat, start_stage - 1, times_of_heat_stage_copy):
                    # if t_tr is not fulfillabe for one heat, return False
                    t_tr_fulfillable_for_all_heats = False
                    return t_tr_fulfillable_for_all_heats, times_of_heat_stage_copy
                else:
                    times_of_heat_stage_copy = self.calc_and_apply_tau_nec(heat, start_stage - 1,
                                                                           times_of_heat_stage_copy)

        # check for each heat on each previous stage, if t_tr is still fulfillable after changing of start_times
        for stage in range(max(start_stage-2, 0), 0, -1):
            for mach in range(1):
                heat_seq_mach = self.get_heat_seq_stage_mach(stage, mach)
                for heat in heat_seq_mach:
                    if not self.check_t_tr_fulfilled(heat, stage, times_of_heat_stage_copy):
                        if not self.check_change_tau_t_tr_fulfillable(heat, stage, times_of_heat_stage_copy):
                            # if t_tr is not fulfillabe for one heat, return False
                            t_tr_fulfillable_for_all_heats = False
                            return t_tr_fulfillable_for_all_heats, times_of_heat_stage_copy
                        else:
                            times_of_heat_stage_copy = self.calc_and_apply_tau_nec(heat, stage,
                                                                                   times_of_heat_stage_copy)

        return t_tr_fulfillable_for_all_heats, times_of_heat_stage_copy

    def calc_t_start_nec(self, heat, stage, times_of_heat_stage_copy):
        # calculate for the given heat and stage, whichstart time t_start on the next stage is necessary to fulfill
        # the transfer time t_tr of the given stage, given, that the maximum tau is used
        t_start_nec = times_of_heat_stage_copy[heat][stage+1][0] - self.get_t_tr_max_stage(stage)\
                      - self.get_tau_max(stage, self.get_mach_of_heat_and_stage(heat, stage))

        return t_start_nec

    def check_change_t_start_collision_free(self, heat, stage, times_of_heat_stage_copy):
        # should be called after check_t_tr_fulfilled;
        # check, if the transfer time t_tr of the given heat on the given stage is fulfillable by the start_time
        # on the next stage by changing the processing time tau of the heat on the given stage

        collision_free = True
        mach_given_stage = self.get_mach_of_heat_and_stage(heat, stage)
        heat_seq_mach = self.get_heat_seq_stage_mach(stage, mach_given_stage)
        if not heat_seq_mach.index(heat) == len(heat_seq_mach)-1:
            following_heat = heat_seq_mach[heat_seq_mach.index(heat)+1]
            if times_of_heat_stage_copy[following_heat][stage][0]\
                < self.calc_t_start_nec(heat, stage, times_of_heat_stage_copy) + \
                    self.get_tau_max(stage, mach_given_stage) + self.get_t_s_stage_mach(stage, mach_given_stage) \
                    + self.get_t_cl_stage_mach(stage, mach_given_stage):
                collision_free = False

        return collision_free

    def calc_and_set_t_start_new(self, heat, stage, times_of_heat_stage_copy):
        # calculate new t_start for the given heat and stage in order to fulfill the transfer time t_tr of the stage;
        # use tau_max of the stage and machine to minimize the change of t_start
        # comment: this prioritization may be changed in later versions, so that at first a change in start time is
        # checked and performed, before a change in the processing time is considered
        t_start_new = self.calc_t_start_nec(heat, stage, times_of_heat_stage_copy)
        times_of_heat_stage_copy[heat][stage][0] = t_start_new
        times_of_heat_stage_copy[heat][stage][1] = self.get_tau_max(stage, self.get_mach_of_heat_and_stage(heat, stage))
        return times_of_heat_stage_copy

    # Methods for the determination of possible actions, which, when performed, lead to a feasible schedule
    # A distinction is made between the actions for the for the scheduling of the stages 1 to 3 and the actions
    # for stage 4

    def set_poss_actions_stages_1_to_3(self):
        # check, for which machines on the next stage the transfer time t_tr_max is fulfillable;
        # only those machines can be chosen as actions during the next step;
        # always check in this order:
        # t_tr_fulfilled? -- no --> t_tr_fulfillable by change of process time on previous stage?
        # -- no --> t_tr_fulfillable by change of start time on previous stage?
        # if no actions are possible, an empty list is returned;
        # used for setting of actions for the scheduling of stages 1 to 3

        self.poss_actions = []

        # generate 2 copies of the times for each heat and stage,
        # to simulate the 2 scenarios of taking each action
        times_of_heat_stage_copy_left = self.copy_times_of_heat_stage_complete()
        times_of_heat_stage_copy_right = self.copy_times_of_heat_stage_complete()

        # check for next_free_heat:
        # first case: taking action 0 (left machine on next stage)
        # second case: taking action 1 (right machine on next stage)
        next_free_heat, _ = self.get_next_free_heat()
        curr_stage = self.get_curr_stage_of_heat(next_free_heat)
        times_of_heat_stage_copy_left[next_free_heat].append(
            [self.get_t_free_total_heat(next_free_heat) + self.get_vis_state()[0],
             self.get_tau_low_stage_mach(curr_stage + 1, 0)])
        times_of_heat_stage_copy_right[next_free_heat].append(
            [self.get_t_free_total_heat(next_free_heat) + self.get_vis_state()[1],
             self.get_tau_low_stage_mach(curr_stage + 1, 1)])

        # check, if action 0 (left machine) is feasible
        # if and elif: view onto the next stage; check compliance with t_tr of current stage
        if self.check_t_tr_fulfilled(next_free_heat, curr_stage, times_of_heat_stage_copy_left):
            self.poss_actions.append(0)
            self.set_times_of_heat_stage_action_complete(times_of_heat_stage_copy_left, 0)

        # check, if a change in tau can lead to the fulfillment of t_tr of the stage
        elif self.check_change_tau_t_tr_fulfillable(next_free_heat, curr_stage, times_of_heat_stage_copy_left):
            # view back to all previous stages; check compliance with t_tr of all previous stages
            all_t_tr_fulfillable, times_of_heat_stage_copy_left = \
                self.change_tau_and_check_all_t_tr_fulfillable(next_free_heat, curr_stage, times_of_heat_stage_copy_left)

            if all_t_tr_fulfillable == True:
                self.poss_actions.append(0)
                self.set_times_of_heat_stage_action_complete(times_of_heat_stage_copy_left, 0)

        # check, if a change of the start time can lead to the fulfillment of t_tr of the stage, without
        # interfering with the start time of the consecutive heat
        elif self.check_change_t_start_collision_free(next_free_heat, curr_stage, times_of_heat_stage_copy_left):
            times_of_heat_stage_copy_left = self.calc_and_set_t_start_new(next_free_heat, curr_stage,
                                                                          times_of_heat_stage_copy_left)
            self.poss_actions.append(0)
            self.set_times_of_heat_stage_action_complete(times_of_heat_stage_copy_left, 0)

        # check, if action 1 (right machine) is feasible
        # if and elif: view onto the next stage; check compliance with t_tr of current stage
        if self.check_t_tr_fulfilled(next_free_heat, curr_stage, times_of_heat_stage_copy_right):
            self.poss_actions.append(1)
            self.set_times_of_heat_stage_action_complete(times_of_heat_stage_copy_right, 1)
        elif self.check_change_tau_t_tr_fulfillable(next_free_heat, curr_stage, times_of_heat_stage_copy_right):
            # view back to all previous stages; check compliance with t_tr of all previous stages
            all_t_tr_fulfillable, times_of_heat_stage_copy_right = \
                self.change_tau_and_check_all_t_tr_fulfillable(next_free_heat, curr_stage,
                                                               times_of_heat_stage_copy_right)

            if all_t_tr_fulfillable == True:
                self.poss_actions.append(1)
                self.set_times_of_heat_stage_action_complete(times_of_heat_stage_copy_right, 1)

        # check, if a change of the start time can lead to the fulfillment of t_tr of the stage, without
        # interfering with the start time of the consecutive heat
        elif self.check_change_t_start_collision_free(next_free_heat, curr_stage, times_of_heat_stage_copy_right):
            times_of_heat_stage_copy_right = self.calc_and_set_t_start_new(next_free_heat, curr_stage,
                                                                           times_of_heat_stage_copy_right)
            self.poss_actions.append(1)
            self.set_times_of_heat_stage_action_complete(times_of_heat_stage_copy_right, 1)

        return

    def set_poss_actions_stage_4(self):
        # used for setting of actions for stage 4 while ensuring both continuous casting and the correct casting seq
        # check, how the heats can be scheduled on stage 4 without violating the transfer time of stage 3 and without
        # starting the processing, before a heat has been finished of stage 3; Check all options and save for a
        # possible action the best configuration (regarding the makespan); when this action is chosen, automatically
        # the best scheduling is done for this machine
        # annotation: at the moment, both casters are identical --> thus, there is no difference, on which caster
        # the heats are being processed; the check is therefore done only for one caster and if the check
        # is positive, both actions are added to possible_actions

        self.poss_actions = []
        times_list = []    # save times_of_heat_stage for each feasible solution; later select best one

        # check all possibilities for all heats and for taking action 0 (left machine on next stage)
        poss_start_times_stage_4 = []
        all_heats = [heat for heat in range(self.get_num_heats())]
        is_feasible = True

        for heat_1 in all_heats:
            # for each heat a test is done, if it is possible to schedule all heats on stage 4 in the desired sequence
            # and continuously, when setting the start time of the heat on stage 4 to the time at which it is finished
            # on stage 3; this corresponds to putting the heats in the correct casting sequence and then moving the
            # complete sequence as far as possible to the left on the time axis; all feasible outcomes are saved and
            # later the one with the shortest makespan is selected

            # generate 1 copy of the times for each heat and stage
            previous_heats = all_heats[:all_heats.index(heat_1)]
            following_heats = all_heats[all_heats.index(heat_1):]
            following_heats.remove(heat_1)
            times_of_heat_stage_copy = self.copy_times_of_heat_stage_complete()
            t_start_stage_4 = self.get_t_free_total_heat(heat_1)
            times_of_heat_stage_copy[heat_1].append([t_start_stage_4, self.get_tau_low_stage_mach(4, 0)])
            t_start_next_heat = t_start_stage_4 + self.get_tau_low_stage_mach(4, 0)
            t_start_stage_4 -= self.get_tau_low_stage_mach(4, 0)

            # check, if all heats following after heat_1 can be scheduled on stage 4 while maintaining feasibility
            is_feasible, times_of_heat_stage_copy = \
                self.check_stage_4_following_heats(t_start_next_heat, following_heats, times_of_heat_stage_copy)
            if is_feasible:
                # check, if all heats preceding heat_1 can be scheduled on stage 4 while maintaining feasibility
                is_feasible, t_start_stage_4, times_of_heat_stage_copy = \
                    self.check_stage_4_previous_heats(t_start_stage_4, previous_heats, times_of_heat_stage_copy)

            # after all necessary changes to tau on stage 3 have been made, a last check must be done to all heats,
            # if the heats are free to be processed on their starting time on stage 4
            if is_feasible:
                is_feasible = self.last_check_stage_4(all_heats, times_of_heat_stage_copy)

            # if the heat sequence with the proposed start time did pass all tests, include it into the list of
            # possible start times; also remember the start- and processing times, that were calculated for this case
            if is_feasible:
                poss_start_times_stage_4.append(t_start_stage_4)
                times_list.append(times_of_heat_stage_copy)

        # from all possible heat sequences, chose the one, that has the earliest start time
        # in order to minimize the makespan
        if len(poss_start_times_stage_4) > 0:
            t_start_stage_4_min = min(poss_start_times_stage_4)
            index = poss_start_times_stage_4.index(t_start_stage_4_min)
            times_of_heat_stage_copy = times_list[index]
            self.set_times_of_heat_stage_action_complete(times_of_heat_stage_copy, 0)
            self.set_times_of_heat_stage_action_complete(times_of_heat_stage_copy, 1)
            self.poss_actions.append(0)
            self.poss_actions.append(1)

        return

    def check_stage_4_following_heats(self, t_start_next_heat, following_heats, times_of_heat_stage_copy):
        # support function for schedule_stage_4
        # after choosing a hypothetical position for a heat on stage 4, check for all consecutively casted heats,
        # if they are free for casting when needed, and if t_tr is fulfilled
        is_feasible = True
        for heat_2 in following_heats:
            times_of_heat_stage_copy[heat_2].append([t_start_next_heat, self.get_tau_low_stage_mach(4, 0)])
            if t_start_next_heat < self.get_t_free_total_heat(heat_2):
                # check, if continuous casting is possible: next heat has to be finished on stage 3
                # when the current heat is finished on stage 4
                is_feasible = False
                break

            elif not self.check_t_tr_fulfilled(heat_2, 3, times_of_heat_stage_copy):
                # check, if t_tr of last stage is fulfilled for heat_2
                if self.check_change_tau_t_tr_fulfillable(heat_2, 3, times_of_heat_stage_copy):
                    # check, if t_tr is fulfillable on all PREVIOUS stages
                    all_t_tr_fulfillable, times_of_heat_stage_copy = \
                        self.change_tau_and_check_all_t_tr_fulfillable(heat_2, 3,
                                                                       times_of_heat_stage_copy)
                    if not all_t_tr_fulfillable:
                        is_feasible = False
                        break

            t_start_next_heat += self.get_tau_low_stage_mach(4, 0)

        # check (from back to forth), if there are still not-fulfilled t_tr; check, if these t_tr can be fulfilled
        # by changes in start times combined with use of maximum tau
        # back to forth is better, because more spaces in the schedule can be used
        if is_feasible:
            for heat_2 in reversed(following_heats):
                if not self.check_t_tr_fulfilled(heat_2, 3, times_of_heat_stage_copy):
                    # check, if t_tr of last stage is fulfilled for heat_2
                    if self.check_change_t_start_collision_free(heat_2, 3, times_of_heat_stage_copy):
                        times_of_heat_stage_copy = self.calc_and_set_t_start_new(heat_2, 3,
                                                                                 times_of_heat_stage_copy)
                    else:
                        is_feasible = False
                        break

        return is_feasible, times_of_heat_stage_copy

    def check_stage_4_previous_heats(self, t_start_stage_4, previous_heats, times_of_heat_stage_copy):
        # support method for schedule_stage_4;
        # after choosing a hypothetical position for a heat on stage 4 and testing all following heats,
        # check for all previous heats if they are free for the processing on stage 4 when needed,
        # and if t_tr is fulfilled

        # first, check if heats are free for casting
        is_feasible = True
        for heat_2 in reversed(previous_heats):
            times_of_heat_stage_copy[heat_2].append(
                [t_start_stage_4, self.get_tau_low_stage_mach(4, 0)])
            if t_start_stage_4 < self.get_t_free_total_heat(heat_2):
                # check, if continuous casting is possible: next heat has to be finished on stage 3
                # when the current heat is finished on stage 4
                is_feasible = False
                break

            t_start_stage_4 -= self.get_tau_low_stage_mach(4, 0)

        # check, if all t_tr are fulfilled
        # done from early to late start times, because changes in process times can resolve conflicts
        # in t_tr of later heats;
        if is_feasible:
            for heat_2 in previous_heats:
                if not self.check_t_tr_fulfilled(heat_2, 3, times_of_heat_stage_copy):
                    # check, if t_tr of last stage is fulfilled for heat_2
                    if self.check_change_tau_t_tr_fulfillable(heat_2, 3, times_of_heat_stage_copy):
                        # check, if t_tr is fulfillable on all PREVIOUS stages
                        all_t_tr_fulfillable, times_of_heat_stage_copy = \
                            self.change_tau_and_check_all_t_tr_fulfillable(heat_2, 3,
                                                                           times_of_heat_stage_copy)
                        if all_t_tr_fulfillable == False:
                            is_feasible = False
                            break

        # check (from back to forth), if there are still not-fulfilled t_tr; check, if these t_tr can be fulfilled
        # by changes in start times combined with use of maximum tau
        if is_feasible:
            for heat_2 in reversed(previous_heats):
                if not self.check_t_tr_fulfilled(heat_2, 3, times_of_heat_stage_copy):
                    # check, if t_tr of last stage is fulfilled for heat_2
                    if self.check_change_t_start_collision_free(heat_2, 3, times_of_heat_stage_copy):
                        times_of_heat_stage_copy = self.calc_and_set_t_start_new(heat_2, 3, times_of_heat_stage_copy)
                    else:
                        is_feasible = False
                        break

        return is_feasible, t_start_stage_4, times_of_heat_stage_copy

    def last_check_stage_4(self, all_heats, times_of_heat_stage_copy):
        # support method for schedule_stage_4;
        # last feasibility check of all heats on stage 3 when scheduling stage 4, to ensure,
        # that changes in processing times don't interfere with starting times on stage 4
        is_feasible = True
        for heat in all_heats:
            if times_of_heat_stage_copy[heat][3][0] + \
                    times_of_heat_stage_copy[heat][3][1] > times_of_heat_stage_copy[heat][4][0]:
                is_feasible = False

        return is_feasible

    # methods for performance of steps in the state-space, respectively a scheduling step

    def step(self, action):
        # perform one step in the state-space, either by moving the next free heat to the next machine
        # or by scheduling the complete last stage; after the step, determine the possible actions, that can be
        # taken from the resulting state without rendering the schedule infeasible; if no action is possible,
        # the next step will lead to the state 'infeasible'
        next_free_heat, stage_3_complete = self.get_next_free_heat()
        next_mach = action
        reward = 0
        is_terminal = False
        is_feasible = 1

        if stage_3_complete:
            self.schedule_stage_4(next_mach)
            self.set_vis_state_terminal()
            #self.render('test_render')
            if self.rounds == 3:
                x = 1
            # self.render('test_render') --> only for testing reasons
            is_terminal = True
            reward = self.calc_reward_makespan()
        else:
            self.schedule_one_heat(next_free_heat, next_mach)
            new_next_heat, stage_3_complete = self.get_next_free_heat()

            if stage_3_complete:
                self.set_vis_state_stage_four()
                #self.render('test_render') --> only for testing reasons
                self.set_poss_actions_stage_4()
            else:
                self.set_vis_state(new_next_heat)
                #self.render('test_render') --> only for testing reasons
                self.set_poss_actions_stages_1_to_3()

        if not self.get_poss_actions():
            self.set_vis_state_infeasible()
            is_feasible = 0
            # self.render('test_render') --> only for testing reasons
            is_terminal = True
            reward = -100

        return self.get_vis_state(), is_terminal, reward, is_feasible

    def schedule_one_heat(self, heat, next_mach):
        # determine the schedule for the given heat on the given next machine on the next stage
        # used for scheduling of stages 1 to 3
        next_stage = self.get_curr_stage_of_heat(heat) + 1
        self.set_heat_seq_stage_mach(next_stage, next_mach, heat)
        self.add_next_mach_of_heat(heat, next_mach)
        times_of_heat_stage_action = self.copy_times_of_heat_stage_action_complete(next_mach)
        self.set_times_of_heat_stage_complete(times_of_heat_stage_action)

        return

    def schedule_stage_4(self, next_mach):
        # determine the schedule for the given next machine on stage 4
        times_of_heat_stage_action = self.copy_times_of_heat_stage_action_complete(next_mach)
        self.set_times_of_heat_stage_complete(times_of_heat_stage_action)
        start_times_stage_4 = []

        for heat in range(self.get_num_heats()):
            start_times_stage_4.append([heat, self.copy_times_of_heat_stage_complete()[heat][-1][0]])
            self.add_next_mach_of_heat(heat, next_mach)

        for heats_and_times in start_times_stage_4:
            self.set_heat_seq_stage_mach(4, next_mach, heats_and_times[0])

        return

    # methods for calculation of reward

    def calc_reward_makespan(self):
        # reward for makespan: basic reward for makespan - actual makespan
        # reward consists of a basic point count, which is reduced with regard to the makespan
        reward_makespan = self.calc_basic_reward_ms() - self.calc_makespan()
        return reward_makespan

    def calc_makespan(self):
        # calculate the makespan of a complete schedule
        return max(self.get_t_free_total_stage_mach(4, 0), self.get_t_free_total_stage_mach(4, 1))

    def calc_basic_reward_ms(self):
        # calculate basic reward for the makespan
        # calculated as a 'worst case' scenario, in which on each stage all the heats are processed on the same machine
        # and one after another, and in which the processing on the next stage only starts after all heats have been
        # processed on the previous stage
        # it is to be noted, that this scenario usually not happens due to other heuristics and rules within the
        # scheduling process, and therefore it is just a theoretical baseline for the makespan

        t_worst_case_stage_1 = max((self.get_tau_low_stage_mach(1, 0) + self.get_t_s_stage_mach(1, 0)
                                    + self.get_t_cl_stage_mach(1, 0)),
                                   (self.get_tau_low_stage_mach(1, 1) + self.get_t_s_stage_mach(1, 1)
                                    + self.get_t_cl_stage_mach(1, 1))) * self.get_num_heats() - 1 \
                               + max(self.get_tau_low_stage_mach(1, 0), self.get_tau_low_stage_mach(1, 1))
        t_worst_case_stage_2 = max((self.get_tau_low_stage_mach(2, 0) + self.get_t_s_stage_mach(2, 0)
                                    + self.get_t_cl_stage_mach(2, 0)),
                                   (self.get_tau_low_stage_mach(2, 1) + self.get_t_s_stage_mach(2, 1)
                                    + self.get_t_cl_stage_mach(2, 1))) * self.get_num_heats() - 1 \
                               + max(self.get_tau_low_stage_mach(2, 0), self.get_tau_low_stage_mach(2, 1))
        t_worst_case_stage_3 = max((self.get_tau_low_stage_mach(3, 0) + self.get_t_s_stage_mach(3, 0)
                                    + self.get_t_cl_stage_mach(3, 0)),
                                   (self.get_tau_low_stage_mach(3, 1) + self.get_t_s_stage_mach(3, 1)
                                    + self.get_t_cl_stage_mach(3, 1))) * self.get_num_heats() - 1 \
                               + max(self.get_tau_low_stage_mach(3, 0), self.get_tau_low_stage_mach(3, 1))
        t_worst_case_stage_4 = max(self.get_tau_low_stage_mach(3, 0), self.get_tau_low_stage_mach(3, 1)) \
                               * self.get_num_heats()

        makespan_worst_case = t_worst_case_stage_1 + t_worst_case_stage_2 + t_worst_case_stage_3 + t_worst_case_stage_4
        return makespan_worst_case

    # Methods for the saving of the output and the graphical rendering of the schedule

    def save_output(self):
        # write output to lists for graphical rendering of the schedule
        machines_stages = [['mach0'], ['mach1', 'mach2'], ['mach3', 'mach4'], ['mach5', 'mach6'], ['mach7', 'mach8']]
        self.heat_seq_out = []
        self.mach_out = []
        self.start_times = []
        self.tau_out = []
        for stage, mach_list in enumerate(self.heat_seq_stage_mach):
            for mach, heat_list in enumerate(mach_list):
                for heat in heat_list:
                    if (stage <= len(self.copy_times_of_heat_stage_complete()[heat])):
                        self.mach_out.append(machines_stages[stage][mach])
                        self.heat_seq_out.append('heat' + str(heat + 1))
                        self.start_times.append(self.get_times_of_heat_stage(heat, stage)[0])
                        self.tau_out.append(self.get_times_of_heat_stage(heat, stage)[1])

        return

    def render(self, filename):
        # render the Gantt chart for display of solution
        self.save_output()
        data = {'Heats': self.heat_seq_out, 'Machines': self.mach_out, 'Start_Times': self.start_times,
                'Processing_Times': self.tau_out}

        df_results_heats = pd.DataFrame.from_dict(data)

        mach1_df = df_results_heats.loc[df_results_heats['Machines'] == 'mach1']
        mach1_df = mach1_df.set_index('Heats')
        mach1_times = list(zip(round(mach1_df.Start_Times), mach1_df.Processing_Times))

        mach2_df = df_results_heats.loc[df_results_heats['Machines'] == 'mach2']
        mach2_df = mach2_df.set_index('Heats')
        mach2_times = list(zip(round(mach2_df.Start_Times), mach2_df.Processing_Times))

        mach3_df = df_results_heats.loc[df_results_heats['Machines'] == 'mach3']
        mach3_df = mach3_df.set_index('Heats')
        mach3_times = list(zip(round(mach3_df.Start_Times), mach3_df.Processing_Times))


        mach4_df = df_results_heats.loc[df_results_heats['Machines'] == 'mach4']
        mach4_df = mach4_df.set_index('Heats')
        mach4_times = list(zip(round(mach4_df.Start_Times), mach4_df.Processing_Times))


        mach5_df = df_results_heats.loc[df_results_heats['Machines'] == 'mach5']
        mach5_df = mach5_df.set_index('Heats')
        mach5_times = list(zip(round(mach5_df.Start_Times), mach5_df.Processing_Times))


        mach6_df = df_results_heats.loc[df_results_heats['Machines'] == 'mach6']
        mach6_df = mach6_df.set_index('Heats')
        mach6_times = list(zip(round(mach6_df.Start_Times), mach6_df.Processing_Times))


        mach7_df = df_results_heats.loc[df_results_heats['Machines'] == 'mach7']
        mach7_df = mach7_df.set_index('Heats')
        mach7_times = list(zip(round(mach7_df.Start_Times), mach7_df.Processing_Times))


        mach8_df = df_results_heats.loc[df_results_heats['Machines'] == 'mach8']
        mach8_df = mach8_df.set_index('Heats')
        mach8_times = list(zip(round(mach8_df.Start_Times), mach8_df.Processing_Times))

        # parameters for plotsize
        x_max = 2 * self.get_num_heats() * (self.get_tau_low_stage_mach(1, 0) +\
                                            self.get_tau_low_stage_mach(4,0))
        y_max = 400
        bar_width = 40
        tick_interval = 50
        font_size = 10

        # Declaring a figure "gnt"
        fig, gnt = plt.subplots(figsize=(9, 7))

        # Setting Y-axis limits
        gnt.set_ylim(0, y_max + tick_interval)

        # Setting X-axis limits
        gnt.set_xlim(0, x_max)

        # Setting labels for x-axis and y-axis
        gnt.set_xlabel('hours since start')
        gnt.set_ylabel('Machines')

        # Setting ticks on y-axis
        gnt.set_yticks([tick_interval, 2 * tick_interval, 3 * tick_interval, 4 * tick_interval,
                        5 * tick_interval, 6 * tick_interval, 7 * tick_interval, 8 * tick_interval])
        # Labelling tickes of y-axis
        gnt.set_yticklabels(['CC 1', 'CC 2', 'LF 1', 'LF 2', 'AOD 1',
                             'AOD 2', 'EAF 1', 'EAF 2'])

        # Setting graph attribute
        gnt.grid(True)

        # Setting the Bars for the heats
        # Bar for EAF 1
        gnt.broken_barh(mach1_times, ((8 * tick_interval - (bar_width / 2)), bar_width), facecolors='tab:orange',
                        edgecolors='black')

        # Bar for EAF 2
        gnt.broken_barh(mach2_times, ((7 * tick_interval - (bar_width / 2)), bar_width), facecolors='tab:orange',
                        edgecolors='black')

        # Bar for AOD 1
        gnt.broken_barh(mach3_times, ((6 * tick_interval - (bar_width / 2)), bar_width), facecolors='tab:orange',
                        edgecolors='black')

        # Bar for AOD 2
        gnt.broken_barh(mach4_times, ((5 * tick_interval - (bar_width / 2)), bar_width), facecolors='tab:orange',
                        edgecolors='black')

        # Bar for LF 1
        gnt.broken_barh(mach5_times, ((4 * tick_interval - (bar_width / 2)), bar_width), facecolors='tab:orange',
                        edgecolors='black')

        # Bar for LF 2
        gnt.broken_barh(mach6_times, ((3 * tick_interval - (bar_width / 2)), bar_width), facecolors='tab:orange',
                        edgecolors='black')

        # Bar for CC 1
        gnt.broken_barh(mach7_times, ((2 * tick_interval - (bar_width / 2)), bar_width), facecolors='tab:orange',
                        edgecolors='black')

        # Bar for CC 2
        gnt.broken_barh(mach8_times, ((tick_interval - (bar_width / 2)), bar_width), facecolors='tab:orange',
                        edgecolors='black')

        # Annotate bars with job-names

        for heat in mach1_df.index:
            gnt.annotate(heat, (mach1_df.loc[heat].Start_Times + (mach1_df.loc[heat].Processing_Times / 2),
                                8 * tick_interval), ha='center', va='center', size=font_size)

        for heat in mach2_df.index:
            gnt.annotate(heat, (mach2_df.loc[heat].Start_Times + (mach2_df.loc[heat].Processing_Times / 2),
                                7 * tick_interval), ha='center', va='center', size=font_size)

        for heat in mach3_df.index:
            gnt.annotate(heat, (mach3_df.loc[heat].Start_Times + (mach3_df.loc[heat].Processing_Times / 2),
                                6 * tick_interval), ha='center', va='center', size=font_size)

        for heat in mach4_df.index:
            gnt.annotate(heat, (mach4_df.loc[heat].Start_Times + (mach4_df.loc[heat].Processing_Times / 2),
                                5 * tick_interval), ha='center', va='center', size=font_size)

        for heat in mach5_df.index:
            gnt.annotate(heat, (mach5_df.loc[heat].Start_Times + (mach5_df.loc[heat].Processing_Times / 2),
                                4 * tick_interval), ha='center', va='center', size=font_size)

        for heat in mach6_df.index:
            gnt.annotate(heat, (mach6_df.loc[heat].Start_Times + (mach6_df.loc[heat].Processing_Times / 2),
                                3 * tick_interval), ha='center', va='center', size=font_size)

        for heat in mach7_df.index:
            gnt.annotate(heat, (mach7_df.loc[heat].Start_Times + (mach7_df.loc[heat].Processing_Times / 2),
                                2 * tick_interval), ha='center', va='center', size=font_size)

        for heat in mach8_df.index:
            gnt.annotate(heat, (mach8_df.loc[heat].Start_Times + (mach8_df.loc[heat].Processing_Times / 2),
                                tick_interval), ha='center', va='center', size=font_size)

        plt.plot((0, x_max), (2.5 * tick_interval, 2.5 * tick_interval), 'k-')
        plt.plot((0, x_max), (4.5 * tick_interval, 4.5 * tick_interval), 'k-')
        plt.plot((0, x_max), (6.5 * tick_interval, 6.5 * tick_interval), 'k-')

        if self.get_vis_state()==[-2, -2]:
            plt.title("infeasible schedule")
        else:
            plt.title("feasible schedule round " + str(self.rounds))

        fig.savefig(filename, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
        plt.show()
        return