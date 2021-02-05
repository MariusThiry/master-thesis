import agent_sgd

import matplotlib.pyplot as plt

def build_schedule_with_sgd(num_heats, sgd_file_action_0, sgd_file_action_1):
    # build a schedule greedily for the given number of heats using SGD-regressors from previous training,
    # saved in the given SGD-files

    # parameters
    epsilon_start = 0
    epsilon_decay_rate = 0
    epsilon_min = 0
    gamma = 0
    alpha = 0
    seed = 0

    #create agent, load regressors and create environment
    build_agent = agent_sgd.Agent(epsilon_start=epsilon_start, epsilon_decay_rate=epsilon_decay_rate, epsilon_min=epsilon_min, gamma=gamma, alpha=alpha, seed=seed)
    rewards = []
    build_agent.load_regressors_from_file(sgd_file_action_0, sgd_file_action_1)
    build_agent.create_environment(num_heats, 1)

    # build schedule using greedy action selection, save the rewards and render the result graphically
    is_feasible = build_agent.build_schedule_greedy()
    rewards.append(build_agent.get_last_reward())
    build_agent.env.render('currentExperiment/build_render')

    # print information to the build-process
    print('')
    ms = build_agent.env.calc_makespan()
    if ms > 0:
        print('Achieved Makespan: ')
        print(ms)
    print('')
    print('Schedule is feasible: ' + str(is_feasible))
    return

def train_with_sgd(num_heats, iterations, seed: int, sgd_file_name):
    # train an agent with the given number of heats and save the results in the given file (2 files are created out
    # of the given 1 file); number of iterations and seed for the random methods can be specified

    # parameters
    epsilon_start = 1.0
    epsilon_decay_rate = 0.995
    epsilon_min = 0.3
    gamma = 0.99
    alpha = 0.2

    # initialize agent and SGD-regressor and create lists for saving of output and results of training process
    train_agent = agent_sgd.Agent(epsilon_start=epsilon_start, epsilon_decay_rate=epsilon_decay_rate,
                                 epsilon_min=epsilon_min, gamma=gamma, alpha=alpha, seed=seed)
    rewards = []
    makespans = []
    epsilons = []
    feasible_schedules_list = []
    train_agent.initialize_sgd()

    # print information about training run
    print('Run with:')
    print('Epsilon_Start:' +str(epsilon_start))
    print('Epsilon_Min:' + str(epsilon_min))
    print('Gamma:' + str(gamma))
    print('Alpha:' + str(alpha))
    print('Seed:' + str(seed))
    print('Iterations:' + str(iterations))
    print('Number of Heats:' + str(num_heats))
    print('')

    # perform training iterations
    for r in range(iterations):
        print('')
        print('')
        print('round ' + str(r))
        train_agent.create_environment(num_heats, r)
        is_terminal = False
        i = 0
        while not is_terminal:
            print('')
            print('')
            print('Step ' + str(i))
            is_terminal = train_agent.perform_learning_step()
            i += 1

        # save results of training iteration

        #train_agent.env.render('currentExperiment/finalSchedule' + str(iterations) + 'Rounds' + str(numHeats) + 'Heats')
        rewards.append(train_agent.get_last_reward())
        makespans.append(train_agent.env.calc_makespan())
        epsilons.append(train_agent.epsilon)
        feasible_schedules_list.append(train_agent.get_is_feasible())
        #if (r == 0) or (r==1) or (r == 10) or (r%500 == 0):
            #train_agent.env.render('test_render_terminal_round' +str(r))
        # del train_agent.env
        #train_agent.env.render('scheduleRound' +str(r))

    # print analyses of rewards, makespans and feasibility of generated schedules
    print('Analysis of Rewards and Makespans:')
    print('')
    print('Analysis of all received Rewards:')
    print_reward_analysis(rewards)
    print('')
    print('Analysis of all Makespans:')
    print('')
    print_makespan_analysis(makespans)
    print('')
    print('Rewards per Interval')
    print('Rewards per Interval: ')
    print_rewards_per_interval(rewards, 10)
    print('')
    print('Makespans per Interval: ')
    print_makespans_per_interval(makespans, 10)

    print('')
    print('Achieved Makespan: ')
    print(train_agent.env.calc_makespan())

    print('')
    print('Feasibility Analysis: ')
    print('')
    print('Total Count:')
    print_feasibility_analysis(feasible_schedules_list)
    print('')
    print('Feasible Schedules per Interval:')
    print_feasibility_per_interval(feasible_schedules_list, 10)

    # save training results in form of trained SGD-regressors to files
    train_agent.save_regressors_to_file('currentExperiment/' + str(sgd_file_name))

    # render the progress of rewards and epsilon during the training progess graphically
    reward_indexes = [i for i, r in enumerate(rewards)]
    plt.scatter(reward_indexes, rewards)

    plt.savefig('currentExperiment/rewards', dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
    plt.close()
    plt.plot(epsilons)
    plt.savefig('currentExperiment/epsilons', dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)

    # render the schedule graphically
    train_agent.env.render('currentExperiment/finalSchedule' + str(iterations) + 'Rounds' + str(num_heats) + 'Heats')
    return

# methods for analyses of feasibility, rewards and makespans

def analyze_feasibility(feasible_schedules_list):
    # return analysis of feasibility based on the given list of feasible schedules (absolute and percentage)
    feasible_absolute = sum(feasible_schedules_list)
    feasible_percent = sum(feasible_schedules_list) / len(feasible_schedules_list) * 100
    return feasible_absolute, feasible_percent

def print_feasibility_analysis(feasibility_list):
    # print feasibility analysis saved in the given feasibility list (absolute and percentage of iterations, in which
    # feasible schedules were achieved)
    feasible_absolute, feasible_percent = analyze_feasibility(feasibility_list)
    print('absolute number of feasible schedules: ' + str(feasible_absolute))
    print('percentage of feasible schedules: ' + str(feasible_percent))
    return

def print_feasibility_per_interval(feasible_schedules_list, interval_size):
    # print a feasibility analysis on the basis on the given list of feasible schedules and for intervals with
    # the given interval size
    for i in range(interval_size):
        interval_start = round(i*len(feasible_schedules_list)/interval_size)
        interval_end = round((i+1)*len(feasible_schedules_list)/interval_size-1)
        list_slice = feasible_schedules_list[interval_start:interval_end]
        print('')
        print('Feasible Schedules in interval ' + str(interval_start) + ' - ' + str(interval_end))
        print_feasibility_analysis(list_slice)

    return

def analyze_reward(rewards_list):
    # return analysis of rewards based on the given list: count number of iterations, in which each distinct
    # reward was achieved
    diff_rewards = []
    num_of_rewards = []
    for el in rewards_list:
        if el not in diff_rewards:
            diff_rewards.append(el)
    
    diff_rewards.sort()
    for el in diff_rewards:
        num_of_rewards.append(rewards_list.count(el))

    return diff_rewards, num_of_rewards

def print_reward_analysis(rewards_list):
    # print analysis of rewards for the given lists of rewards
    diff_rewards, num_of_rewards = analyze_reward(rewards_list)
    for i in range(len(diff_rewards)):
        num_rel = num_of_rewards[i]/sum(num_of_rewards) * 100
        print('Reward: ' + str(diff_rewards[i]) + ', achieved in ' + str(num_rel) + '%')
    return

def print_rewards_per_interval(rewards_list, interval_size):
    # print analysis of rewards in the given list for each interval, of iterations, using the given interval size
    for i in range(interval_size):
        interval_start = round(i*len(rewards_list)/interval_size)
        interval_end = round((i+1)*len(rewards_list)/interval_size-1)
        list_slice = rewards_list[interval_start:interval_end]
        print('')
        print('Rewards for interval ' + str(interval_start) + ' - ' + str(interval_end))
        print_reward_analysis(list_slice)

    return

def analyze_makespans(ms_list):
    # analyze makespans based on given list of makespans achieved during the iterations;
    # count number of iterations, in which each distinct makespan was achieved
    diff_makespans = []
    num_of_makespans = []
    for el in ms_list:
        if el not in diff_makespans:
            diff_makespans.append(el)

    diff_makespans.sort(reverse=True)
    for el in diff_makespans:
        num_of_makespans.append(ms_list.count(el))

    return diff_makespans, num_of_makespans

def print_makespan_analysis(ms_list):
    # print analysis of makespans in the given list
    diff_makespans, num_of_makespans = analyze_makespans(ms_list)
    for i in range(len(diff_makespans)):
        num_rel = num_of_makespans[i]/sum(num_of_makespans) * 100
        print('Makespan: ' + str(diff_makespans[i]) + ', achieved in ' + str(num_rel) + '%')
    return

def print_makespans_per_interval(ms_list, interval_size):
    # print analysis of makespans in the given list for intervals of iterations, using the given interval size
    for i in range(interval_size):
        interval_start = round(i*len(ms_list)/interval_size)
        interval_end = round((i+1)*len(ms_list)/interval_size-1)
        list_slice = ms_list[interval_start:interval_end]
        print('')
        print('Makespans for interval ' + str(interval_start) + ' - ' + str(interval_end))
        print_makespan_analysis(list_slice)

    return

# main method

if __name__=='__main__':
    #build_schedule_with_sgd(4, 'currentExperiment/test_sgd_action_0', 'currentExperiment/test_sgd_action_1')
    train_with_sgd(4, 300, 15, 'test_sgd')
