"""
TestTrainedModelGym_hermite: tests and plots for the RL algorithm

Author: Veronica Saz Ulibarrena
Last modified: 5-March-2025
"""
import numpy as np
from env.BridgedCluster_env import Cluster_env
from TrainRL import train_net
from Plots_TestTrained import plot_test_reward
from TestEnvironment import run_trajectory, load_state_files, plot_trajs
from Plots_TestEnvironment import plot_energy_vs_tcomp_avg_together, plot_trajs_tree

colors = ['steelblue', 'darkgoldenrod', 'mediumseagreen', 'coral',  \
        'mediumslateblue', 'deepskyblue', 'navy']


def load_reward(a, suffix = ''):
    """
    load_reward: load rewards from file 
    INPUTS:
        a: environment
        suffix: file suffix to load
    OUTPUTS:
        score: rewards
        EnergyE: energy error
        EnergyE_rel: second energy error case
        HuberLoss: huber loss
        tcomp: computation time
        testReward: reward for the test cases
        trainingTime: training time
    """
    score = []
    with open(a.settings['Training']['savemodel'] + suffix + "rewards.txt", "r") as f:
        for y in f.read().split('\n'):
            score_r = list()
            for j in y.split():
                score_r.append(float(j))
            score.append(score_r)

    EnergyE = []
    with open(a.settings['Training']['savemodel'] + suffix + "EnergyError.txt", "r") as f:
        for y in f.read().split('\n'):
            Energy_r = list()
            for j in y.split():
                Energy_r.append(float(j))
            EnergyE.append(Energy_r)

    EnergyE_rel = []
    with open(a.settings['Training']['savemodel'] + suffix + "EnergyError_rel.txt", "r") as f:
        for y in f.read().split('\n'):
            Energy_rel_r = list()
            for j in y.split():
                Energy_rel_r.append(float(j))
            EnergyE_rel.append(Energy_rel_r)

    tcomp = []
    with open(a.settings['Training']['savemodel'] + suffix + "Tcomp.txt", "r") as f:
        for y in f.read().split('\n'):
            tcomp_r = list()
            for j in y.split():
                tcomp_r.append(float(j))
            tcomp.append(tcomp_r)

    testReward = []
    with open(a.settings['Training']['savemodel'] + suffix + "TestReward.txt", "r") as f:
        for y in f.read().split('\n'):
            testreward_r = list()
            for j in y.split():
                testreward_r.append(float(j))
            testReward.append(testreward_r)

    trainingTime = []
    with open(a.settings['Training']['savemodel'] + suffix + "TrainingTime.txt", "r") as f:
        for j in f.read().split():
            trainingTime.append(float(j))

    HuberLoss = []

    return score, EnergyE, EnergyE_rel, HuberLoss, tcomp, testReward, trainingTime

if __name__ == '__main__':
    experiment = 3 # number of the experiment to be run

    if experiment == 0: # Train network
        env = Cluster_env()
        env.settings['Training']['RemovePlanets'] = False # train without planets (they do not contribute to the total energy error)
        env.settings['Integration']['subfolder'] = 'currentTraining/'
        # Without pretraining
        # train_net(env = env, suffix = "currentTraining/")

        # With pretraining
        model_path = env.settings['Training']['savemodel'] + 'model_weights173.pth'
        train_net(env = env, suffix = "currentTraining/", model_path_pretrained = model_path)

    elif experiment == 1: 
        # Plot training results
        env = Cluster_env()
        reward, EnergyError, EnergyError_rel, HuberLoss, tcomp, testReward, trainingTime = \
                load_reward(env, suffix = '')
        plot_test_reward(env, testReward, trainingTime)
        
    elif experiment == 2:
        # 2: use network trained with hyperparameters chosen by hand
        env = Cluster_env()
        env.settings['Integration']['subfolder'] = '4_run_RL_base/'
        env.settings['Training']['RemovePlanets'] = False

        env.settings['Integration']['max_error_accepted'] = 1e5
        env.settings['Integration']['max_steps'] = 40

        seed = 3
        env.settings['InitialConditions']['seed'] = seed
        env.settings['InitialConditions']['bodies_in_system'] = 'fixed'
        env.settings['InitialConditions']['n_bodies'] = 9

        model_path_index = '417'
        model_path = './Training_Results/model_weights'+model_path_index +'.pth'
        index_to_plot = [0, 1,3,6, 8, 10]
        
        NAMES = []
        TITLES = []
        NAMES.append('_actionRL')
        env.settings['Integration']['suffix'] = NAMES[0]
        TITLES.append(r'RL-variable $\mu$')
        run_trajectory(env, action = 'RL', model_path= model_path)
        for act in range(env.settings['RL']['number_actions']):
            NAMES.append('_action_%0.3E_seed%i'%(env.actions[act], seed))
            env.settings['Integration']['suffix'] = NAMES[act+1]
            TITLES.append(r'%i: $\mu$ = %.1E'%(act, env.actions[act]))
            run_trajectory(env, action = act)

        STATE = []
        CONS = []
        TCOMP = []
        TITLES2 = []
        for act in index_to_plot:
            env.settings['Integration']['suffix'] = NAMES[act]
            state, cons, tcomp = load_state_files(env)
            STATE.append(state)
            CONS.append(cons)
            TCOMP.append(tcomp)
            TITLES2.append(TITLES[act])

        save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
            'Action_comparison_RL.png'
        plot_trajs(env, STATE, CONS, TCOMP, TITLES2, save_path, plot_traj_index=[0,1])
        save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
            'Action_comparison_RL2.png'
        # plot_state_diff(env, STATE, CONS, TCOMP, TITLES, save_path)
        # save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
            # 'Action_comparison_RL3.png'
        # plot_comparison_end(env, STATE, CONS, TCOMP, NAMES, save_path, plot_traj_index=[0,1])

    elif experiment == 3:
        # 2: use network trained with hyperparameters chosen by hand
        env = Cluster_env()
        env.settings['Training']['RemovePlanets'] = False

        env.settings['Integration']['max_error_accepted'] = 1e5
        env.settings['Integration']['max_steps'] = 100
        seed = 1
        hybrid = True
        env.settings['InitialConditions']['seed'] = seed
        env.settings['InitialConditions']['bodies_in_system'] = 'fixed'

        model_path_index = '173'
        model_path = './Training_Results/model_weights'+model_path_index +'.pth'
        index_to_plot = [0,2,5, 7, 9]
        
        Bodies = [5, 9, 15]
        for i in range(len(Bodies)):
            env.settings['InitialConditions']['n_bodies'] = Bodies[i]
            env.settings['Integration']['subfolder'] = '42_runmodel_Nvary/%ibodies/'%Bodies[i]
            NAMES = []
            TITLES = []

            # RL
            NAMES.append('_actionRL_seed%i'%seed)
            env.settings['Integration']['suffix'] = NAMES[0]
            TITLES.append(r"RL-"+model_path_index)
            env.settings['Integration']["hybrid"] = False
            # run_trajectory(env, action = 'RL', model_path= model_path)

            if hybrid == True:
            # Hybrid
                NAMES.append('_actionHRL_seed%i'%seed)
                env.settings['Integration']['suffix'] = NAMES[1]
                TITLES.append(r"H-RL-"+model_path_index)
                env.settings['Integration']["hybrid"] = True
                # run_trajectory(env, action = 'RL', model_path= model_path)
                traj_index = 'HRLbest'
            else:
                traj_index = 'RLbest'

            # Fixed-step
            for a_i, act in enumerate(index_to_plot): # index_toplot includes RL
                action = act
                NAMES.append('_action_%0.3E_seed%i'%(env.actions[action], seed))
                env.settings['Integration']['suffix'] = NAMES[a_i]
                TITLES.append(r'%i: $\Delta t_B$ = %.1E'%(action, env.actions[action]))
                # run_trajectory(env, action = action)

            STATE = []
            CONS = []
            TCOMP = []
            TITLES2 = []
            for a_i in range(len(TITLES)):
                env.settings['Integration']['suffix'] = NAMES[a_i]
                state, cons, tcomp = load_state_files(env)
                STATE.append(state)
                CONS.append(cons)
                TCOMP.append(tcomp)
                TITLES2.append(TITLES[a_i])

            env.settings['Integration']['subfolder'] = '42_runmodel_Nvary/%iseed_%ibodies_'%(seed, Bodies[i])
            save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
                'Action_comparison_RL.png'
            plot_trajs(env, STATE, CONS, TCOMP, TITLES2, save_path, plot_traj_index=traj_index, hybrid = hybrid)

            save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
                'Action_comparison_avse.png'
            # plot_a_vs_e(env, STATE, CONS, TCOMP, TITLES2, save_path, plot_traj_index=traj_index)
        
    
    
    def start_env(bodies, subfolder, integrators, radius_cluster, t_step_param, \
                  hybrid = False, steps = 40):
        env = Cluster_env()
        env.settings['Integration']['subfolder'] = subfolder +'%ibodies/'%bodies
        env.settings['Integration']['savestate'] = True
        env.settings['Integration']['max_error_accepted'] = 1e5
        env.settings['InitialConditions']['bodies_in_system'] = 'fixed'
        env.settings['Integration']['max_steps'] = steps
        env.settings['InitialConditions']['n_bodies'] = bodies

        env.settings['InitialConditions']['radius_cluster'] = radius_cluster
        env.settings['Integration']['integrator_global'] = integrators[0]
        env.settings['Integration']['integrator_local'] = integrators[1]
        env.settings['RL']['t_step_param'] = t_step_param
        env.settings['Integration']["hybrid"] = hybrid
        env._initialize_RL() # To reset actions
        return env
    
    def run_many(subfolder, integrators, radius_cluster, t_step_param = 1, hybrid = False, steps = 40):
        initializations = 10
        seeds = np.arange(initializations)
        NAMES = []
        index_to_plot = [0,2,5, 9]
        Bodies = [5, 9, 15]
        model_path_index = '27'
        model_path = './Training_Results/model_weights'+model_path_index +'.pth'

        STATE_list = []
        CONS_list = []
        TCOMP_list = []
        TITLES_list = []

        for Body_i in range(len(Bodies)):
            TITLES = []

            # RL
            for i in range(initializations):
                env = start_env(Bodies[Body_i], subfolder, integrators, radius_cluster, t_step_param, hybrid = False, steps = steps)
                env.settings['InitialConditions']['seed'] = seeds[i]
                name = '_actionRL_%i'%i
                NAMES.append(name)
                env.settings['Integration']['suffix'] = name
                # run_trajectory(env, action = 'RL', model_path=model_path)
            TITLES.append(r"RL-"+model_path_index)

            # # Hybrid
            if hybrid == True:
                for i in range(initializations):
                    env = start_env(Bodies[Body_i], subfolder, integrators, radius_cluster, t_step_param, hybrid =True, steps = steps)
                    env.settings['InitialConditions']['seed'] = seeds[i]
                    name = '_actionHRL_%i'%i
                    NAMES.append(name)
                    env.settings['Integration']['suffix'] = name
                    # run_trajectory(env, action = 'RL', model_path=model_path)
                TITLES.append(r"H-RL-"+model_path_index)

            # Fixed-size
            for a_i, act in enumerate(index_to_plot): # index_toplot includes RL
                for i in range(initializations):
                    print(act, i)
                    env = start_env(Bodies[Body_i], subfolder, integrators, radius_cluster, t_step_param, hybrid = False, steps = steps)
                    env.settings['InitialConditions']['seed'] = seeds[i]
                    name = '_action_%i_%i'%(act, i)
                    NAMES.append(name)
                    env.settings['Integration']['suffix'] = name
                    # run_trajectory(env, action = act)
                TITLES.append(r'$\Delta t_B$ = %.1E'%(env.actions[act]*t_step_param))

            STATE = []
            CONS = []
            TCOMP = []
            for act in range(len(NAMES)):
                env.settings['Integration']['suffix'] = NAMES[act]
                state, cons, tcomp = load_state_files(env)
                STATE.append(state)
                CONS.append(cons)
                TCOMP.append(tcomp)

            env = start_env(Bodies[Body_i], subfolder, integrators, radius_cluster, t_step_param)
            save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
                'Energy_vs_tcomp.png'
            # plot_energy_vs_tcomp(env, STATE, CONS, TCOMP, TITLES, initializations, save_path, plot_traj_index=index_to_plot)
            # plot_energy_vs_tcomp_avg(env, STATE, CONS, TCOMP, TITLES, initializations, save_path, plot_traj_index=index_to_plot)

            STATE_list.append(STATE)
            CONS_list.append(CONS)
            TCOMP_list.append(TCOMP)
            TITLES_list.append(TITLES)
        save_path = env.settings['Integration']['savefile'] + subfolder 
        plot_energy_vs_tcomp_avg_together(env, STATE_list, CONS_list, TCOMP_list, TITLES_list, initializations, save_path, plot_traj_index=index_to_plot, hybrid = hybrid)

    if experiment == 4: # plot in tcomp vs energy together
        subfolder = '6_run_many/model27/'
        integrators = ['Ph4', 'Huayno']
        radius_cluster = 0.1
        run_many(subfolder, integrators, radius_cluster)
    
    if experiment == 5: # plot in tcomp vs energy together long term hybrid
        subfolder = '6_run_many/173_100steps_hybrid/'
        integrators = ['Ph4', 'Huayno']
        radius_cluster = 0.1
        run_many(subfolder, integrators, radius_cluster, hybrid = True, steps = 100)

    elif experiment == 6: # plot in tcomp vs energy together different integrators
        subfolder = '8_run_many_integrators/'
        integrators = ['Hermite', 'Ph4']
        radius_cluster = 0.1
        run_many(subfolder, integrators, radius_cluster)

    elif experiment == 7: # plot in tcomp vs energy together time-step parameter
        subfolder = '9_run_many_timestepparam/'
        integrators = ['Ph4', 'Huayno']
        radius_cluster = 0.1
        t_step_param = 1e-1
        run_many(subfolder, integrators, radius_cluster, t_step_param=t_step_param)

    elif experiment == 8: # run BHTree for a large number of particles
        env = Cluster_env()
        env.settings['Training']['RemovePlanets'] = False

        env.settings['Integration']['max_error_accepted'] = 1e5
        env.settings['Integration']['max_steps'] = 100
        seed = 2

        hybrid = True
        env.settings['InitialConditions']['seed'] = seed
        env.settings['InitialConditions']['bodies_in_system'] = 'fixed'
        Bodies = 1000
        env.settings['InitialConditions']['n_bodies'] = Bodies
        env.settings['Integration']['subfolder'] = '5_TreeCode/'
        env.settings['Integration']['integrator_global'] = 'BHTree'


        model_path_index = '173'
        model_path = './Training_Results/model_weights'+model_path_index +'.pth'
        index_to_plot = [0,2,5, 7, 9]
        
        NAMES = []
        TITLES = []

        # RL
        NAMES.append('_actionRL_seed%i'%seed)
        env.settings['Integration']['suffix'] = NAMES[0]
        TITLES.append(r"RL-"+model_path_index)
        env.settings['Integration']["hybrid"] = False
        # run_trajectory(env, action = 'RL', model_path= model_path)

        if hybrid == True:
        # Hybrid
            NAMES.append('_actionHRL_seed%i'%seed)
            env.settings['Integration']['suffix'] = NAMES[1]
            TITLES.append(r"H-RL-"+model_path_index)
            env.settings['Integration']["hybrid"] = True
            # run_trajectory(env, action = 'RL', model_path= model_path)
            traj_index = 'RLbest'
        else:
            traj_index = 'RLbest'

        # Fixed-step
        for a_i, act in enumerate(index_to_plot): # index_toplot includes RL
            action = act
            NAMES.append('_action_%0.3E_seed%i'%(env.actions[action], seed))
            env.settings['Integration']['suffix'] = NAMES[a_i+2]
            TITLES.append(r'%i: $\Delta t_B$ = %.1E'%(action, env.actions[action]))
            # run_trajectory(env, action = action)

        STATE = []
        CONS = []
        TCOMP = []
        TITLES2 = []
        for a_i in range(len(TITLES)):
            env.settings['Integration']['suffix'] = NAMES[a_i]
            state, cons, tcomp = load_state_files(env)
            STATE.append(state)
            CONS.append(cons)
            TCOMP.append(tcomp)
            TITLES2.append(TITLES[a_i])

        save_path = env.settings['Integration']['savefile'] + env.settings['Integration']['subfolder'] +\
            'Action_comparison_RL_seed%i.png'%seed
        plot_trajs_tree(env, STATE, CONS, TCOMP, TITLES2, save_path, plot_traj_index=traj_index, hybrid = hybrid)
