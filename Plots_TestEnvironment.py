"""
TestEnvironment: tests simulation environments

Author: Veronica Saz Ulibarrena
Last modified: 8-February-2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch

from PlotsFunctions import plot_planetary_system_trajectory, plot_planets_trajectory,\
    plot_evolution, plot_distance_to_one, plot_actions_taken, plot_diff_state

colors = ['steelblue', 'darkgoldenrod', 'mediumseagreen', 'coral',  \
        'mediumslateblue', 'deepskyblue', 'navy', 'black']
colors2 = ['navy']
lines = ['-', '--', ':', '-.' ]
markers = ['o', 'x', '.', '^', 's']

def calculate_errors(states, cons, tcomp):
    cases = len(states)
    steps = np.shape(cons[0][:, 0])[0]

    # Calculate the energy errors
    R = np.zeros((steps, cases))
    E_T = np.zeros((steps, cases))
    E_T_rel = np.zeros((steps, cases))
    T_c = np.zeros((steps, cases))
    Action = np.zeros((steps, cases))
    for i in range(cases):
        R[1:, i] = cons[i][1:steps, 1]
        E_T_rel[1:, i] = abs(cons[i][1:steps, 2]) # absolute relative energy error
        E_T[1:, i] = abs(cons[i][1:steps, 3]) # absolute relative energy error
        T_c[1:, i] = np.cumsum(tcomp[i][1:steps]) # add individual computation times
        Action[1:, i] = cons[i][1:steps, 0]

    return [E_T_rel, E_T], T_c, R, Action

def plot_convergence(env, STATES, CONS, TCOMP, Titles, save_path, plot_traj_index = 'bestworst'):
    # Setup plot
    label_size = 18
    linestyle = ['--', ':', '-', '-', '-', '-', '-', '-', '-']
    Energy_error, T_comp, R, action = calculate_errors(STATES, CONS, TCOMP)

    for PLOT in range(len(STATES)):
        fig = plt.figure(figsize = (10,15))
        gs1 = matplotlib.gridspec.GridSpec(6, 2, 
                                        left=0.1, wspace=0.3, hspace = 0.3, right = 0.93,
                                        top = 0.88, bottom = 0.12)
        
        
        # Plot trajectories 2D
        # name_bodies = (np.arange(np.shape(STATES[0][[0]])[1])+1).astype(str)
        name_bodies = [r"$S_1$", r"$S_2$", r"$S_3$", r"$S_4$", r"$S_5$", r"$P_1$", r"$P_2$"]
        legend = True

        plot_traj_index = [0, PLOT] # plot best and current
        for case_i, case in enumerate(plot_traj_index): 
            ax1 = fig.add_subplot(gs1[0, case_i])
            ax12 = fig.add_subplot(gs1[1, case_i])
            plot_planets_trajectory(ax1, STATES[case], name_bodies, \
                                labelsize=label_size, steps = env.settings['Integration']['max_steps'], legend_on = False)
            plot_planetary_system_trajectory(ax12, STATES[case], name_bodies, \
                                labelsize=label_size, steps = env.settings['Integration']['max_steps'], legend_on = False)
            ax1.set_title(Titles[case], fontsize = label_size + 2)
            ax1.set_xlabel('x (au)', fontsize = label_size)
            ax1.set_ylabel('y (au)', fontsize = label_size)
            ax12.set_xlabel('x (au)', fontsize = label_size)
            ax12.set_ylabel('y (au)', fontsize = label_size)
            if case_i == 0: 
                legend = False
                ax1.legend(loc='upper center', bbox_to_anchor=(1, 1.9), \
                       fancybox = True, ncol = 4, fontsize = label_size-2)
            
        # Plot energy error
        x_axis = np.arange(1, len(T_comp), 1)
        ax2 = fig.add_subplot(gs1[2, :])
        ax3 = fig.add_subplot(gs1[3, :])
        ax4 = fig.add_subplot(gs1[4, :])
        ax5 = fig.add_subplot(gs1[5, :])

        plot_distance_to_one(ax2, x_axis, STATES[PLOT][1:] ) # plot for the most accurate case
        plot_distance_to_one(ax3, x_axis, STATES[0][1:], legend = False) # plot for the most accurate case
        # plot_actions_taken(ax3, x_axis, action[1:, 0]) # only for RL

        for case in range(len(STATES)):
            if case == PLOT: # make bold
                linewidth = 3
            else:
                linewidth = 3
            plot_evolution(ax4, x_axis, Energy_error[1][1:, case], label = Titles[case][1:], \
                        colorindex = case, linestyle = linestyle[case], linewidth= linewidth)
            plot_evolution(ax5, x_axis, T_comp[1:, case], label = Titles[case], \
                        colorindex = case, linestyle = linestyle[case], linewidth= linewidth)
        
        for ax in [ax2, ax3, ax4,  ax5]:
            ax.set_yscale('log')

        ax5.set_xlabel('Step', fontsize = label_size)

        ax2.set_ylabel(r'$\vert \vec r_x -\vec r_{S_5}\vert$'+'\n'+Titles[case], fontsize = label_size-4)
        # ax2.set_ylabel(r'Distance to $S_5$'+'\n'+Titles[case], fontsize = label_size-4)
        # ax3.set_ylabel(r'Distance to $S_5$'+'\n'+Titles[0], fontsize = label_size-4)
        ax3.set_ylabel(r'$\vert \vec r_x -\vec r_{S_5}\vert$'+'\n'+Titles[0], fontsize = label_size-4)
        ax4.set_ylabel(r'$\Delta E_{Total}$', fontsize = label_size)
        ax5.set_ylabel(r'$T_{Comp}$ (s)', fontsize = label_size)
        
        # ax3.legend(fontsize = label_size -3)
        ax5.legend(loc='upper center', bbox_to_anchor=(0.45, -0.38), \
                        fancybox = True, ncol = 3, fontsize = label_size-2)
        
        plt.savefig(save_path+'_%i.png'%PLOT, dpi = 150)
        plt.show()

def plot_convergence_togetherseeds(env, STATES_list, CONS_list, TCOMP_list, Titles_list, save_path, plot_traj_index = 'bestworst'):
    # Setup plot
    label_size = 20
    linestyle = ['--', ':', '-', '-', '-', '-', '-', '-', '-']


    fig = plt.figure(figsize = (10,6))
    gs1 = matplotlib.gridspec.GridSpec(1, 1, 
                                        left=0.1, wspace=0.3, hspace = 0.3, right = 0.93,
                                        top = 0.95, bottom = 0.12)
    ax = fig.add_subplot(gs1[:, :])
    seeds = len(STATES_list)
    actions = 5
    points_to_join_x = np.zeros((actions, seeds))
    points_to_join_y = np.zeros((actions, seeds))
    for j in range(seeds):
        Energy_error, T_comp, R, action = calculate_errors(STATES_list[j], CONS_list[j], TCOMP_list[j])
        x = T_comp[-1, :]
        y = Energy_error[1][-1, :]
        ax.plot(x, y, label = 'Seed %i'%(j+1), marker = 'o', color = colors[j])
        
        points_to_join_x[:, j] = x
        points_to_join_y[:, j] = y

        if j == 0:
            for i in range(len(Titles_list[j])): # annotate time-step of each point
                ax.annotate(Titles_list[j][i], (x[i], y[i]*1.2), ha = 'center',
            color = colors[j], fontsize = 12)

    
        
    for j in range(actions):
        sorted_index = np.argsort(points_to_join_y[j, :])
        a = np.take_along_axis(points_to_join_x[j, :], sorted_index, axis = 0)
        b = np.take_along_axis(points_to_join_y[j, :], sorted_index, axis = -1)
        ax.plot(a, b, linestyle = '--', color = 'black', alpha = 0.4)

        
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$\Delta E_{Total}$', fontsize = label_size)
    ax.set_xlabel(r'$T_{Comp}$ (s)', fontsize = label_size)

    ax.tick_params(axis='both', which='major', labelsize=label_size-3)
        
    ax.legend(loc='upper right',  fontsize = label_size-2)
        
    plt.savefig(save_path+'.png', dpi = 150)
    plt.show()

def plot_convergence_direct_integra_row(env, Bodies, STATES, CONS, TCOMP, Titles, save_path):
    # Setup plot
    label_size = 18
    linestyle = ['--', ':', '-', '-', '-', '-', '-', '-', '-']
    markers = ['o', 'x', 's', 'o', 'x']
    markersize = 13

    fig = plt.figure(figsize = (10,10))
    gs1 = matplotlib.gridspec.GridSpec(1, 1, 
                                        left=0.1, wspace=0.3, hspace = 0.3, right = 0.9,
                                        top = 0.95, bottom = 0.12)
    ax = fig.add_subplot(gs1[:, :])
    ax1 = ax.twinx()
    seeds = len(STATES)
    Energy_error, T_comp, R, action = calculate_errors(STATES, CONS, TCOMP)
    x = Bodies
    y = Energy_error[1][-1, :]
    y2 = T_comp[-1, :]

    print(np.shape(y))

    cases = len(STATES)//len(Bodies)
    bodies_n = len(Bodies)
    print(cases)
    print(Titles)
    for p_i in range(cases):
        ax.plot(x, y[p_i*bodies_n: bodies_n*(p_i+1)], marker = markers[p_i], markersize = markersize, \
                linestyle = linestyle[p_i], color = colors[0], label = Titles[p_i*len(Bodies)][0:-2])
        ax1.plot(x, y2[p_i*bodies_n: bodies_n*(p_i+1)], marker = markers[p_i], markersize = markersize, \
                 linestyle = linestyle[p_i], color = colors[1])
    
    ax.set_yscale('log')
    # ax1.set_yscale('log')

    ax.set_xlabel("N", fontsize = label_size)
    ax.set_ylabel(r'$\Delta E_{Total}$', fontsize = label_size, color = colors[0])
    ax1.set_ylabel(r'$T_{Comp}$ (s)', fontsize = label_size, color = colors[1])

    ax.tick_params(axis='both', which='major', labelsize=label_size-3)
    ax.tick_params(axis='y', labelcolor = colors[0])
    ax1.tick_params(axis='both', which='major', labelsize=label_size-3, labelcolor = colors[1])
        
    ax.legend(loc='upper right',  fontsize = label_size-2)
        
    plt.savefig(save_path+'.png', dpi = 150)
    plt.show()

def plot_convergence_direct_integra(env, Bodies, seeds, STATES, CONS, TCOMP, Titles, save_path):
    # Setup plot
    label_size = 18
    linestyle = ['--', ':', '-', '-', '-', '-', '-', '-', '-']
    markers = ['o', 'x', 's', '^', 'x']
    markersize = 11

    fig = plt.figure(figsize = (12,6))
    gs1 = matplotlib.gridspec.GridSpec(1, 2, 
                                        left=0.1, wspace=0.3, hspace = 0.3, right = 0.9,
                                        top = 0.95, bottom = 0.12)
    ax = fig.add_subplot(gs1[:, 0])
    ax1 = fig.add_subplot(gs1[:, 1])
    Energy_error, T_comp, R, action = calculate_errors(STATES, CONS, TCOMP)
    x = Bodies
    y = Energy_error[1][-1, :]
    y2 = T_comp[-1, :]

    print(np.shape(y))

    cases = len(STATES)//len(seeds)//len(Bodies)
    bodies_n = len(Bodies)

    print("CASES", cases, len(seeds), len(Bodies))
    for p_i in range(cases):
        y_seeds = np.zeros((len(seeds), len(Bodies)))
        y2_seeds = np.zeros((len(seeds), len(Bodies)))
        for s_i in range(len(seeds)):
            y_seeds[s_i, :] = y[s_i*(cases*len(Bodies))+p_i*bodies_n: s_i*(cases*len(Bodies)) + bodies_n*(p_i+1)]
            y2_seeds[s_i, :] = y2[s_i*(cases*len(Bodies))+p_i*bodies_n: s_i*(cases*len(Bodies)) + bodies_n*(p_i+1)]

        y_avg = np.mean(y_seeds, axis = 0)
        y_std = np.std(y_seeds, axis = 0)
        y2_avg = np.mean(y2_seeds, axis = 0)
        y2_std = np.std(y2_seeds, axis = 0)

        # ax.errorbar(x, y_avg, y_std, marker = markers[p_i], markersize = markersize, \
        #         linestyle = linestyle[p_i], color = colors[0], label = Titles[p_i*len(Bodies)][0:-2],\
        #          markeredgecolor = 'black')
        # ax1.errorbar(x, y2_avg, y2_std, marker = markers[p_i], markersize = markersize, \
        #          linestyle = linestyle[p_i], color = colors[1], markeredgecolor = 'black')

        ax.plot(x, y_avg, marker = markers[p_i], markersize = markersize, \
                linestyle = linestyle[p_i], color = colors[0], \
                    label = Titles[p_i*len(Bodies)][0:-2],\
                 markeredgecolor = 'black')
        ax1.plot(x, y2_avg, marker = markers[p_i], markersize = markersize, \
                 label = Titles[p_i*len(Bodies)][0:-2],\
                 linestyle = linestyle[p_i], color = colors[1], markeredgecolor = 'black')
    
    ax.set_yscale('log')
    ax1.set_yscale('log')
    ax.set_xscale('log')
    ax1.set_xscale('log')

    ax.set_xlabel("N", fontsize = label_size)
    ax.set_ylabel(r'$\Delta E_{Total}$', fontsize = label_size, color = colors[0])
    ax1.set_xlabel("N", fontsize = label_size)
    ax1.set_ylabel(r'$T_{Comp}$ (s)', fontsize = label_size, color = colors[1])

    ax.tick_params(axis='both', which='major', labelsize=label_size-3)
    ax.tick_params(axis='y', labelcolor = colors[0])
    ax1.tick_params(axis='both', which='major', labelsize=label_size-3)
    ax1.tick_params(axis='y', labelcolor = colors[1])
        
    ax1.legend(loc='upper left',  fontsize = label_size-2)
        
    plt.savefig(save_path+'.png', dpi = 150)
    plt.show()

def plot_trajs_reward(env, STATES, CONS, TCOMP, Titles, save_path, plot_traj_index = 'bestworst'):
    # Setup plot
    label_size = 18
    fig = plt.figure(figsize = (10,15))
    gs1 = matplotlib.gridspec.GridSpec(6, 2, 
                                    left=0.08, wspace=0.3, hspace = 0.3, right = 0.93,
                                    top = 0.88, bottom = 0.12)
    
    
    # Plot trajectories 2D
    # name_bodies = (np.arange(np.shape(STATES[0][[0]])[1])+1).astype(str)
    name_bodies = [r"$S_1$", r"$S_2$", r"$S_3$", r"$S_4$", r"$S_5$", r"$P_1$", r"$P_2$"]
    legend = True
    if plot_traj_index == 'bestworst':
        plot_traj_index = [0, len(STATES)-1] # plot best and worst
    for case_i, case in enumerate(plot_traj_index): 
        ax1 = fig.add_subplot(gs1[0, case_i])
        # ax12 = fig.add_subplot(gs1[1, case_i])
        plot_planets_trajectory(ax1, STATES[case], name_bodies, \
                            labelsize=label_size, steps = env.settings['Integration']['max_steps'], legend_on = False)
        # plot_planetary_system_trajectory(ax12, STATES[case], name_bodies, \
        #                     labelsize=label_size, steps = env.settings['Integration']['max_steps'], legend_on = False)
        ax1.set_title(Titles[case], fontsize = label_size + 2)
        ax1.set_xlabel('x (au)', fontsize = label_size)
        ax1.set_ylabel('y (au)', fontsize = label_size)
        # ax12.set_xlabel('x (au)', fontsize = label_size)
        # ax12.set_ylabel('y (au)', fontsize = label_size)
        if case == 0: 
            legend = False
            ax1.legend(loc='upper center', bbox_to_anchor=(0.8, 1.7), \
                       fancybox = True, ncol = 5, fontsize = label_size-2)
        

    # Plot energy error
    linestyle = ['--', '--', '-', '-', '-', '-', '-', '-', '-']
    Energy_error, T_comp, R, action = calculate_errors(STATES, CONS, TCOMP)
    x_axis = np.arange(1, len(T_comp), 1)
    ax2 = fig.add_subplot(gs1[1, :])
    ax3 = fig.add_subplot(gs1[2, :])
    ax4 = fig.add_subplot(gs1[3, :])
    ax5 = fig.add_subplot(gs1[4, :])
    ax6 = fig.add_subplot(gs1[5, :])

    plot_distance_to_one(ax2, x_axis, STATES[-1][1:] ) # plot for the most accurate case
    plot_actions_taken(ax3, x_axis, action[1:, 0]) # only for RL

    for case in range(len(STATES)):
        plot_evolution(ax4, x_axis, R[1:, case], label = Titles[case][1:], \
                       colorindex = case, linestyle = '-')
        plot_evolution(ax5, x_axis, Energy_error[1][1:, case], label = Titles[case][1:], \
                       colorindex = case, linestyle = '-')
        plot_evolution(ax6, x_axis, T_comp[1:, case], label = Titles[case][1:], \
                       colorindex = case, linestyle = '-')
    
    for ax in [ax2, ax5, ax6]:
        ax.set_yscale('log')

    ax4.set_yscale('symlog')

    ax6.set_xlabel('Step', fontsize = label_size)

    ax2.set_ylabel(r'$\vert \vec r_x -\vec r_S\vert$ ', fontsize = label_size)
    ax3.set_ylabel(r'Action', fontsize = label_size)
    ax4.set_ylabel(r'R', fontsize = label_size)
    ax5.set_ylabel(r'$\Delta E_{Total}$', fontsize = label_size)
    ax6.set_ylabel(r'$T_{Comp}$ (s)', fontsize = label_size)
    
    # ax3.legend(fontsize = label_size -3)
    ax6.legend(loc='upper center', bbox_to_anchor=(0.45, -0.38), \
                       fancybox = True, ncol = 3, fontsize = label_size-2)
    plt.savefig(save_path, dpi = 150)
    plt.show()

def plot_trajs(env, STATES, CONS, TCOMP, Titles, save_path, plot_traj_index = 'bestworst', subplots = 2):
    # Setup plot
    label_size = 18
    linewidth = 2
    fig = plt.figure(figsize = (10,15))
    gs1 = matplotlib.gridspec.GridSpec(6, subplots, 
                                    left=0.15, wspace=0.5, hspace = 0.5, right = 0.93,
                                    top = 0.88, bottom = 0.11)
    
    # Plot trajectories 2D
    name_bodies = (np.arange(np.shape(STATES[0][[0]])[1])+1).astype(str)
    # name_bodies = [r"$S_1$", r"$S_2$", r"$S_3$", r"$S_4$", r"$S_5$", r"$P_1$", r"$P_2$"]

    legend = True
    if plot_traj_index == 'bestworst':
        plot_traj_index = [0, len(STATES)-1] # plot best and worst
    if subplots == 3:
        plot_traj_index= np.arange(3)

    for case_i, case in enumerate(plot_traj_index): 
        ax1 = fig.add_subplot(gs1[0, case_i])
        ax12 = fig.add_subplot(gs1[1, case_i])
        plot_planets_trajectory(ax1, STATES[case], name_bodies, \
                            labelsize=label_size, steps = env.settings['Integration']['max_steps'], 
                            legend_on = False, axislabel_on = False)
        plot_planetary_system_trajectory(ax12, STATES[case], name_bodies, \
                            labelsize=label_size, steps = env.settings['Integration']['max_steps'], 
                            legend_on = False, axislabel_on = False)
        ax1.set_title(Titles[case], fontsize = label_size )
        ax1.set_xlabel('x (au)', fontsize = label_size)
        ax1.set_ylabel('y (au)', fontsize = label_size)
        ax12.set_xlabel('x (au)', fontsize = label_size)
        ax12.set_ylabel('y (au)', fontsize = label_size)

        ax1.tick_params(axis='both', which='major', labelsize=label_size-4)
        ax12.tick_params(axis='both', which='major', labelsize=label_size-4)
        
        if case_i == 0: 
            legend = False
            ax1.legend(loc='upper center', bbox_to_anchor=(1.2, 2.3), \
                       fancybox = True, ncol = 4, fontsize = label_size-2)
        

    # Plot energy error
    linestyle = ['--', '--', '-', '-', '-', '-', '-', '-', '-']
    Energy_error, T_comp, R, action = calculate_errors(STATES, CONS, TCOMP)
    x_axis = np.arange(1, len(T_comp), 1) *1e-2 # Multiplied by check step to have in myr
    ax2 = fig.add_subplot(gs1[2, :])
    ax3 = fig.add_subplot(gs1[3, :])
    ax4 = fig.add_subplot(gs1[4, :])
    ax6 = fig.add_subplot(gs1[5, :])

    plot_distance_to_one(ax2, x_axis, STATES[0][1:], legend = False) # plot for the most accurate case

    # Actions
    plot_actions_taken(ax3, x_axis, action[1:, 0]) # only for RL
    n_actions = env.settings['RL']['number_actions']
    ax3.set_ylim([-1, n_actions+1])
    labels = np.arange(n_actions)[::3]
    ax3.set_yticks(np.arange(n_actions)[::3])
    LABELS = []
    for label in labels:
        LABELS.append('%i: %.0E'%(label, env.actions[label]))
    ax3.set_yticklabels(LABELS)


    for case in range(len(STATES)):
        if case == 0:
            linestyle = '--'
        else:
            linestyle = '-'
        plot_evolution(ax4, x_axis, Energy_error[1][1:, case], label = Titles[case], \
                       colorindex = case, linestyle = linestyle, linewidth = linewidth)
        plot_evolution(ax6, x_axis, T_comp[1:, case], label = Titles[case], \
                       colorindex = case, linestyle = linestyle, linewidth = linewidth)
    
    for ax in [ax2, ax4,  ax6]:
        ax.set_yscale('log')
    for ax_i in [ax2, ax3, ax4, ax6]:
        ax_i.tick_params(axis='both', which='major', labelsize=label_size-2)

    ax6.set_xlabel('Time (Myr)', fontsize = label_size)

    # ax2.set_ylabel(r'$\Delta E_{Bridge}$ ', fontsize = label_size)
    ax2.set_ylabel(r'$\vert \vec r_x -\vec r_S\vert$ (au) ', fontsize = label_size)
    ax3.set_ylabel(r'Action taken', fontsize = label_size)
    ax4.set_ylabel(r'$\Delta E_{Total}$', fontsize = label_size)
    ax6.set_ylabel(r'$T_{Comp}$ (s)', fontsize = label_size)

    # ax3.legend(fontsize = label_size -3)
    ax6.legend(loc='upper center', bbox_to_anchor=(0.45, -0.38), \
                       fancybox = True, ncol = 3, fontsize = label_size-2)
    plt.savefig(save_path, dpi = 150)
    plt.show()

def plot_intializations(env, STATES, CONS, TCOMP, Titles, save_path):
    # Setup plot
    label_size = 18
    fig = plt.figure(figsize = (17,10))
    rows = 4
    columns = 2
    # gs1 = matplotlib.gridspec.GridSpec(rows, columns, 
    #                                 left=0.14, wspace=0.3, hspace = 0.4, right = 0.96,
    #                                 top = 0.84, bottom = 0.04)  # for vertical
    
    gs1 = matplotlib.gridspec.GridSpec(columns,rows, 
                                    left=0.1, wspace=0.35, hspace = 0.35, right = 0.96,
                                    top = 0.84, bottom = 0.06)
    # Plot trajectories 2D
    # name_bodies = (np.arange(np.shape(STATES[0][[0]])[1])+1).astype(str)
    name_bodies = [r"$S_1$", r"$S_2$", r"$S_3$", r"$S_4$", r"$S_5$", \
                   r"$S_6$", r"$S_7$", r"$S_8$", r"$S_9$", \
                   r"$P_1$", r"$P_2$",r"$P_3$", r"$P_4$", r"$P_5$", r"$P_6$", r"P_7"]
    legend = True
     # plot best and worst
    for case_i in range(rows): 
        ax1 = fig.add_subplot(gs1[0, case_i])
        ax12 = fig.add_subplot(gs1[1, case_i])
        plot_planets_trajectory(ax1, STATES[case_i], name_bodies, \
                            labelsize=label_size, steps = env.settings['Integration']['max_steps'], \
                            legend_on = False, axis = 'xy')
        plot_planetary_system_trajectory(ax12, STATES[case_i], name_bodies, \
                            labelsize=label_size, steps = env.settings['Integration']['max_steps'],\
                                  legend_on = False, axis = 'xy')
        ax1.set_title(Titles[case_i], fontsize = label_size + 2)
        # if case_i == 0: 
            # ax1.legend(loc='upper center', bbox_to_anchor=(1, 1.7), \
                    #    fancybox = True, ncol = 4, fontsize = label_size-2)
        if case_i == 1:
            ax1.legend(loc='upper center', bbox_to_anchor=(1., 1.5), \
                       fancybox = True, ncol = 4, fontsize = label_size-2)
            
        ax1.tick_params(axis='both', which='major', labelsize=label_size-3)
        ax12.tick_params(axis='both', which='major', labelsize=label_size-3)
        
    plt.savefig(save_path, dpi = 150)
    plt.show()
        
def plot_rewards(env, STATES, CONS, TCOMP, Titles, save_path):
    # Setup plot
    label_size = 18
    fig = plt.figure(figsize = (10,15))
    gs1 = matplotlib.gridspec.GridSpec(4, 2, 
                                    left=0.08, wspace=0.3, hspace = 0.3, right = 0.93,
                                    top = 0.9, bottom = 0.07)
    
    # Plot energy error
    linestyle = ['-', '--', '-.', '-', '-', '-', '-', '-', '-']
    Energy_error, Energy_error_local, T_comp, R, action= calculate_errors(STATES, CONS, TCOMP)
    x_axis = np.arange(1, len(T_comp), 1)
    ax2 = fig.add_subplot(gs1[0, :])
    ax3 = fig.add_subplot(gs1[1, :])
    ax4 = fig.add_subplot(gs1[2, :])
    ax5 = fig.add_subplot(gs1[3, :])
    for case in range(len(STATES)):
        plot_evolution(ax2, x_axis, Energy_error[1:, case], label = Titles[case][1:], \
                       colorindex = case//2, linestyle = linestyle[case%2])
        plot_evolution(ax3, x_axis, Energy_error_local[1:, case], label = Titles[case][1:], \
                       colorindex = case//2, linestyle = linestyle[case%2])
        plot_evolution(ax4, x_axis, T_comp[1:, case], label = Titles[case][1:], \
                       colorindex = case//2, linestyle = linestyle[case%2])
        plot_evolution(ax5, x_axis, R[1:, case], label = Titles[case][1:], \
                       colorindex = case//2, linestyle = linestyle[case%2])
    
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.7), \
                       fancybox = True, ncol = 3, fontsize = label_size-3)

    for ax in [ax2, ax3, ax4, ax5]:
        ax.set_yscale('log')

    ax5.set_xlabel('Step', fontsize = label_size)

    ax2.set_ylabel('Energy Error', fontsize = label_size)
    ax3.set_ylabel('Energy Error Local', fontsize = label_size)
    ax4.set_ylabel('Computation time (s)', fontsize = label_size)
    ax5.set_ylabel('Reward', fontsize = label_size)
    

    plt.savefig(save_path, dpi = 150)
    plt.show()

def calculate_rewards(E_E, E_E_local, T_comp, action, type_reward, W):
    len_array = len(E_E[2:])

    if type_reward == 0:
        # a = -(W[0]*(np.log10(abs(E_E[2:]))-np.log10(abs(E_E[1:-1]))) + \
        #     W[1]*(np.log10(abs(E_E_local[2:])))).flatten() +\
        #     np.ones(len_array) *W[2]*1/abs(np.log10(action))
    
        a = (W[0]*(np.log10(abs(E_E[2:]))-np.log10(abs(E_E[1:-1]))) + \
            W[1]*(np.log10(abs(E_E_local[2:]))-np.log10(abs(E_E_local[1:-1])))).flatten() +\
            np.ones(len_array) *W[3]*1/abs(np.log10(action))
        a = a/abs(a) * np.log(abs(a))
        
    elif type_reward == 1:
        a = (-W[0]*( np.log10(abs(E_E[2:]) + abs(E_E_local[2:]))) + \
            W[1]*(np.log10(abs(E_E[2:]))-np.log10(abs(E_E[1:-1]))) +\
            W[1]*(np.log10(abs(E_E_local[2:]))-np.log10(abs(E_E_local[1:-1]))) 
            ).flatten()
        b = np.ones(len_array) *W[3]*1/abs(np.log10(action))
        
        a += b

    elif type_reward == 2:
        a = -(W[0]*( np.log10(abs(E_E[2:]))-np.log10(abs(E_E[1:-1])) ) / np.log10(abs(E_E[1:-1])) + \
            W[1]*(np.log10(abs(E_E_local[2:]))-np.log10(abs(E_E_local[1:-1]))) / np.log10(abs(E_E_local[1:-1]))).flatten() +\
            np.ones(len_array) *W[3]*1/abs(np.log10(action))
        
        a = a/abs(a) * np.log(abs(a))

    elif type_reward == 3:
        b = np.divide( E_E[2:]-E_E[1:-1], abs(E_E[1:-1]) ).flatten()
        print(np.shape(b))
        print(np.shape(np.ones(len_array) *W[3]*1/abs(np.log10(action))))
        a = W[0]*np.divide(b, abs(b)) *np.log10(abs(b))+\
            np.ones(len_array) *W[3]*1/abs(np.log10(action))
        
        a = a/abs(a) * np.log(abs(a))

    return a

def plot_reward_comparison(env, STATES, CONS, TCOMP, Titles, save_path):
    # Setup plot
    label_size = 18
    fig = plt.figure(figsize = (10,6))

    gs1 = matplotlib.gridspec.GridSpec(1, 1, 
                                    left=0.08, wspace=0.3, hspace = 0.3, right = 0.93,
                                    top = 0.75, bottom = 0.09)
    
    # Plot energy error
    linestyle = ['-', '--', '-.', '-', '-', '-', '-', '-', '-']
    [Energy_error, Energy_error_local], T_comp, R, action = calculate_errors(STATES, CONS, TCOMP)
    x_axis = np.arange(1, len(T_comp), 1)

    W = np.array([[0, 1.0, 1.0, 10, 1e-2],
                  [1, 10.0, 1.0, 10, 1e-2],
                  [2, 1.0, 1.0, 10, 1e-2],
                  [2, 100.0, 1000.0,10, 1e-2],
                  [3, 1.0, 0.0, 0.0, 1e-2],
                  [3, 100, 0.0, 0.0, 1e-2]
                  ])
    
    ax4 = fig.add_subplot(gs1[0, :])
    
    plot_evolution(ax4, x_axis, Energy_error[1:, 0], label = r'$\Delta E$', \
                       colorindex = 0, linestyle = linestyle[0])
    plot_evolution(ax4, x_axis, Energy_error_local[1:, 0], label = r'$\Delta E_{local}$', \
                       colorindex = 0, linestyle = linestyle[1])
    # plot_evolution(ax4, x_axis, T_comp[1:, 0], label = r'$T_{comp} (s)$', \
    #                    colorindex = 0, linestyle = linestyle[2])
    
    secax_y = ax4.twinx()
    
    Rewards = np.zeros((len(W), len(Energy_error[2:, 0])+1))
    for case_i, case in enumerate(W[:, 0]):
        case = int(case)
        print(case_i)
        Rewards[case_i, 1:] = calculate_rewards(Energy_error, Energy_error_local, T_comp, \
                                    env.actions[int(action[0])], case, W[case_i, 1:])
        plot_evolution(secax_y, x_axis[1:], Rewards[case_i,1:], \
                       label = r'$R_{type}$ %i, W = %i, %i, %.0E'%(W[case_i, 0], W[case_i, 1], W[case_i, 2], W[case_i, 3]), \
                       colorindex = 1+ case_i//2, linestyle = linestyle[case_i%2])

    ax4.set_yscale('symlog', linthresh = 1e-3)
    ax4.set_ylabel('Energy Error', fontsize = label_size)
    secax_y.set_ylabel('Reward', fontsize = label_size)
    ax4.set_title('Action %.1E'%env.actions[int(action[0])])

    ax4.legend(loc='upper left', fontsize = label_size-5)
    secax_y.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), \
                       fancybox = True, ncol = 3, fontsize = label_size-5)
    ax4.set_xlabel('Step', fontsize = label_size)

    plt.savefig(save_path, dpi = 150)
    plt.show()


def plot_comparison_end(env, STATES, CONS, TCOMP, Titles, save_path, plot_traj_index = 'bestworst'):
    # Setup plot
    label_size = 18
    fig = plt.figure(figsize = (10,15))
    gs1 = matplotlib.gridspec.GridSpec(4, 2, 
                                    left=0.08, wspace=0.3, hspace = 0.3, right = 0.93,
                                    top = 0.9, bottom = 0.07)
    
    
    # Plot trajectories 2D
    # name_bodies = (np.arange(np.shape(STATES[0][[0]])[1])+1).astype(str)
    name_bodies = name_bodies = [r"$S_1$", r"$S_2$", r"$S_3$", r"$S_4$", r"$S_5$", r"$P_1$", r"$P_2$"]
    legend = True

    # Plot energy error
    linestyle = ['--', '--', '-', '-', '-', '-', '-', '-', '-']
    Energy_error, Energy_error_local, T_comp, R, action = calculate_errors(STATES, CONS, TCOMP)

    x_axis = np.arange(1, len(T_comp), 1)

    ax1 = fig.add_subplot(gs1[0, 0])
    ax12 = fig.add_subplot(gs1[0, 1])
    ax2 = fig.add_subplot(gs1[1, :])
    ax3 = fig.add_subplot(gs1[2, :])
    ax4 = fig.add_subplot(gs1[3, :])
    for case in range(len(STATES)):
        print(T_comp[-1, case], Energy_error[-1, case])
        ax1.scatter(T_comp[-1, case], Energy_error[-1, case], label = Titles[case][1:], \
                    color = colors[(case+2)%len(colors)])
        ax12.scatter(T_comp[-1, case], Energy_error_local[-1, case], label = Titles[case][1:], \
                    color = colors[(case+2)%len(colors)])
        plot_evolution(ax2, x_axis, Energy_error[1:, case], label = Titles[case][1:], \
                       colorindex = case, linestyle = linestyle[case])
        plot_evolution(ax3, x_axis, Energy_error_local[1:, case], label = Titles[case][1:], \
                       colorindex = case, linestyle = linestyle[case])
        plot_evolution(ax4, x_axis, T_comp[1:, case], label = Titles[case][1:], \
                       colorindex = case, linestyle = linestyle[case])
    
    for ax in [ax1, ax12, ax2, ax3, ax4]:
        ax.set_yscale('log')

    ax1.set_xscale('log')
    ax12.set_xscale('log')

    ax1.legend()

    ax4.set_xlabel('Step', fontsize = label_size)

    ax1.set_ylabel('Energy Error', fontsize = label_size)
    ax12.set_ylabel('Energy Error Local', fontsize = label_size)
    ax1.set_xlabel('Computation time (s)', fontsize = label_size)
    ax12.set_xlabel('Computation time (s)', fontsize = label_size)

    ax2.set_ylabel('Energy Error', fontsize = label_size)
    ax3.set_ylabel('Energy Error Local', fontsize = label_size)
    ax4.set_ylabel('Computation time (s)', fontsize = label_size)
    
    ax2.legend(fontsize = label_size -3)

    plt.savefig(save_path, dpi = 150)
    plt.show()

def plot_distance_action(env, STATES, CONS, TCOMP, Titles, save_path):
    # Setup plot
    label_size = 18
    fig = plt.figure(figsize = (10,15))
    gs1 = matplotlib.gridspec.GridSpec(3, 1, 
                                    left=0.08, wspace=0.3, hspace = 0.3, right = 0.93,
                                    top = 0.9, bottom = 0.07)
    
    
    # Plot trajectories 2D
    # name_bodies = (np.arange(np.shape(STATES[0][[0]])[1])+1).astype(str)
    # legend = True
    # if plot_traj_index == 'bestworst':
    #     plot_traj_index = [0, len(STATES)-1] # plot best and worst
    # # for case_i, case in enumerate(plot_traj_index): 
    #     ax1 = fig.add_subplot(gs1[0, case_i])
    #     ax12 = fig.add_subplot(gs1[1, case_i])
    #     plot_planets_trajectory(ax1, STATES[case], name_bodies, \
    #                         labelsize=label_size, steps = env.settings['Integration']['max_steps'], legend_on = False)
    #     plot_planetary_system_trajectory(ax12, STATES[case], name_bodies, \
    #                         labelsize=label_size, steps = env.settings['Integration']['max_steps'], legend_on = False)
    #     ax1.set_title(Titles[case], fontsize = label_size + 2)
    #     ax1.set_xlabel('x (au)', fontsize = label_size)
    #     ax1.set_ylabel('y (au)', fontsize = label_size)
    #     ax12.set_xlabel('x (au)', fontsize = label_size)
    #     ax12.set_ylabel('y (au)', fontsize = label_size)
    #     if case == 0: 
    #         legend = False
    #         ax1.legend(loc='upper center', bbox_to_anchor=(0.8, 1.7), \
    #                    fancybox = True, ncol = 3, fontsize = label_size-2)
        

    # Plot energy error
    linestyle = ['--', '--', '-', '-', '-', '-', '-', '-', '-']
    Energy_error, T_comp, R, action = calculate_errors(STATES, CONS, TCOMP)
    x_axis = np.arange(0, len(T_comp), 1)
    ax2 = fig.add_subplot(gs1[0, :])
    ax3 = fig.add_subplot(gs1[1, :])
    ax4 = fig.add_subplot(gs1[2, :])
    plot_distance_to_one(ax2, x_axis, STATES[0] ) # plot for RL one
    plot_actions_taken(ax3, x_axis, action[:, 0])
    for case in range(len(STATES)):
        plot_evolution(ax4, x_axis[1:], Energy_error[3][1:, case], label = Titles[case][1:], \
                       colorindex = case, linestyle = linestyle[case])
        # plot_evolution(ax3, x_axis, Energy_error_local[1:, case], label = Titles[case][1:], \
        #                colorindex = case, linestyle = linestyle[case])
        # plot_evolution(ax4, x_axis, T_comp[1:, case], label = Titles[case][1:], \
        #                colorindex = case, linestyle = linestyle[case])
    
    for ax in [ax2, ax4]:
        ax.set_yscale('log')

    ax4.set_xlabel('Step', fontsize = label_size)

    ax4.set_ylabel('Energy Error', fontsize = label_size)
    
    ax2.legend(fontsize = label_size -3)
    ax4.legend(fontsize = label_size -3)

    plt.savefig(save_path, dpi = 150)
    plt.show()

def plot_state_diff(env, STATES, CONS, TCOMP, Titles, save_path):
    # Setup plot
    label_size = 18
    fig = plt.figure(figsize = (10,15))
    gs1 = matplotlib.gridspec.GridSpec(5, 1, 
                                    left=0.08, wspace=0.3, hspace = 0.3, right = 0.93,
                                    top = 0.9, bottom = 0.07)
    
    # Plot energy error
    linestyle = ['--', '--', '-', '-', '-', '-', '-', '-', '-']
    Energy_error, T_comp, R, action = calculate_errors(STATES, CONS, TCOMP)
    x_axis = np.arange(0, len(T_comp), 1)
    ax1 = fig.add_subplot(gs1[0, :])
    ax2 = fig.add_subplot(gs1[1, :])
    ax3 = fig.add_subplot(gs1[2, :])
    ax4 = fig.add_subplot(gs1[3, :])
    ax5 = fig.add_subplot(gs1[4, :])
    plot_diff_state(ax1, ax2, x_axis, STATES[0], STATES[1])
    plot_distance_to_one(ax3, x_axis, STATES[0] ) # plot for RL one
    plot_distance_to_one(ax4, x_axis, STATES[1] ) # plot for most accurate one
    for case in range(len(STATES)):
        plot_evolution(ax5, x_axis[1:], Energy_error[1][1:, case], label = Titles[case][1:], \
                       colorindex = case, linestyle = linestyle[case])
        # plot_evolution(ax3, x_axis, Energy_error_local[1:, case], label = Titles[case][1:], \
        #                colorindex = case, linestyle = linestyle[case])
        # plot_evolution(ax4, x_axis, T_comp[1:, case], label = Titles[case][1:], \
        #                colorindex = case, linestyle = linestyle[case])
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_yscale('log')

    ax1.set_ylabel('r_{RL}- r_0', fontsize = label_size)
    ax2.set_ylabel('v_{RL}- v_0', fontsize = label_size)
    ax3.set_ylabel('Distance to particle for RL', fontsize = label_size)
    ax4.set_ylabel('Distance to particle for 0', fontsize = label_size)
    ax5.set_ylabel('Energy Error', fontsize = label_size)
    ax5.set_xlabel('Step', fontsize = label_size)
    
    ax2.legend(fontsize = label_size -3)
    ax4.legend(fontsize = label_size -3)

    plt.savefig(save_path, dpi = 150)
    plt.show()


def plot_energy_vs_tcomp(env, STATES, cons, tcomp, Titles, initializations, save_path, plot_traj_index = [0,1,2]):
    """
    plot_EvsTcomp: plot energy error vs computation time for different cases
    INPUTS:
        values: list of the actions taken for each case
        initializations: number of initializations used to generate the data
        steps: steps taken
        env: environment used
    """
    cases = len(STATES)

    fig = plt.figure(figsize = (6,6))
    gs1 = matplotlib.gridspec.GridSpec(2, 2, figure = fig, width_ratios = (3, 1), height_ratios = (1, 3), \
                                       left=0.15, wspace=0.3, 
                                       hspace = 0.2, right = 0.99,
                                        top = 0.97, bottom = 0.11)
    
    msize = 50
    alphavalue = 0.5
    alphavalue2 = 0.9
    
    ax1 = fig.add_subplot(gs1[1, 0]) 
    ax2 = fig.add_subplot(gs1[0, 0])
    ax3 = fig.add_subplot(gs1[1, 1])

    markers = ['o', 'x', 's', 'o', 'x', '^']
    order = [1,2,0, 3, 4, 5, 6]
    alpha = [0.5, 0.5, 0.9, 0.8, 0.9, 0.7]


    # Calculate the energy errors
    E_T = np.zeros((initializations, len(plot_traj_index)))
    E_B = np.zeros((initializations, len(plot_traj_index)))
    T_c = np.zeros((initializations, len(plot_traj_index)))
    # Labels = np.zeros((initializations, len(plot_traj_index)))
    nsteps_perepisode = np.zeros((initializations, len(plot_traj_index)))
    

    for act in range(len(plot_traj_index)):
        for i in range(initializations):
            nsteps_perepisode = len(cons[plot_traj_index[act]*initializations +i][:,0])
            E_B[i, act] = abs(cons[plot_traj_index[act]*initializations + i][2, -1]) # absolute relative energy error
            E_T[i, act] = abs(cons[plot_traj_index[act]*initializations + i][3, -1]) # absolute relative energy error

            T_c[i, act] = np.sum(tcomp[plot_traj_index[act]*initializations + i])/nsteps_perepisode # add individual computation times
            # Labels[i, act] = plot_traj_index[act]

    def minimum(a, b):
        if a <= b:
            return a
        else:
            return b
        
    def maximum(a, b):
        if a >= b:
            return a
        else:
            return b
        
    min_x = 10
    min_y = 10
    max_x = 0
    max_y = 0 #random initial values
    for i in range(len(plot_traj_index)):  
        X = T_c[:, i]
        Y = E_B[:, i]
        ax1.scatter(X, Y, color = colors[i], alpha = alphavalue, marker = markers[i],\
                s = msize, label = Titles[plot_traj_index[i]], zorder =order[i])
        min_x = minimum(min_x, min(X))
        min_y = minimum(min_y, min(Y))
        max_x = maximum(max_x, max(X))
        max_y = maximum(max_y, max(Y))
    binsx = np.logspace(np.log10(min_x),np.log10(max_x), 50)
    binsy = np.logspace(np.log10(min_y),np.log10(max_y), 50)  

    for i in range(len(plot_traj_index)):  
        X = T_c[:, i]
        Y = E_B[:, i]
        ax2.hist(X, bins = binsx, color = colors[i],  alpha = alpha[i], edgecolor=colors[i], \
                 linewidth=1.2, zorder =order[i])
        ax2.set_yscale('log')
        ax3.hist(Y, bins = binsy, color = colors[i], alpha = alpha[i], orientation='horizontal',\
                 edgecolor=colors[i], linewidth=1.2, zorder =order[i])
    
    labelsize = 12
    ax1.legend(fontsize = labelsize)
    ax1.set_xlabel('Total computation time (s)',  fontsize = labelsize)
    ax1.set_ylabel('Final Energy Error',  fontsize = labelsize)
    ax1.set_yscale('log')
    ax3.set_yscale('log')
    # ax1.set_xlim([5e-1, 3])
    # ax2.set_xlim([5e-1, 3])
    # ax1.set_ylim([1e-14, 1e1])
    # ax3.set_ylim([1e-14, 1e1])
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='both', which='major', labelsize=labelsize)
    
    plt.savefig(save_path+'tcomp_vs_Eerror_evaluate.png', dpi = 100)
    plt.show()