"""
TestEnvironment: tests simulation environments

Author: Veronica Saz Ulibarrena
Last modified: 5-March-2025
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from PlotsFunctions import plot_planetary_system_trajectory, plot_planets_trajectory,\
    plot_evolution, plot_distance_to_one, plot_actions_taken, plot_diff_state, calculate_planet_elements, \
    plot_evolution_keplerian, eliminate_escaped_planets, plot_hydro_system_trajectory, \
     plot_distance_to_one_tree

colors = ['steelblue', 'darkgoldenrod', 'seagreen', 'coral',  \
        'mediumslateblue', 'deepskyblue', 'blue', 'black', 'red']
colors2 = ['navy']
lines = ['-', '--', ':', '-.' ]
markers = ['o', 'x', '.', '^', 's']

def calculate_errors(states, cons, tcomp, steps = None):
    """
    calculate_errors: from files, calculate reward, energy error...
    INPUTS:
        states: states file
        cons: cons file
        tcom: tcomp file
        steps: number of steps to extract
    OUTPUTS:
        E_T_rel, E_T: test energy errors relative and abs
        T_c: computation time
        R: reward
        Action: action chosen
        Action_H: action for hybrid method
    """
    cases = len(states)
    if steps == None:
        steps = np.shape(cons[0][:, 0])[0]

    # Calculate the energy errors
    R = np.zeros((steps, cases))
    E_T = np.zeros((steps, cases))
    E_T_rel = np.zeros((steps, cases))
    T_c = np.zeros((steps, cases))
    Action = np.zeros((steps, cases))
    Action_H = np.zeros((steps, cases))
    for i in range(cases):
        R[1:, i] = cons[i][1:steps, 1]
        E_T_rel[1:, i] = abs(cons[i][1:steps, 2]) # absolute relative energy error
        E_T[1:, i] = abs(cons[i][1:steps, 3]) # absolute relative energy error
        T_c[1:, i] = np.cumsum(tcomp[i][1:steps]) # add individual computation times
        Action[1:, i] = cons[i][1:steps, 0]
        if len(cons[i][0,:])>4:
            Action_H[1:, i] = cons[i][1:steps, 4]

    return [E_T_rel, E_T], T_c, R, Action, Action_H

def plot_convergence(env, STATES, CONS, TCOMP, Titles, save_path, plot_traj_index = 'bestworst'):
    """
    plot_convergence: plot convergence
    INPUTS: 
        env: environment
        STATES: states file for all cases
        CONS: cons file for all cases
        TCOMP: tcomp file for all acases
        Titles: name of each case
        save_path: path to save plot
        plot_traj_index: indexes for plotting scatter
    """
    # Setup plot
    label_size = 18
    linestyle = ['--', ':', '-', '-', '-', '-', '-', '-', '-']
    Energy_error, T_comp, R, action, action_H = calculate_errors(STATES, CONS, TCOMP)

    for PLOT in range(len(STATES)):
        fig = plt.figure(figsize = (10,15))
        gs1 = matplotlib.gridspec.GridSpec(6, 2, 
                                        left=0.1, wspace=0.3, hspace = 0.3, right = 0.93,
                                        top = 0.88, bottom = 0.12)
        
        # Plot trajectories 2D
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
        ax3.set_ylabel(r'$\vert \vec r_x -\vec r_{S_5}\vert$'+'\n'+Titles[0], fontsize = label_size-4)
        ax4.set_ylabel(r'$\Delta E_{Total}$', fontsize = label_size)
        ax5.set_ylabel(r'$T_{Comp}$ (s)', fontsize = label_size)
        
        ax5.legend(loc='upper center', bbox_to_anchor=(0.45, -0.38), \
                        fancybox = True, ncol = 3, fontsize = label_size-2)
        
        plt.savefig(save_path+'_%i.png'%PLOT, dpi = 150)
        plt.show()

def plot_convergence_togetherseeds(env, STATES_list, CONS_list, TCOMP_list, Titles_list, save_path, plot_traj_index = 'bestworst'):
    """
    plot_convergence_togetherseeds: plot convergence for multiple seeds
    INPUTS: 
        env: environment
        STATES_list: states file for all cases
        CONS_list: cons file for all cases
        TCOMP_list: tcomp file for all acases
        Titles_list: name of each case
        save_path: path to save plot
        plot_traj_index: indexes for plotting scatter
    """
    # Setup plot
    label_size = 22
    linestyle = ['--', ':', '-', '-', '-', '-', '-', '-', '-']


    fig = plt.figure(figsize = (15,8))
    gs1 = matplotlib.gridspec.GridSpec(1, 1, 
                                        left=0.1, wspace=0.3, hspace = 0.3, right = 0.93,
                                        top = 0.95, bottom = 0.12)
    ax = fig.add_subplot(gs1[:, :])
    seeds = len(STATES_list)
    actions = 5
    points_to_join_x = np.zeros((actions, seeds))
    points_to_join_y = np.zeros((actions, seeds))
    for j in range(seeds):
        Energy_error, T_comp, R, action, action_H = calculate_errors(STATES_list[j], CONS_list[j], TCOMP_list[j])
        x = T_comp[-1, :]
        y = Energy_error[1][-1, :]
        ax.plot(x, y, label = 'Seed %i'%(j+1), marker = 'o', markersize = 12, color = colors[j])
        
        points_to_join_x[:, j] = x
        points_to_join_y[:, j] = y

        if j == 0:
            for i in range(len(Titles_list[j])): # annotate time-step of each point
                ax.annotate(Titles_list[j][i], (x[i], y[i]*1.3), ha = 'center',
            color = colors[j], fontsize = 14,
            bbox=dict(facecolor='w', alpha=0.7))
        
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
    ax.grid(axis = 'y', alpha = 0.5)
        
    plt.savefig(save_path+'.png', dpi = 150)
    plt.show()

def plot_convergence_direct_integra_row(env, Bodies, STATES, CONS, TCOMP, Titles, save_path):
    """
    plot_convergence_direct_integra_row: plot convergence for direct integration comparison in a row
    INPUTS: 
        env: environment
        Bodies: number of bodies
        STATEs: states file for all cases
        CONS: cons file for all cases
        TCOMP: tcomp file for all acases
        Titles: name of each case
        save_path: path to save plot
    """
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
    Energy_error, T_comp, R, action, action_H = calculate_errors(STATES, CONS, TCOMP)
    x = Bodies
    y = Energy_error[1][-1, :]
    y2 = T_comp[-1, :]

    cases = len(STATES)//len(Bodies)
    bodies_n = len(Bodies)
    for p_i in range(cases):
        ax.plot(x, y[p_i*bodies_n: bodies_n*(p_i+1)], marker = markers[p_i], markersize = markersize, \
                linestyle = linestyle[p_i], color = colors[0], label = Titles[p_i*len(Bodies)][0:-2])
        ax1.plot(x, y2[p_i*bodies_n: bodies_n*(p_i+1)], marker = markers[p_i], markersize = markersize, \
                 linestyle = linestyle[p_i], color = colors[1])
    
    ax.set_yscale('log')
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
    """
    plot_convergence_direct_integra_row: plot convergence for direct integration comparison
    INPUTS: 
        env: environment
        Bodies: number of bodies
        STATEs: states file for all cases
        CONS: cons file for all cases
        TCOMP: tcomp file for all acases
        Titles: name of each case
        save_path: path to save plot
    """
    # Setup plot
    label_size = 18
    linestyle = ['-', '-', ':', '-', ':', '-', '-', '-', '-']
    markers = ['o', 'x', 'x', '^', '^']
    markersize = 11

    fig = plt.figure(figsize = (12,6))
    gs1 = matplotlib.gridspec.GridSpec(1, 2, 
                                        left=0.1, wspace=0.3, hspace = 0.3, right = 0.9,
                                        top = 0.95, bottom = 0.12)
    ax = fig.add_subplot(gs1[:, 0])
    ax1 = fig.add_subplot(gs1[:, 1])
    Energy_error, T_comp, R, action, action_H = calculate_errors(STATES, CONS, TCOMP)
    x = Bodies
    y = Energy_error[1][-1, :]
    y2 = T_comp[-1, :]

    cases = len(STATES)//len(seeds)//len(Bodies)
    bodies_n = len(Bodies)

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

        ax.plot(x, y_avg, marker = markers[p_i], markersize = markersize, \
                linestyle = linestyle[p_i], color = colors[0], \
                    label = Titles[p_i*len(Bodies)][0:-2],\
                 markeredgecolor = 'black')
        ax1.plot(x, y2_avg, marker = markers[p_i], markersize = markersize, \
                 label = Titles[p_i*len(Bodies)][0:-2],\
                 linestyle = linestyle[p_i], color = colors[1], markeredgecolor = 'black')
    
    ax.set_yscale('log')

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
    """
    plot_trajs_reward: plot trajectories, actions and reward main plot
    INPUTS: 
        env: environment
        STATEs: states file for all cases
        CONS: cons file for all cases
        TCOMP: tcomp file for all acases
        Titles: name of each case
        save_path: path to save plot
        plot_traj_index: indexes to plot
    """
    # Setup plot
    label_size = 18
    fig = plt.figure(figsize = (10,15))
    gs1 = matplotlib.gridspec.GridSpec(6, 2, 
                                    left=0.08, wspace=0.3, hspace = 0.3, right = 0.93,
                                    top = 0.88, bottom = 0.12)

    # Plot trajectories 2D
    name_bodies = [r"$S_1$", r"$S_2$", r"$S_3$", r"$S_4$", r"$S_5$", r"$P_1$", r"$P_2$"]
    legend = True
    if plot_traj_index == 'bestworst':
        plot_traj_index = [0, len(STATES)-1] # plot best and worst
    for case_i, case in enumerate(plot_traj_index): 
        ax1 = fig.add_subplot(gs1[0, case_i])
        plot_planets_trajectory(ax1, STATES[case], name_bodies, \
                            labelsize=label_size, steps = env.settings['Integration']['max_steps'], legend_on = False)    
        ax1.set_title(Titles[case], fontsize = label_size + 2)
        ax1.set_xlabel('x (au)', fontsize = label_size)
        ax1.set_ylabel('y (au)', fontsize = label_size)
        if case == 0: 
            legend = False
            ax1.legend(loc='upper center', bbox_to_anchor=(0.8, 1.7), \
                       fancybox = True, ncol = 5, fontsize = label_size-2)

    # Plot energy error
    linestyle = ['--', '--', '-', '-', '-', '-', '-', '-', '-']
    Energy_error, T_comp, R, action, action_H = calculate_errors(STATES, CONS, TCOMP)
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
    
    ax6.legend(loc='upper center', bbox_to_anchor=(0.45, -0.38), \
                       fancybox = True, ncol = 3, fontsize = label_size-2)
    plt.savefig(save_path, dpi = 150)
    plt.show()

def plot_trajs(env, STATES, CONS, TCOMP, Titles, save_path, plot_traj_index = 'bestworst',
                subplots = 2, hybrid = False):
    """
    plot_trajs: plot trajectories, actions and reward main plot
    INPUTS: 
        env: environment
        STATEs: states file for all cases
        CONS: cons file for all cases
        TCOMP: tcomp file for all acases
        Titles: name of each case
        save_path: path to save plot
        plot_traj_index: indexes to plot
        hybrid: true or false whether the case is included
    """
    # Setup plot
    label_size = 18
    linewidth = 2
    fig = plt.figure(figsize = (10,15))
    gs1 = matplotlib.gridspec.GridSpec(6, subplots, 
                                    left=0.15, wspace=0.5, hspace = 0.5, right = 0.93,
                                    top = 0.85, bottom = 0.11)
    
    # Plot trajectories 2D
    name_bodies = (np.arange(np.shape(STATES[0][[0]])[1])+1).astype(str)
    Energy_error, T_comp, R, action, action_H = calculate_errors(STATES, CONS, TCOMP)

    legend = True
    if plot_traj_index == 'bestworst':
        plot_traj_index = [0, len(STATES)-1] # plot best and worst
    elif plot_traj_index == 'RLbest':
        last_E_error = Energy_error[0][-1, 1:]
        best_index = np.where(last_E_error == min(last_E_error))[0][0]
        plot_traj_index = [0, best_index+1]
    elif plot_traj_index == 'HRLbest':
        last_E_error = Energy_error[0][-1, 2:]
        best_index = np.where(last_E_error == min(last_E_error))[0][0]
        plot_traj_index = [1, best_index+2]

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
            ax1.legend(loc='upper center', bbox_to_anchor=(1.2, 2.7), \
                       fancybox = True, ncol = 4, fontsize = label_size-2)
        
    # Plot energy error
    linestyle = ['--', '--', '-', '-', '-', '-', '-', '-', '-']
    x_axis = np.arange(1, len(T_comp), 1) *1e-2 # Multiplied by check step to have in myr
    ax2 = fig.add_subplot(gs1[2, :])
    ax3 = fig.add_subplot(gs1[3, :])
    ax4 = fig.add_subplot(gs1[4, :])
    ax6 = fig.add_subplot(gs1[5, :])

    plot_distance_to_one(ax2, x_axis, STATES[plot_traj_index[0]][1:], legend = False) # plot for the most accurate case

    # Actions
    plot_actions_taken(ax3, x_axis, action[1:, 0], label = Titles[0]) # only for RL
    if hybrid == True:
        plot_actions_taken(ax3, x_axis, action[1:, 1], action_H = action_H[1:, 1],\
                       color = 'darkred', marker = '.', label = Titles[1]) # only for H-RL
        ax3.legend(fontsize = label_size-4, ncol = 2, loc ='best')

    n_actions = env.settings['RL']['number_actions']
    ax3.set_ylim([-1, n_actions+1])
    labels = np.arange(n_actions)[::3]
    ax3.set_yticks(np.arange(n_actions)[::3])
    LABELS = []
    for label in labels:
        LABELS.append('%i: %.0E'%(label, env.actions[label]))
    ax3.set_yticklabels(LABELS)

    for case in range(len(STATES)):
        # make line thicker for RL cases
        if case == 0:
            linestyle = '-'
            linewidth = 3
            alpha = 1
        elif case == 1 and hybrid == True:
            linestyle = '-'
            linewidth = 2.5
            alpha = 1
        else:
            linestyle = '-.'
            linewidth = 2
            alpha = 0.8

        if hybrid == True and case == 1:
            colorindex = len(STATES)-1
        elif hybrid == True and case >1:
            colorindex = case-1
        else:
            colorindex = case

        plot_evolution(ax4, x_axis, Energy_error[0][1:, case], label = Titles[case], \
                    colorindex = colorindex, linestyle = linestyle, linewidth = linewidth,
                    alpha = alpha)
        plot_evolution(ax6, x_axis, T_comp[1:, case], label = Titles[case], \
                    colorindex = colorindex, linestyle = linestyle, linewidth = linewidth,
                    alpha = alpha)
    
    for ax in [ax2, ax4,  ax6]:
        ax.set_yscale('log')
    for ax_i in [ax2, ax3, ax4, ax6]:
        ax_i.tick_params(axis='both', which='major', labelsize=label_size-2)
        ax_i.tick_params(axis='both', which='minor', labelsize=label_size-2)

    ax4.set_yticks( [1e-5, 1e-3, 1e-1] )
    ax6.set_xlabel('Time (Myr)', fontsize = label_size)
    ax2.set_ylabel(r'$\vert \vec r_x -\vec r_S\vert$ (au) ', fontsize = label_size)
    ax3.set_ylabel(r'Action taken', fontsize = label_size)
    ax4.set_ylabel(r'$\Delta E_{Total}$', fontsize = label_size)
    ax6.set_ylabel(r'$T_{Comp}$ (s)', fontsize = label_size)

    ax6.legend(loc='upper center', bbox_to_anchor=(0.45, -0.4), \
                       fancybox = True, ncol = 3, fontsize = label_size-2)
    plt.savefig(save_path, dpi = 150)
    # plt.show()

def plot_trajs_tree(env, STATES, CONS, TCOMP, Titles, save_path, plot_traj_index = 'bestworst',
                subplots = 2, hybrid = False):
    """
    plot_trajs_tree: plot trajectories, actions and reward main plot for the case with a big tree integrator
    INPUTS: 
        env: environment
        STATEs: states file for all cases
        CONS: cons file for all cases
        TCOMP: tcomp file for all acases
        Titles: name of each case
        save_path: path to save plot
        plot_traj_index: indexes to plot
        hybrid: true or false whether the case is included
    """
    # Setup plot
    label_size = 18
    linewidth = 2
    fig = plt.figure(figsize = (10,15))
    gs1 = matplotlib.gridspec.GridSpec(6, subplots, 
                                    left=0.15, wspace=0.5, hspace = 0.5, right = 0.93,
                                    top = 0.97, bottom = 0.13)
    
    # Plot trajectories 2D
    name_bodies = (np.arange(np.shape(STATES[0][[0]])[1])+1).astype(str)
    Energy_error, T_comp, R, action, action_H = calculate_errors(STATES, CONS, TCOMP)

    legend = True
    if plot_traj_index == 'bestworst':
        plot_traj_index = [0, len(STATES)-1] # plot best and worst
    elif plot_traj_index == 'RLbest':
        last_E_error = Energy_error[0][-1, 1:]
        best_index = np.where(last_E_error == min(last_E_error))[0][0]
        plot_traj_index = [0, best_index+1]
    elif plot_traj_index == 'HRLbest':
        last_E_error = Energy_error[0][-1, 2:]
        best_index = np.where(last_E_error == min(last_E_error))[0][0]
        plot_traj_index = [1, best_index+2]

    if subplots == 3:
        plot_traj_index= np.arange(3)

    case = plot_traj_index[0]
    ax1 = fig.add_subplot(gs1[0, :])
    ax12 = fig.add_subplot(gs1[1, 0])
    ax13 = fig.add_subplot(gs1[1, 1])
    plot_planets_trajectory(ax1, STATES[case]/1000, name_bodies, \
                        labelsize=label_size, steps = env.settings['Integration']['max_steps'], 
                        legend_on = False, axislabel_on = False)
    plot_planets_trajectory(ax12, STATES[case]/1000, name_bodies, \
                        labelsize=label_size, steps = env.settings['Integration']['max_steps'], 
                        legend_on = False, axislabel_on = False)
    plot_planetary_system_trajectory(ax13, STATES[case], name_bodies, \
                        labelsize=label_size, steps = env.settings['Integration']['max_steps'], 
                        legend_on = False, axislabel_on = False)
    ax1.set_title(Titles[case], fontsize = label_size )
    ax1.set_xlabel(r'x (1000$\times$ au)', fontsize = label_size)
    ax1.set_ylabel('y (au)', fontsize = label_size)
    ax12.set_xlabel(r'x (1000$\times$ au)', fontsize = label_size)
    ax12.set_ylabel('y (au)', fontsize = label_size)
    ax13.set_xlabel('x (au)', fontsize = label_size)
    ax13.set_ylabel('y (au)', fontsize = label_size)

    ax1.set_xlim((-450, 500))
    ax1.set_ylim((-450, 450))
    ax12.set_xlim((-70, 70))
    ax12.set_ylim((-70, 70))

    ax1.tick_params(axis='both', which='major', labelsize=label_size-4)
    ax12.tick_params(axis='both', which='major', labelsize=label_size-4)
    ax13.tick_params(axis='both', which='major', labelsize=label_size-4)
        
    # Plot energy error
    linestyle = ['--', '--', '-', '-', '-', '-', '-', '-', '-']
    x_axis = np.arange(1, len(T_comp), 1) *1e-2 # Multiplied by check step to have in myr
    ax2 = fig.add_subplot(gs1[2, :])
    ax3 = fig.add_subplot(gs1[3, :])
    ax4 = fig.add_subplot(gs1[4, :])
    ax6 = fig.add_subplot(gs1[5, :])

    plot_distance_to_one_tree(ax2, x_axis, STATES[plot_traj_index[0]][1:], legend = False) # plot for the most accurate case

    # Actions
    plot_actions_taken(ax3, x_axis, action[1:, 0], label = Titles[0]) # only for RL
    # if hybrid == True:
    #     plot_actions_taken(ax3, x_axis, action[1:, 1], action_H = action_H[1:, 1],\
    #                    color = 'darkred', marker = '.', label = Titles[1]) # only for H-RL
    #     ax3.legend(fontsize = label_size-4, ncol = 2, loc ='best')

    n_actions = env.settings['RL']['number_actions']
    ax3.set_ylim([-1, n_actions+1])
    labels = np.arange(n_actions)[::3]
    ax3.set_yticks(np.arange(n_actions)[::3])
    LABELS = []
    for label in labels:
        LABELS.append('%i: %.0E'%(label, env.actions[label]))
    ax3.set_yticklabels(LABELS)

    for case in range(len(STATES)):
        # make line thicker for RL cases
        if case == 0:
            linestyle = '-'
            linewidth = 3
            alpha = 1
        elif case == 1 and hybrid == True:
            linestyle = '-'
            linewidth = 2.5
            alpha = 1
        else:
            linestyle = '-.'
            linewidth = 2
            alpha = 0.8

        if hybrid == True and case == 1:
            colorindex = len(STATES)-1
        elif hybrid == True and case >1:
            colorindex = case-1
        else:
            colorindex = case

        plot_evolution(ax4, x_axis, Energy_error[0][1:, case], label = Titles[case], \
                    colorindex = colorindex, linestyle = linestyle, linewidth = linewidth,
                    alpha = alpha)
        plot_evolution(ax6, x_axis, T_comp[1:, case], label = Titles[case], \
                    colorindex = colorindex, linestyle = linestyle, linewidth = linewidth,
                    alpha = alpha)
    
    for ax in [ax2, ax4,  ax6]:
        ax.set_yscale('log')
    for ax_i in [ax2, ax3, ax4, ax6]:
        ax_i.tick_params(axis='both', which='major', labelsize=label_size-2)
        ax_i.tick_params(axis='both', which='minor', labelsize=label_size-2)

    ax4.set_ylim((2e-1, 4.5e-1)) # for tree code example
    ax6.set_xlabel('Time (Myr)', fontsize = label_size)
    ax2.set_ylabel(r'$\vert \vec r_x -\vec r_S\vert$ (au) ', fontsize = label_size)
    ax3.set_ylabel(r'Action taken', fontsize = label_size)
    ax4.set_ylabel(r'$\Delta E_{Total}$', fontsize = label_size)
    ax6.set_ylabel(r'$T_{Comp}$ (s)', fontsize = label_size)
    ax6.legend(loc='upper center', bbox_to_anchor=(0.45, -0.4), \
                       fancybox = True, ncol = 3, fontsize = label_size-2)
    plt.savefig(save_path, dpi = 150)
    # plt.show()

def plot_trajs_noRL(env, STATES, CONS, TCOMP, Titles, save_path, plot_traj_index = 'bestworst',
                subplots = 2, hybrid = False):
    """
    plot_trajs_noRL: plot trajectories not including RL, actions and reward main plot for the case with a big tree integrator
    INPUTS: 
        env: environment
        STATEs: states file for all cases
        CONS: cons file for all cases
        TCOMP: tcomp file for all acases
        Titles: name of each case
        save_path: path to save plot
        plot_traj_index: indexes to plot
        hybrid: true or false whether the case is included
    """
    # Setup plot
    label_size = 18
    linewidth = 2
    fig = plt.figure(figsize = (10,15))
    gs1 = matplotlib.gridspec.GridSpec(4, subplots, 
                                    left=0.15, wspace=0.7, hspace = 0.5, right = 0.95,
                                    top = 0.85, bottom = 0.11)
    
    # Plot trajectories 2D
    name_bodies = (np.arange(np.shape(STATES[0][[0]])[1])+1).astype(str)
    Energy_error, T_comp, R, action, action_H = calculate_errors(STATES, CONS, TCOMP)

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
        ax12.set_xlabel('x (au)', fontsize = label_size)
        if case_i == 0:
            ax1.set_ylabel('y (au)', fontsize = label_size)
            ax12.set_ylabel('y (au)', fontsize = label_size)

        ax1.tick_params(axis='both', which='major', labelsize=label_size-4)
        ax12.tick_params(axis='both', which='major', labelsize=label_size-4)
        
        if case_i == 1: 
            legend = False
            ax1.legend(loc='upper center', bbox_to_anchor=(0.45, 2.0), \
                       fancybox = True, ncol = 4, fontsize = label_size-2)
        
    # Plot energy error
    linestyle = ['--', '--', '-', '-', '-', '-', '-', '-', '-']
    x_axis = np.arange(1, len(T_comp), 1) *1e-2 # Multiplied by check step to have in myr
    ax4 = fig.add_subplot(gs1[2, :])
    ax6 = fig.add_subplot(gs1[3, :])

    for case in range(len(STATES)):
        if case == 0:
            linestyle = '--'
        else:
            linestyle = '-'
        plot_evolution(ax4, x_axis, Energy_error[1][1:, case], label = Titles[case], \
                       colorindex = case, linestyle = linestyle, linewidth = linewidth)
        plot_evolution(ax6, x_axis, T_comp[1:, case], label = Titles[case], \
                       colorindex = case, linestyle = linestyle, linewidth = linewidth)
    
    for ax in [ax4,  ax6]:
        ax.set_yscale('log')
    for ax_i in [ax4, ax6]:
        ax_i.tick_params(axis='both', which='major', labelsize=label_size-2)

    ax4.set_xlabel('Time (Myr)', fontsize = label_size)
    ax6.set_xlabel('Time (Myr)', fontsize = label_size)
    ax4.set_ylabel(r'$\Delta E_{Total}$', fontsize = label_size)
    ax6.set_ylabel(r'$T_{Comp}$ (s)', fontsize = label_size)

    ax6.legend(loc='upper center', bbox_to_anchor=(0.45, -0.4), \
                       fancybox = True, ncol = 3, fontsize = label_size-2)
    plt.savefig(save_path, dpi = 150)
    # plt.show()

def plot_a_vs_e(env, STATES, CONS, TCOMP, Titles, save_path, plot_traj_index = 'bestworst', hybrid = False):
    """
    plot_a_vs_e: plot eccentricity vs a including energy error
    INPUTS: 
        env: environment
        STATEs: states file for all cases
        CONS: cons file for all cases
        TCOMP: tcomp file for all acases
        Titles: name of each case
        save_path: path to save plot
        plot_traj_index: indexes to plot
        hybrid: true or false whether the case is included
    """
    # Setup plot
    label_size = 18
    linewidth = 2
    fig = plt.figure(figsize = (10,15))
    gs1 = matplotlib.gridspec.GridSpec(4,5 ,
                                    left=0.12, wspace=1.5, hspace = 0.5, right = 0.93,
                                    top = 0.98, bottom = 0.14)
    
    ax1 = fig.add_subplot(gs1[0, 0:3])
    ax2 = fig.add_subplot(gs1[1, 0:3])
    ax3 = fig.add_subplot(gs1[1, 3:])
    ax4 = fig.add_subplot(gs1[2, 0:3])
    ax5 = fig.add_subplot(gs1[2, 3:])
    ax6 = fig.add_subplot(gs1[3, 0:3])
    ax7 = fig.add_subplot(gs1[3, 3:])
    AX = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
    
    # Plot trajectories 2D
    name_bodies = (np.arange(np.shape(STATES[0][[0]])[1])+1).astype(str)
    Energy_error, T_comp, R, action, action_H = calculate_errors(STATES, CONS, TCOMP)
    steps = len(Energy_error[1][:, 0])

    # Plot energy error
    linestyle = ['--', '--', '-', '-', '-', '-', '-', '-', '-']
    x_axis = np.arange(1, len(T_comp), 1) *1e-2 # Multiplied by check step to have in myr
    
    for case in range(len(STATES)):
        if case == 0:
            linestyle = '-'
            linewidth = 3
            alpha = 1
        else:
            linestyle = '-.'
            linewidth = 2
            alpha = 0.8
            
        plot_evolution(ax1, x_axis, Energy_error[1][1:, case], label = Titles[case], \
                       colorindex = case, linestyle = linestyle, linewidth = linewidth)
        
        planets_elements = calculate_planet_elements(STATES[case], steps)

        for planet_plot in range(3): # semi-major axis and distance
            # planet_plot 1 to not included the central star
            norm_dist = np.linalg.norm(planets_elements[1:, planet_plot+1, 0:3], axis = 1)
            plot_evolution(AX[2*planet_plot+1], x_axis, norm_dist, label = Titles[case], \
                       colorindex = case, linestyle = linestyle, linewidth = linewidth,alpha = 0.7)
            
            plot_evolution_keplerian(AX[2*planet_plot+2], planets_elements[1:, planet_plot+1, 6], \
                        planets_elements[1:, planet_plot+1, 7],
                        label = Titles[case], alpha = 0.5,\
                       colorindex = case, linestyle = linestyle, linewidth = linewidth)

    for ax in [ax1]:
        ax.set_yscale('log')
        ax.set_xlabel('Time (Myr)', fontsize = label_size)
    for ax_i in AX:
        ax_i.tick_params(axis='both', which='major', labelsize=label_size-2)
        ax_i.tick_params(axis='both', which='minor', labelsize=label_size-2)
    for i, ax_i in enumerate([ax2, ax4, ax6]):
        ax_i.set_ylabel(r'$\vert \vec r_p - \vec r_S\vert$ (au)', fontsize = label_size)
        ax_i.set_title("Planet %i"%(i+1), fontsize = label_size+2)
        ax_i.set_xlabel('Time (Myr)', fontsize = label_size)
    for ax_i in [ax3, ax5, ax7]:
        ax_i.set_xlabel(r'$a$ (au)', fontsize = label_size)
        ax_i.set_ylabel(r'$e$ ', fontsize = label_size)

    ax1.set_ylabel(r'$\Delta E_{Total}$', fontsize = label_size)
    ax6.legend(loc='upper center', bbox_to_anchor=(0.85, -0.4), \
                       fancybox = True, ncol = 3, fontsize = label_size-2)
    plt.savefig(save_path, dpi = 150)
    # plt.show()

def plot_trajs_hydro(env, STATES, CONS, TCOMP, Titles, save_path, plot_traj_index = 'bestworst',
                subplots = 2, hybrid = False):
    """
    plot_trajs_hydro: plot trajectories for hydro case
    INPUTS: 
        env: environment
        STATEs: states file for all cases
        CONS: cons file for all cases
        TCOMP: tcomp file for all acases
        Titles: name of each case
        save_path: path to save plot
        plot_traj_index: indexes to plot
        subplots: number of subplots 
        hybrid: true or false whether the case is included
    """
    # Setup plot
    label_size = 18
    linewidth = 2
    fig = plt.figure(figsize = (10,15))
    gs1 = matplotlib.gridspec.GridSpec(5, subplots, 
                                    left=0.15, wspace=0.5, hspace = 0.5, right = 0.93,
                                    top = 0.92, bottom = 0.1)
    
    # Plot trajectories 2D
    nbodies = len(STATES[0][0,:, -1])
    nparticles = np.count_nonzero(STATES[0][0, :, -1])
    nstars = nbodies - nparticles +1 # Include the central star
    name_bodies = (np.arange(nstars)+1).astype(str)

    Energy_error, T_comp, R, action, action_H = calculate_errors(STATES, CONS, TCOMP, steps = None)

    legend = True
    if plot_traj_index == 'bestworst':
        plot_traj_index = [0, len(STATES)-1] # plot best and worst
    elif plot_traj_index == 'RLbest':
        last_E_error = Energy_error[0][-1, 1:]
        best_index = np.where(last_E_error == min(last_E_error))[0][0]
        plot_traj_index = [0, best_index+1]
    elif plot_traj_index == 'HRLbest':
        last_E_error = Energy_error[0][-1, 2:]
        best_index = np.where(last_E_error == min(last_E_error))[0][0]
        plot_traj_index = [1, best_index+2]

    if subplots == 3:
        plot_traj_index= np.arange(3)

    for case_i, case in enumerate([plot_traj_index[0]]): 
        ax1 = fig.add_subplot(gs1[0, 0])
        ax12 = fig.add_subplot(gs1[0, 1])
        plot_planets_trajectory(ax1, STATES[case][:, 0:nstars,:], name_bodies, \
                            labelsize=label_size, steps = env.settings['Integration']['max_steps'], 
                            legend_on = False, axislabel_on = False)
        plot_hydro_system_trajectory(ax12, STATES[case], name_bodies, \
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
            ax1.legend(loc='upper center', bbox_to_anchor=(1.2, 1.7), \
                       fancybox = True, ncol = 4, fontsize = label_size-2)

    # Plot energy error
    linestyle = ['--', '--', '-', '-', '-', '-', '-', '-', '-']
    x_axis = np.arange(1, len(T_comp), 1) * 100 # Multiplied by check step to have it in years
    ax2 = fig.add_subplot(gs1[1, :])
    ax3 = fig.add_subplot(gs1[2, :])
    ax4 = fig.add_subplot(gs1[3, :])
    ax6 = fig.add_subplot(gs1[4, :])

    plot_distance_to_one(ax2, x_axis, STATES[plot_traj_index[0]][1:], legend = False) # plot for the most accurate case

    # Actions
    plot_actions_taken(ax3, x_axis, action[1:, 0], label = Titles[0]) # only for RL
    if hybrid == True:
        plot_actions_taken(ax3, x_axis, action[1:, 1], action_H = action_H[1:, 1],\
                       color = 'darkred', marker = '.', label = Titles[1]) # only for H-RL
        ax3.legend(fontsize = label_size-4, ncol = 2, loc ='upper right')

    n_actions = env.settings['RL']['number_actions']
    ax3.set_ylim([-1, n_actions+1])
    labels = np.arange(n_actions)[::3]
    ax3.set_yticks(np.arange(n_actions)[::3])
    LABELS = []
    for label in labels:
        LABELS.append('%i: %i'%(label, env.actions[label]))
    ax3.set_yticklabels(LABELS)

    for case in range(len(STATES)):
        if case == 0:
            linestyle = '--'
            alpha = 1
        else:
            linestyle = '-'
            alpha = 0.5

        index_0 = np.where(Energy_error[0][1:, case] ==0)[0]
        x2 = np.delete(x_axis, index_0)
        y2 = np.delete(Energy_error[0][1:, case], index_0)
        plot_evolution(ax4, x2, y2, label = Titles[case], \
                       colorindex = case, linestyle = linestyle, linewidth = linewidth, alpha = alpha)
        plot_evolution(ax6, x_axis, T_comp[1:, case], label = Titles[case], \
                       colorindex = case, linestyle = linestyle, linewidth = linewidth, alpha = alpha)
    
    for ax in [ax2, ax4]:
        ax.set_yscale('log')
    for ax_i in [ax2, ax3, ax4, ax6]:
        ax_i.tick_params(axis='both', which='major', labelsize=label_size-2)

    ax6.set_xlabel('Time (yr)', fontsize = label_size)
    ax4.set_ylim((1e-5, 1e0))

    ax2.set_ylabel(r'$\vert \vec r_x -\vec r_S\vert$ (au) ', fontsize = label_size)
    ax3.set_ylabel(r'Action taken', fontsize = label_size)
    ax4.set_ylabel(r'$\Delta E_{Total}$', fontsize = label_size)
    ax6.set_ylabel(r'$T_{Comp}$ (s)', fontsize = label_size)

    ax6.legend(loc='upper center', bbox_to_anchor=(0.45, -0.42), \
                       fancybox = True, ncol = 3, fontsize = label_size-2)
    plt.savefig(save_path, dpi = 150)
    plt.show()

def plot_intializations(env, STATES, CONS, TCOMP, Titles, save_path):
    """
    plot_initializations: plot trajectory for different initial seeds
    INPUTS: 
        env: environment
        STATES: states file for all cases
        CONS: cons file for all cases
        TCOMP: tcomp file for all acases
        Titles: name of each case
        save_path: path to save plot
    """
    # Setup plot
    label_size = 18
    fig = plt.figure(figsize = (17,10))
    rows = 4
    columns = 2
    gs1 = matplotlib.gridspec.GridSpec(columns,rows, 
                                    left=0.1, wspace=0.5, hspace = 0.25, right = 0.96,
                                    top = 0.84, bottom = 0.06)
    # Plot trajectories 2D
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
        if case_i == 1:
            ax1.legend(loc='upper center', bbox_to_anchor=(1., 1.47), \
                       fancybox = True, ncol = 4, fontsize = label_size-2)
            
        ax1.tick_params(axis='both', which='major', labelsize=label_size-3)
        ax12.tick_params(axis='both', which='major', labelsize=label_size-3)
        
    plt.savefig(save_path, dpi = 150)
    # plt.show()
        
def plot_rewards(env, STATES, CONS, TCOMP, Titles, save_path):
    """
    plot_rewards: plot rewards for different cases
    INPUTS: 
        env: environment
        STATES: states file for all cases
        CONS: cons file for all cases
        TCOMP: tcomp file for all acases
        Titles: name of each case
        save_path: path to save plot
    """
    # Setup plot
    label_size = 18
    fig = plt.figure(figsize = (10,15))
    gs1 = matplotlib.gridspec.GridSpec(4, 2, 
                                    left=0.08, wspace=0.3, hspace = 0.3, right = 0.93,
                                    top = 0.9, bottom = 0.07)
    
    # Plot energy error
    linestyle = ['-', '--', '-.', '-', '-', '-', '-', '-', '-']
    Energy_error, Energy_error_local, T_comp, R, action, action_H= calculate_errors(STATES, CONS, TCOMP)
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
    """
    calculate_rewards: calculate reward value to plot
    INPUTS: 
        E_E: energy error
        E_E_local: energy error at a step
        T_comp: computation time
        action: action taken
        type_reward: reward function
    OUTPUTS:
        a: reward values
    """
    len_array = len(E_E[2:])

    if type_reward == 0:
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
    """
    plot_reward_comparison: plot_reward comparison
    INPUTS: 
        env: environment
        STATES: states file for all cases
        CONS: cons file for all cases
        TCOMP: tcomp file for all acases
        Titles: name of each case
        save_path: path to save plot
    """
    # Setup plot
    label_size = 18
    fig = plt.figure(figsize = (10,6))

    gs1 = matplotlib.gridspec.GridSpec(1, 1, 
                                    left=0.08, wspace=0.3, hspace = 0.3, right = 0.93,
                                    top = 0.75, bottom = 0.09)
    
    # Plot energy error
    linestyle = ['-', '--', '-.', '-', '-', '-', '-', '-', '-']
    [Energy_error, Energy_error_local], T_comp, R, action, action_H = calculate_errors(STATES, CONS, TCOMP)
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
    """
    plot_comparison_end: 
    INPUTS: 
        env: environment
        STATES: states file for all cases
        CONS: cons file for all cases
        TCOMP: tcomp file for all acases
        Titles: name of each case
        save_path: path to save plot
        plot_traj_index: indexes to plot cartesian
    """
    # Setup plot
    label_size = 18
    fig = plt.figure(figsize = (10,15))
    gs1 = matplotlib.gridspec.GridSpec(4, 2, 
                                    left=0.08, wspace=0.3, hspace = 0.3, right = 0.93,
                                    top = 0.9, bottom = 0.07)
    
    
    # Plot trajectories 2D
    name_bodies = name_bodies = [r"$S_1$", r"$S_2$", r"$S_3$", r"$S_4$", r"$S_5$", r"$P_1$", r"$P_2$"]
    legend = True

    # Plot energy error
    linestyle = ['--', '--', '-', '-', '-', '-', '-', '-', '-']
    Energy_error, Energy_error_local, T_comp, R, action, action_H = calculate_errors(STATES, CONS, TCOMP)

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
    """
    plot_distance_action: plot closest approach and action associated 
    INPUTS: 
        env: environment
        STATES: states file for all cases
        CONS: cons file for all cases
        TCOMP: tcomp file for all acases
        Titles: name of each case
        save_path: path to save plot
    """
    # Setup plot
    label_size = 18
    fig = plt.figure(figsize = (10,15))
    gs1 = matplotlib.gridspec.GridSpec(3, 1, 
                                    left=0.08, wspace=0.3, hspace = 0.3, right = 0.93,
                                    top = 0.9, bottom = 0.07)
    
    # Plot energy error
    linestyle = ['--', '--', '-', '-', '-', '-', '-', '-', '-']
    Energy_error, T_comp, R, action, action_H = calculate_errors(STATES, CONS, TCOMP)
    x_axis = np.arange(0, len(T_comp), 1)
    ax2 = fig.add_subplot(gs1[0, :])
    ax3 = fig.add_subplot(gs1[1, :])
    ax4 = fig.add_subplot(gs1[2, :])
    plot_distance_to_one(ax2, x_axis, STATES[0] ) # plot for RL one
    plot_actions_taken(ax3, x_axis, action[:, 0])
    for case in range(len(STATES)):
        plot_evolution(ax4, x_axis[1:], Energy_error[3][1:, case], label = Titles[case][1:], \
                       colorindex = case, linestyle = linestyle[case])
    
    for ax in [ax2, ax4]:
        ax.set_yscale('log')

    ax4.set_xlabel('Step', fontsize = label_size)
    ax4.set_ylabel('Energy Error', fontsize = label_size)
    ax2.legend(fontsize = label_size -3)
    ax4.legend(fontsize = label_size -3)

    plt.savefig(save_path, dpi = 150)
    plt.show()

def plot_state_diff(env, STATES, CONS, TCOMP, Titles, save_path):
    """
    plot_state_diff: plot difference in states
    INPUTS: 
        env: environment
        STATES: states file for all cases
        CONS: cons file for all cases
        TCOMP: tcomp file for all acases
        Titles: name of each case
        save_path: path to save plot
    """
    # Setup plot
    label_size = 18
    fig = plt.figure(figsize = (10,15))
    gs1 = matplotlib.gridspec.GridSpec(5, 1, 
                                    left=0.08, wspace=0.3, hspace = 0.3, right = 0.93,
                                    top = 0.9, bottom = 0.07)
    
    # Plot energy error
    linestyle = ['--', '--', '-', '-', '-', '-', '-', '-', '-']
    Energy_error, T_comp, R, action, action_H = calculate_errors(STATES, CONS, TCOMP)
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
    plot_energy_vs_tcomp: plot energy against computation time
    INPUTS: 
        env: environment
        STATES: states file for all cases
        CONS: cons file for all cases
        TCOMP: tcomp file for all acases
        Titles: name of each case
        initializations: number of initializations for each case
        save_path: path to save plot
        plot_traj_index: index to be plotted
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

    markers = ['o', 'x', 's', 'o', 'x', '^', 'o']
    order = [1,2,0, 3, 4, 5, 6, 7]
    alpha = [0.5, 0.5, 0.9, 0.8, 0.9, 0.7, 0.7]

    # Calculate the energy errors
    E_T = np.zeros((initializations, len(plot_traj_index)))
    T_c = np.zeros((initializations, len(plot_traj_index)))
    nsteps_perepisode = np.zeros((initializations, len(plot_traj_index)))

    for act in range(len(plot_traj_index)):
        for i in range(initializations):
            nsteps_perepisode = len(cons[act*initializations +i][:,0])
            E_T[i, act] = abs(cons[act*initializations + i][3, -1]) # absolute relative energy error
            T_c[i, act] = np.sum(tcomp[act*initializations + i])/nsteps_perepisode # add individual computation times

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
        Y = E_T[:, i]
        ax1.scatter(X, Y, color = colors[i], alpha = alphavalue, marker = markers[i],\
                s = msize, label = Titles[i], zorder =order[i])
        min_x = minimum(min_x, min(X))
        min_y = minimum(min_y, min(Y))
        max_x = maximum(max_x, max(X))
        max_y = maximum(max_y, max(Y))
    binsx = np.logspace(np.log10(min_x),np.log10(max_x), 50)
    binsy = np.logspace(np.log10(min_y),np.log10(max_y), 50)  

    for i in range(len(plot_traj_index)):  
        X = T_c[:, i]
        Y = E_T[:, i]
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
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='both', which='major', labelsize=labelsize)
    
    plt.savefig(save_path+'tcomp_vs_Eerror_evaluate.png', dpi = 100)
    # plt.show()

def plot_energy_vs_tcomp_avg(env, STATES, cons, tcomp, Titles, initializations, save_path, plot_traj_index = [0,1,2]):
    """
    plot_energy_vs_tcomp_avg: plot energy against computation time avg
    INPUTS: 
        env: environment
        STATES: states file for all cases
        CONS: cons file for all cases
        TCOMP: tcomp file for all acases
        Titles: name of each case
        initializations: number of initializations for each case
        save_path: path to save plot
        plot_traj_index: index to be plotted
    """
    cases = len(STATES)

    fig = plt.figure(figsize = (6,6))
    gs1 = matplotlib.gridspec.GridSpec(1, 1, figure = fig, 
                                    #    width_ratios = (3, 1), height_ratios = (1, 3), \
                                       left=0.15, wspace=0.3, 
                                       hspace = 0.2, right = 0.99,
                                        top = 0.97, bottom = 0.11)
    msize = 50
    alphavalue = 0.5
    alphavalue2 = 0.9
    
    ax1 = fig.add_subplot(gs1[0, 0]) 

    markers = ['o', 'x', 's', 'o', 'x', '^', 'o']
    order = [1,2,0, 3, 4, 5, 6, 7]
    alpha = [0.5, 0.5, 0.9, 0.8, 0.9, 0.7, 0.7]

    # Calculate the energy errors
    E_T = np.zeros((initializations, len(plot_traj_index)))
    T_c = np.zeros((initializations, len(plot_traj_index)))
    nsteps_perepisode = np.zeros((initializations, len(plot_traj_index)))

    for act in range(len(plot_traj_index)):
        for i in range(initializations):
            nsteps_perepisode = len(cons[act*initializations +i][:,0])
            E_T[i, act] = abs(cons[act*initializations + i][3, -1]) # absolute relative energy error
            T_c[i, act] = np.sum(tcomp[act*initializations + i])/nsteps_perepisode # add individual computation times

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
        Y = E_T[:, i]
        Y = np.log10(Y)
        ax1.errorbar(np.mean(X), np.mean(Y), 
                     xerr = np.std(X), yerr = np.std(Y), 
                    fmt='-o',
                label = Titles[i], zorder =order[i])
        min_x = minimum(min_x, min(X))
        min_y = minimum(min_y, min(Y))
        max_x = maximum(max_x, max(X))
        max_y = maximum(max_y, max(Y))

    labelsize = 12
    ax1.legend(fontsize = labelsize)
    ax1.set_xlabel('Total computation time (s)',  fontsize = labelsize)
    ax1.set_ylabel(r'log$_{10}(\vert\Delta E\vert$) final',  fontsize = labelsize)
    for ax in [ax1]:
        ax.tick_params(axis='both', which='major', labelsize=labelsize)
    
    plt.savefig(save_path+'tcomp_vs_Eerror_evaluate2.png', dpi = 100)
    # plt.show()

def plot_energy_vs_tcomp_avg_together(env, STATES_list, cons_list, tcomp_list, Titles_list, \
                                      initializations, save_path, plot_traj_index = [0,1,2], \
                                        hybrid = False):
    """
    plot_energy_vs_tcomp_avg_together: plot energy against computation time for a list of cases
    INPUTS: 
        env: environment
        STATES_list: states file for all cases
        CONS_list: cons file for all cases
        TCOMP_list: tcomp file for all acases
        Titles_list: name of each case
        initializations: number of initializations for each case
        save_path: path to save plot
        plot_traj_index: index to be plotted
        hybrid: include hybrid case or not
    """
    cases = len(STATES_list[0])

    fig = plt.figure(figsize = (18,7))
    gs1 = matplotlib.gridspec.GridSpec(1, 3, figure = fig, 
                                    #    width_ratios = (3, 1), height_ratios = (1, 3), \
                                       left=0.07, wspace=0.4, 
                                       hspace = 0.2, right = 0.99,
                                        top = 0.95, bottom = 0.2)
    msize = 50
    alphavalue = 0.5
    alphavalue2 = 0.9

    markers = ['o', 'x', 's', 'o', 'x', '^', 'o']
    order = [1,2,0, 3, 4, 5, 6, 7]
    alpha = [0.5, 0.5, 0.9, 0.8, 0.9, 0.7, 0.7]

    if hybrid == False:
        colors2 = [colors[0]] + colors[2:]
    else:
        colors2 = colors

    plot_title = ['$N = 5$', '$N = 9$', '$N = 15$']

    for plot_i in range(3):
        ax1 = fig.add_subplot(gs1[0, plot_i]) 
        cons = cons_list[plot_i]
        tcomp = tcomp_list[plot_i]
        Titles = Titles_list[plot_i]
        ax1.set_title(plot_title[plot_i], fontsize = 20)

        # Calculate the energy errors
        E_T = np.zeros((initializations, len(Titles)))
        T_c = np.zeros((initializations, len(Titles)))
        nsteps_perepisode = np.zeros((initializations, len(Titles)))

        index_eliminate = np.zeros((len(Titles), initializations))
        for act in range(len(Titles)):
            for i in range(initializations):
                nsteps_perepisode = len(cons[act*initializations +i][:,0])
                E_T[i, act] = abs(cons[act*initializations + i][-1, 3]) # absolute relative energy error
                T_c[i, act] = np.sum(tcomp[act*initializations + i]) # add individual computation times
            
                index_eliminate[act, i] = eliminate_escaped_planets(STATES_list[plot_i][act*initializations+i], nsteps_perepisode)

        join_dots = np.zeros((len(Titles), 2))
        for i in range(len(Titles)):  
            X = T_c[:, i]
            Y = E_T[:, i]
            Y = np.log10(Y)
            if i == 0: 
                msize = 15
                marker = '^'
            elif i == 1 and hybrid == True:
                msize = 15
                marker = '^'
            else: 
                msize = 12
                marker = 'o'

            index_remove = np.where(index_eliminate[i, :] ==1)[0]
            X2 = np.delete(X, index_remove)
            Y2 = np.delete(Y, index_remove)

            ax1.errorbar(np.mean(X2), np.mean(Y2), 
                        xerr = np.std(X2), yerr = np.std(Y2), 
                        # fmt='-o',
                        color = colors2[i],marker = marker,\
                         alpha = 1, capsize = 5,
                    markersize = msize, 
                    label = Titles[i],
                    mec = 'k', ecolor = 'k',
                    zorder =order[i])
            
            join_dots[i, 0] = np.mean(X2)
            join_dots[i, 1] = np.mean(Y2)
            
            # plot all unfilled
            ax1.scatter(np.mean(X2)*np.ones(len(Y)), Y, 
                        edgecolors = colors2[i], 
                        marker = marker,\
                    s = msize+30, 
                    alpha = 1,
                    zorder =order[i], facecolors = 'none')
            
            # plot unescaped filled
            ax1.scatter(np.mean(X2)*np.ones(len(Y2)), Y2, 
                        color = colors2[i], 
                        marker = marker,\
                    s = msize+30, 
                    alpha = 1,
                    zorder =order[i])

            labelsize = 20
            if plot_i == 0:
                ax1.legend(bbox_to_anchor=(1.8, -0.28), fontsize = labelsize,\
                           loc='lower center', ncol = 6)

            ax1.set_xlabel(r'$T_{comp}$ (s)',  fontsize = labelsize)
            ax1.set_ylabel(r'log$_{10}(\vert\Delta E_{Total}\vert$) ',  fontsize = labelsize)
            for ax in [ax1]:
                ax.tick_params(axis='both', which='major', labelsize=labelsize)
        
        if hybrid == False:
            plt.plot(join_dots[1:, 0], join_dots[1:, 1], color = 'black')
        else:
            plt.plot(join_dots[2:, 0], join_dots[2:, 1], color = 'black')

    plt.savefig(save_path+'tcomp_vs_Eerror_evaluate3.png', dpi = 100)
    plt.show()