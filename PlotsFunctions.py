"""
PlotsFunctions: plotting functions

Author: Veronica Saz Ulibarrena
Last modified: 5-March-2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch

from amuse.units import units, constants
from amuse.ext.orbital_elements import get_orbital_elements_from_arrays, get_orbital_elements_from_binary

from TrainRL import DQN

colors = ['steelblue', 'darkgoldenrod', 'seagreen', 'coral',  \
        'mediumslateblue', 'deepskyblue', 'blue', 'black', 'red']
colors2 = ['navy']
lines = ['-', '--', ':', '-.' ]
markers = ['o', 'o','o','o','o','o','o','o','o','x', 'x','x', '^', 's']

def plot_planets_trajectory(ax, state, name_planets, labelsize = 15, steps = 30, \
                            legend_on = True, axislabel_on = True, marker = 'o', axis = 'xy'):
    """
    plot_planets_trajectory: plot trajectory of three bodies
    INPUTS:
        ax: matplotlib ax to be plotted in 
        state: array with the state of each of the particles at every step
        name_planets: array with the names of the bodies
        labelsize: size of matplotlib labels
        steps: steps to be plotted
        legend_on: True or False to display the legend
    """
    if axis == 'xy': 
        indexes = [2, 3]
    elif axis == 'xz':
        indexes = [2, 4]

    n_bodies = np.shape(state)[1]
    n_planets = np.count_nonzero(state[0, :, -1])
    markers = ['o']*(n_bodies-n_planets+1) + ['x'] * (n_planets-1)
    for j in range(n_bodies):
        x = state[0:steps, j, indexes[0]]/1.496e11
        y = state[0:steps, j, indexes[1]]/1.496e11
        m = state[0, j, 1]
        size_marker = np.log10(m)/10

        ax.scatter(x[0], y[0], marker = markers[j%len(markers)], s = 20*size_marker,\
                   c = colors[j%len(colors)], \
                    label = "Particle "+ name_planets[j])
        
        ax.plot(x[1:], y[1:], marker = None, 
                    markersize = size_marker, \
                    linestyle = '-',\
                    color = colors[j%len(colors)], \
                    alpha = 0.1)
        
        ax.scatter(x[1:], y[1:], marker = markers[j%len(markers)], s = size_marker, \
                    c = colors[j%len(colors)])        
        
    # if legend_on == True:
    #     ax.legend(fontsize = labelsize)
    if axislabel_on == True:
        ax.set_xlabel('x (au)', fontsize = labelsize)
        ax.set_ylabel('y (au)', fontsize = labelsize)

def plot_planetary_system_trajectory(ax, state, name_planets, labelsize = 15, steps = 30, \
                            legend_on = True, axislabel_on = True, marker = 'o', axis = 'xy'):
    """
    plot_planetary_system_trajectory: plot trajectory of the planetary system
    INPUTS:
        ax: matplotlib ax to be plotted in 
        state: array with the state of each of the particles at every step
        name_planets: array with the names of the bodies
        labelsize: size of matplotlib labels
        steps: steps to be plotted
        legend_on: True or False to display the legend
    """
    if axis == 'xy': 
        indexes = [2, 3]
    elif axis == 'xz':
        indexes = [2, 4]

    # Calculate center of gravity movement to remove it
    n_bodies = np.shape(state)[1]
    n_planets = np.count_nonzero(state[0, :, -1])
    X = np.zeros((steps, n_planets))
    Y = np.zeros((steps, n_planets))
    M = np.zeros(n_planets)
    counter = 0

    for j in range(n_bodies):
        if state[0, j, -1] == 1: # planet type
            if counter == 0:
                subtract_X = state[0:steps, j, indexes[0]]/1.496e11
                subtract_Y = state[0:steps, j, indexes[1]]/1.496e11
            X[:, counter] = state[0:steps, j, indexes[0]]/1.496e11 - subtract_X
            Y[:, counter] = state[0:steps, j, indexes[1]]/1.496e11 - subtract_Y
            M[counter] = state[0, j, 1]
            counter += 1 # count planet index

    index_plot_colors = n_bodies-n_planets
    size_marker = 10
    markers = ['o'] + ['x'] * (n_planets-1)
    for j in range(n_planets):
        ax.scatter(X[0, j], Y[0, j], marker = markers[(index_plot_colors+j)%len(markers)],\
                   s = 1*size_marker,\
                    c = colors[(index_plot_colors+j)%len(colors)], \
                    label = "Particle "+ name_planets[j])
        ax.plot(X[:, j], Y[:, j], marker = markers[(index_plot_colors+j)%len(markers)], 
                    markersize = size_marker, \
                    linestyle = '-',\
                    color = colors[(index_plot_colors+j)%len(colors)], \
                    alpha = 0.1)
        ax.scatter(X[:, j], Y[:, j], marker = markers[(index_plot_colors+j)%len(markers)], s = 3*size_marker, \
                    c = colors[(index_plot_colors+j)%len(colors)])        
        
    if legend_on == True:
        ax.legend(fontsize = labelsize)
    if axislabel_on == True:
        ax.set_xlabel('x (au)', fontsize = labelsize)
        ax.set_ylabel('y (au)', fontsize = labelsize)
    
def plot_hydro_system_trajectory(ax, state, name_planets, labelsize = 15, steps = 30, \
                            legend_on = True, axislabel_on = True, marker = 'o', axis = 'xy'):
    """
    plot_hydro_system_trajectory: plot trajectory of disk
    INPUTS:
        ax: matplotlib ax to be plotted in 
        state: array with the state of each of the particles at every step
        name_planets: array with the names of the bodies
        labelsize: size of matplotlib labels
        steps: steps to be plotted
        legend_on: True or False to display the legend
    """
    if axis == 'xy': 
        indexes = [2, 3]
    elif axis == 'xz':
        indexes = [2, 4]

    # Calculate center of gravity movement to remove it
    n_bodies = np.shape(state)[1]
    n_planets = np.count_nonzero(state[0, :, -1])
    X = np.zeros((steps, n_planets))
    Y = np.zeros((steps, n_planets))
    M = np.zeros(n_planets)
    counter = 0

    for j in range(n_bodies):
        if state[0, j, -1] == 1: # planet type
            if counter == 0:
                subtract_X = state[0:steps, j, indexes[0]]/1.496e11
                subtract_Y = state[0:steps, j, indexes[1]]/1.496e11
            X[:, counter] = state[0:steps, j, indexes[0]]/1.496e11 - subtract_X
            Y[:, counter] = state[0:steps, j, indexes[1]]/1.496e11 - subtract_Y
            M[counter] = state[0, j, 1]
            counter += 1 # count planet index

    index_plot_colors = n_bodies-n_planets
    size_marker = 10
    markers = ['o'] + ['.'] * (n_planets-1)
    for j in range(n_planets):
        if j == 1:
            label = 'disk'
            size_mag = 1
        elif j == 0:
            label = 'star'
            size_mag = 4
        else:
            label = None
        ax.scatter(X[0, j], Y[0, j], marker = markers[j],\
                   s = size_mag*size_marker,\
                    c = colors[(index_plot_colors+j)%len(colors)], \
                    label = label)
        ax.plot(X[:15, j], Y[:15, j], marker = markers[j], 
                    markersize = size_marker, \
                    linestyle = '-',\
                    color = colors[(index_plot_colors+j)%len(colors)], \
                    alpha = 0.1)
        # ax.scatter(X[:, j], Y[:, j], marker = markers[(index_plot_colors+j)%len(markers)], s = 3*size_marker, \
                    # c = colors[(index_plot_colors+j)%len(colors)])        
        
    # if legend_on == True:
    ax.legend(fontsize = labelsize)
    if axislabel_on == True:
        ax.set_xlabel('x (au)', fontsize = labelsize)
        ax.set_ylabel('y (au)', fontsize = labelsize)

    # plt.show()

def plot_planets_distance(ax, x_axis, state, name_planets, labelsize = 12, 
                          steps = 30, legend = True):
    """
    plot_planets_distance: plot steps vs pairwise-distance of the bodies
    INPUTS:
        ax: matplotlib ax to be plotted in 
        x_axis: time or steps to be plotted in the x axis
        state: array with the state of each of the particles at every step
        name_planets: array with the names of the bodies
        labelsize: size of matplotlib labels
        steps: steps to be plotted
    """
    n_planets = np.shape(state)[1]
    Dist = []
    Labels = []
    for i in range(n_planets):
        r1 = state[0:steps, i, 2:5]/1.496e11
        m = state[0, i, 1]
        for j in range(i+1, n_planets):
            r2 = state[0:steps, j, 2:5]/1.496e11
            Dist.append(np.linalg.norm(r2-r1, axis = 1))
            Labels.append('Particle %i-%i'%(i, j))
        
        size_marker = np.log(m)/30
    for i in range(len(Dist)):
        ax.plot(x_axis, Dist[i], label = Labels[i], linewidth = 2.5)
    if legend == True:
        ax.legend(fontsize =labelsize, framealpha = 0.5)
    ax.set_yscale('log')
    return Dist

def eliminate_escaped_planets(state, steps):
    """
    eliminate_escaped_planets: find unbound planets
    INPUTS:
        state: state of the system
        steps: max number of steps
    OUTPUTS
        index_escaped: index of escaped bodies
    """
    indexes = [2, 3, 4, 5, 6, 7] # for x, y z, vx, vy, vz

    # Calculate center of gravity movement to remove it
    n_bodies = np.shape(state)[1]
    n_planets = np.count_nonzero(state[0, :, -1]) # not count star
    counter = 0
    M = np.zeros(n_planets)
    BODIES = np.zeros((steps, n_planets, 12)) # cartesian + keplerian

    subtract_central = np.zeros((steps, 6))
    for j in range(n_bodies):
        if state[0, j, -1] == 1: # planet type
            if counter == 0: # for the central star
                for z in range(6):
                    subtract_central[:, z] = state[0:steps, j, indexes[z]]
            for z in range(6):
                BODIES[:, counter, z] = state[0:steps, j, indexes[z]] - subtract_central[:, z]
            M[counter] = state[0, j, 1]
            counter += 1 # count planet index
    
    for j in range(1, n_planets):
        i = -1 # only at the last step required 
        orbital = get_orbital_elements_from_arrays(BODIES[i, j, 0:3] | units.m, \
                                                    BODIES[i, j, 3:6] | units.m/ units.s, \
                                                    M[0] | units.kg, 
                                                    constants.G)
        
        BODIES[i, j, 6] = orbital[0].value_in(units.au)
        BODIES[i, j, 7] = orbital[1]
        BODIES[i, j, 8] = orbital[2].value_in(units.rad)
        BODIES[i, j, 9] = orbital[3].value_in(units.rad)
        BODIES[i, j, 10] = orbital[4].value_in(units.rad)
        BODIES[i, j, 11] = orbital[5].value_in(units.rad)

    # Evaluate evolution in time
    counter = 0
    index_escaped = 0
    for j in range(n_bodies):
        if state[0, j, -1] == 1: # planet type
            if counter >0:
                # with distance to star
                distance_to_star = np.linalg.norm(BODIES[:, counter, 0:3], axis = 1)
                if max(BODIES[:, counter, 7]) > 0.99: # large eccentricity
                    index_escaped = 1
            counter += 1

    return index_escaped

def calculate_lim_distance_to_one(state, steps):
    """
    calculate_lim_distance_to_one: calculate max distance to center body
    INPUTS:
        state: state of the system
        steps: max number of steps
    OUTPUTS
        index_escaped: index of escaped bodies
    """
    Dist = []
    index_escaped = 0
    
    star_index = np.where(state[0, :, 8] == 0)[0]
    planet_index = np.where(state[0, :, 8] == 1)[0][0]
    r0 = state[0:steps, planet_index, 2:5]/1.496e11
    for i in range(len(star_index)): # for ecah particle except for the one with particles
        r1 = state[0:steps, star_index[i], 2:5]/1.496e11
        m = state[0, i, 1]
        Dist.append(np.linalg.norm(r0-r1, axis = 1))
        print(Dist[i])
        if min(Dist[i]) < 1000:
            index_escaped = 1

    return index_escaped

def plot_distance_to_one(ax, x_axis, state, labelsize = 12, 
                         legend = True):
    """
    plot_distance_to_one: plot distance to one body
    INPUTS:
        ax: matplotlib ax to be plotted in 
        x_axis: time or steps to be plotted in the x axis
        state: array with the state of each of the particles at every step
        labelsize: size of matplotlib labels
    """
    steps = len(x_axis)
    Dist = []
    Labels = []
    
    star_index = np.where(state[0, :, 8] == 0)[0]
    planet_index = np.where(state[0, :, 8] == 1)[0][0]
    r0 = state[0:steps, planet_index, 2:5]/1.496e11
    for i in range(len(star_index)): # for ecah particle except for the one with particles
        r1 = state[0:steps, star_index[i], 2:5]/1.496e11
        m = state[0, i, 1]
        Dist.append(np.linalg.norm(r0-r1, axis = 1))
        Labels.append('Particle %i-%i'%(i+1, planet_index+1))
        
        size_marker = np.log(m)/30
    for i in range(len(Dist)):
        ax.plot(x_axis, Dist[i], label = Labels[i], linewidth = 2.5, 
                color = colors[(star_index[i])%len(colors)])
    if legend == True:
        ax.legend(fontsize =labelsize, framealpha = 0.5)

    return Dist

def plot_distance_to_one_tree(ax, x_axis, state, labelsize = 12, 
                         legend = True):
    """
    plot_distance_to_one_tree: plot distance to one in a tree code
    INPUTS:
        ax: matplotlib ax to be plotted in 
        x_axis: time or steps to be plotted in the x axis
        state: array with the state of each of the particles at every step
        name_planets: array with the names of the bodies
        labelsize: size of matplotlib labels
    """
    steps = len(x_axis)
    Dist = []
    Labels = []
    
    last_index = np.where(state[0, :, 8] == 1)[0][0]
    r0 = state[0:steps, last_index, 2:5]/1.496e11
    for i in range(0, last_index): # for ecah particle except for the last one
        r1 = state[0:steps, i, 2:5]/1.496e11
        m = state[0, i, 1]
        Dist.append(np.linalg.norm(r0-r1, axis = 1))
        Labels.append('Particle %i-%i'%(i+1, last_index+1))
        
        size_marker = np.log(m)/30
    
    min_distance = np.ones(steps)*1e16
    for j in range(steps):
        for i in range(0, last_index):
            if Dist[i][j] < min_distance[j]:
                min_distance[j] = Dist[i][j]

    for i in range(len(Dist)):
        ax.plot(x_axis, Dist[i], label = Labels[i], linewidth = 2.5, 
                color = colors[(i)%len(colors)], alpha = 0.3)
    ax.plot(x_axis, min_distance, color = colors2[0])

    if legend == True:
        ax.legend(fontsize =labelsize, framealpha = 0.5)

    return Dist

def plot_diff_state(ax1, ax2,  x_axis, state1, state2, labelsize = 12, 
                         legend = True):
    """
    plot_diff_state: plot difference in state coords
    INPUTS:
        ax: matplotlib ax to be plotted in 
        x_axis: time or steps to be plotted in the x axis
        state1: array with the state of each of the particles at every step
        state2: array with the state of each of the particles at every step
        labelsize: size of matplotlib labels
    """
    steps = len(x_axis)
    Diff_r = []
    Diff_v = []
    Labels = []
    
    particles = np.shape(state1)[1]
    for i in range(particles): # for ecah particle except for the last one
        r0 = state1[0:steps, i, 2:5]
        r1 = state2[0:steps, i, 2:5]
        v0 = state1[0:steps, i, 5:]
        v1 = state2[0:steps, i, 5:]
        Diff_r.append(np.linalg.norm(r0-r1, axis = 1))
        Diff_v.append(np.linalg.norm(v0-v1, axis = 1))
        Labels.append('Particle %i'%(i+1))
        
    for i in range(len(Diff_r)):
        ax1.plot(x_axis, Diff_r[i], label = Labels[i], linewidth = 2.5)
        ax2.plot(x_axis, Diff_v[i], label = Labels[i], linewidth = 2.5)
    if legend == True:
        ax1.legend(fontsize =labelsize, framealpha = 0.5)
    # ax.set_yscale('log')

    return Diff_r, Diff_v

def calculate_planet_elements(state, steps):
    """
    calculate evolution of planets
    INPUTS:
        state: array with the state of each of the particles at every step
    """
    indexes = [2, 3, 4, 5, 6, 7] # for x, y z, vx, vy, vz

    # Calculate center of gravity movement to remove it
    n_bodies = np.shape(state)[1]
    n_planets = np.count_nonzero(state[0, :, -1]) 
    M = np.zeros(n_planets)
    counter = 0
    BODIES = np.zeros((steps, n_planets, 12)) # cartesian + keplerian

    # CARTESIAN
    subtract_central = np.zeros((steps, 6))
    for j in range(n_bodies):
        if state[0, j, -1] == 1: # planet type
            if counter == 0: # for the central star
                for z in range(6):
                    subtract_central[:, z] = state[0:steps, j, indexes[z]]
            for z in range(6):
                BODIES[:, counter, z] = state[0:steps, j, indexes[z]] - subtract_central[:, z]
            M[counter] = state[0, j, 1]
            counter += 1 # count planet index

    # KEPLERIAN
    for j in range(1, n_planets):
        for i in range(steps):

            orbital = get_orbital_elements_from_arrays(BODIES[i, j, 0:3] | units.m, \
                                                       BODIES[i, j, 3:6] | units.m/ units.s, \
                                                        M[0] | units.kg, 
                                                        constants.G)
            BODIES[i, j, 6] = orbital[0].value_in(units.au)
            BODIES[i, j, 7] = orbital[1]
            BODIES[i, j, 8] = orbital[2].value_in(units.rad)
            BODIES[i, j, 9] = orbital[3].value_in(units.rad)
            BODIES[i, j, 10] = orbital[4].value_in(units.rad)
            BODIES[i, j, 11] = orbital[5].value_in(units.rad)

    BODIES[:, :, 0:6] /= 1.496e11 # move state to au, including a 

    return BODIES

def plot_evolution_keplerian(ax, x_axis, y_axis, label = None, color = None, 
                   colorindex = None, linestyle = None, linewidth = 1, alpha = 1):
    """
    plot_evolution_keplerian: plot steps vs another measurement
    INPUTS:
        ax: matplotlib ax to be plotted in 
        x_axis: time or steps to be plotted in the x axis
        y_axis: data for the y axis
        label: label for the legend of each data line
        color: color selection for each line
        colorindex: index of the general color selection
        linestyle: type of line
        linewidth: matplotlib parameter for the width of the line
    """
    if colorindex != None:
        color = colors[(colorindex+2)%len(colors)] # start in the blues
    ax.plot(x_axis, y_axis, color = color, marker = 'o', 
            linestyle = linestyle, label = label, 
            linewidth = linewidth, alpha = alpha)


def plot_actions_taken(ax, x_axis, y_axis, action_H= [0], 
                       label = None, color = None, marker = None):
    """
    plot_actions_taken: plot steps vs actions taken by the RL algorithm
    INPUTS:
        ax: matplotlib ax to be plotted in 
        x_axis: time or steps to be plotted in the x axis
        y_axis: data for the y axis
        action_H: in case of hybrid method, actions
    """
    if color == None:
        colors = colors2[0]
    else:
        colors = color

    if marker == None:
        marker = '.'
    else:
        marker = marker
    ax.plot(x_axis, y_axis, color = colors, linestyle = '-', alpha = 0.5,
            marker = marker, markersize = 8, label = label)
    
    if len(action_H) >1:
        print(action_H)
        index_flag = np.where(action_H != 0)[0]
        print(index_flag)
        x = x_axis[index_flag]

        for note in range(len(x)):
            ax.plot([x[note],x[note]], [y_axis[index_flag[note]], 9.5], 
                # marker = 'x', 
                    linestyle = '--',
                alpha = 0.5, color = colors)
            ax.annotate(int(action_H[index_flag][note]), xy=(x[note], 9.5),
                        textcoords='offset points',
                        ha='center', va='top', fontsize = 14)
    ax.grid(axis='y')

def plot_evolution(ax, x_axis, y_axis, label = None, color = None, 
                   colorindex = None, linestyle = None, linewidth = 1, alpha = 1):
    """
    plot_evolution: plot steps vs another measurement
    INPUTS:
        ax: matplotlib ax to be plotted in 
        x_axis: time or steps to be plotted in the x axis
        y_axis: data for the y axis
        label: label for the legend of each data line
        color: color selection for each line
        colorindex: index of the general color selection
        linestyle: type of line
        linewidth: matplotlib parameter for the width of the line
    """
    if colorindex != None:
        color = colors[(colorindex+2)%len(colors)] # start in the blues
    ax.plot(x_axis, y_axis, color = color, linestyle = linestyle, label = label, 
            linewidth = linewidth, alpha = alpha)
