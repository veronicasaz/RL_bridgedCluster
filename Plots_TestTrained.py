import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

from scipy.signal import savgol_filter

import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math

from scipy.signal import butter, lfilter, freqz

from collections import namedtuple, deque

colors = ['steelblue', 'darkgoldenrod', 'mediumseagreen', 'coral',  \
        'mediumslateblue', 'deepskyblue', 'navy']

def plot_durations(episode_rewards, episode, show_result=False):
    """
    plot_durations: plot training progress
    INPUTS:
        episode_rewards: reward for each episode
        episode: episode number
        show_result: if True, shows plot
    https://www.linkedin.com/advice/0/how-do-you-evaluate-performance-robustness-your-reinforcement:
    cumulative reward, the average reward per episode, the number of steps per episode, or the success rate over time.
    """
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    x = np.arange(len(rewards_t.numpy()))
    plt.scatter(x, rewards_t.numpy())
    plt.yscale('symlog', linthresh = 1e-4)
    if episode %50 == 0:
        plt.savefig('./SympleIntegration_training/reward_progress_%i'%episode)


def plot_reward(a, reward, Eerror, HuberLoss):
    """
    plot_reward: plot training parameters such as reward
    INPUTS:
        a: environment
        reward: rewards
        Eerror: energy error
        HuberLoss: huber loss
    """
    episodes = len(reward)-1 #TODO: why is it saving an empty array at the end?
    x_episodes = np.arange(episodes)

    steps_perepisode = np.zeros(episodes)
    cumul_reward_perepisode = np.zeros(episodes)
    avg_reward_perepisode = np.zeros(episodes)
    avg_energy_perepisode = np.zeros(episodes)
    last_energy_perepisode = np.zeros(episodes)
    cumul_energy_perepisode = np.zeros(episodes)
    reward_flat = list()
    energyerror_flat = list()
    episode_end_list = list()
    huberloss_flat = list()
    episode_flat = list()
    energyerror_flat_total = list()

    for i in range(episodes):
        energyerror_flat = energyerror_flat + Eerror[i][1:]
        # energyerror_flat_total = energyerror_flat + Eerror[i][0][1:]
        reward_flat = reward_flat + reward[i][1:]
        episode_flat = episode_flat + [i]*len(reward[i][1:])

        steps_perepisode[i] = len(reward[i])
        cumul_reward_perepisode[i] = sum(reward[i])
        avg_reward_perepisode[i] = np.mean(reward[i])
        avg_energy_perepisode[i] = np.mean(abs(np.array(Eerror[i])))
        cumul_energy_perepisode[i] = np.log10(abs(sum(Eerror[i])))/steps_perepisode[i]
        try:
            last_energy_perepisode[i] = abs(Eerror[i][-1])
            
        except:
            last_energy_perepisode[i] = 0
        # try:
        #     energyerror_flat = energyerror_flat + Eerror[i][1:]
        # except:
        #     energyerror_flat = energyerror_flat + [0]
        if len(reward[i][1:])>0:
            episode_end_list = episode_end_list + [1] + [0]*(len(reward[i][1:])-1)

    x_all = np.arange(len(reward_flat))
    
    f, ax = plt.subplots(4, 1, figsize = (10,6))
    plt.subplots_adjust(left=0.19, right=0.97, top=0.96, bottom=0.15, hspace = 0.3)
    fontsize = 18

    def filter(x, y):
        xlen = 500
        y2 = np.ones(len(y))
        for i in range(len(x)//xlen):
            y2[xlen*i:xlen*(i+1)] *=  np.quantile(y[xlen*i:xlen*(i+1)], 0.5)
        return y2

    pts = 11
    # ax[0].plot(x_episodes, steps_perepisode, color = colors[0], alpha = 1)
    y = reward_flat
    ax[0].plot(x_all, y, color = colors[0], alpha = 1)
    yy = savgol_filter(np.ravel(y), pts*11, 1)    
    ax[0].plot(x_all, yy, color = 'black')
    ax[0].set_ylabel('Reward', fontsize = fontsize)

    # y = avg_reward_perepisode
    y = cumul_reward_perepisode
    ax[1].plot(x_episodes,y, color = colors[0], alpha = 1)
    yy = savgol_filter(np.ravel(y), pts, 1)
    ax[1].plot(x_episodes, yy, color = 'black')
    ax[1].set_ylabel(r'Cumulative $R$/episode', fontsize = fontsize)
    # ax[1].set_yscale('symlog', linthresh = 1e1)

    y = cumul_energy_perepisode
    ax[2].plot(x_episodes, y, color = colors[0], alpha = 1)
    # y = energyerror_flat
    # ax[2].plot(x_all, y, color = colors[0], alpha = 1)
    yy = savgol_filter(np.ravel(y), pts, 1)
    ax[2].plot(x_episodes, yy, color = 'black')
    ax[2].set_ylabel(r'Slope $\Delta E/episode$', fontsize = fontsize)
    ax[2].set_yscale('symlog', linthresh = 1e-1)

    y = steps_perepisode
    ax[3].plot(x_episodes, y, color = colors[0], alpha = 1)
    yy = savgol_filter(np.ravel(y), pts, 1)
    ax[3].plot(x_episodes, yy, color = 'black')
    ax[3].set_ylabel(r'Steps/episode', fontsize = fontsize)

    for ax_i in ax: 
        ax_i.tick_params(axis='x', labelsize=fontsize-5)
        ax_i.tick_params(axis='y', labelsize=fontsize-5)

    ax[-1].set_xlabel('Episode', fontsize = fontsize)
    
    # Plot line
    # line_x = [2100, 2100]
    # line_y1 = [8e1, 1e2]
    # line_y2 = [-1.5e3, 1e2]
    # ax[0].plot(line_x, line_y1, linestyle = '--', color = 'red', alpha = 1)
    # ax[1].plot(line_x, line_y2, linestyle = '--', color = 'red', alpha = 1)

    # ax[0].set_xlim([-1, 3000])
    # ax[1].set_xlim([-1, 3000])
    # ax[0].set_ylim([line_y1[0], line_y1[1]])
    # ax[1].set_ylim([line_y2[0], line_y2[1]])

    plt.savefig(a.settings['Training']['savemodel']+'_cumulative_reward.png', dpi = 100)
    plt.show()


def plot_test_reward(a, test_reward, trainingTime, episodes = None):
    """
    plot_test_reward: plot training parameters taken from the test dataset
    INPUTS:
        a: environment
        test_reward: array with each row for each episode and columns: 
            [Reward, Energy error, Computation time]
    """
    f, ax = plt.subplots(3, 1, figsize = (10,7))
    plt.subplots_adjust(left=0.08, right=0.97, top=0.94, \
                        bottom=0.1, hspace = 0.6)
    fontsize = 18

    def filter(x, y):
        xlen = 500
        y2 = np.ones(len(y))
        for i in range(len(x)//xlen):
            y2[xlen*i:xlen*(i+1)] *=  np.quantile(y[xlen*i:xlen*(i+1)], 0.5)
        return y2
    
    # pts = 11
    # ax[0].plot(x_episodes, steps_perepisode, color = colors[0], alpha = 1)
    if episodes == None:
        episodes = len(test_reward) 
    x_episodes = np.arange(episodes)
    
    REWARD_avg = []
    REWARD_std = []
    EERROR_avg = []
    EERROR_std = []
    EERRORBRIDGE_avg = []
    EERRORBRIDGE_std = []
    TCOMP_avg = []
    TCOMP_std = []
    EERROR_jump_avg = []
    EERROR_jump_std= []
    print(np.shape(test_reward))
    for e in range(episodes):
        reshaped = np.array(test_reward[e]).reshape((-1, 4))
        REWARD_avg.append(np.mean(reshaped[:, 0]))
        REWARD_std.append(np.std(reshaped[:, 0]))

        EERROR_avg.append(np.mean(np.log10(abs(reshaped[:, 1]))))
        EERROR_std.append(np.std(np.log10(abs(reshaped[:, 1]))))

        # EERRORBRIDGE_avg.append(np.mean(np.log10(abs(reshaped[:, 1]))))
        # EERRORBRIDGE_std.append(np.std(np.log10(abs(reshaped[:, 1]))))

        TCOMP_avg.append(np.mean(reshaped[:, 3]))
        TCOMP_std.append(np.std(reshaped[:, 3]))

    y = [REWARD_avg, EERROR_avg, TCOMP_avg]
    e = [REWARD_std, EERROR_std, TCOMP_std]
    for plot in range(len(y)):
        # ax[plot].errorbar(x_episodes, y[plot], e[plot], color = colors[0], \
        #                   alpha = 1, fmt='o')
        y[plot] = np.array(y[plot])
        e[plot] = np.array(e[plot])
        ax[plot].plot(x_episodes, y[plot] + e[plot], color = colors[1], \
                          alpha = 0.2, marker = '.')
        ax[plot].plot(x_episodes, y[plot] - e[plot], color = colors[1], \
                          alpha = 0.2, marker = '.')
        ax[plot].plot(x_episodes, y[plot], color= colors[0], \
                          alpha = 1, marker = '.')

    def maxN(elements, n):
        a = sorted(elements, reverse=True)[:n]
        index = np.zeros(n)
        for i in range(n):
            print(np.where(elements == a[i])[0])
            index[i] = np.where(elements == a[i])[0][0]
        return index, a
    index, value = maxN(y[0], 5) 
    for i in range(len(index)):
        ax[0].plot([index[i], index[i]], \
                   [min(np.array(REWARD_avg)-np.array(REWARD_std)),\
                    value[i]], linestyle = '-', marker = 'x', linewidth = 2, color = 'red')
    
    ax[0].tick_params(axis='both', which='major', labelsize=fontsize-5)
    for ax_i in ax[1:]: 
        ax_i.tick_params(axis='both', which='major', labelsize=fontsize-3)
        # ax_i.tick_params(axis='y', labelsize=fontsize-3)

    f.suptitle('Training time: %.2f min'%(trainingTime[episodes-1]/60), y = 0.99, x = 0.23, fontsize = fontsize -3)

    ax[-1].set_xlabel('Episode', fontsize = fontsize)
    ax[0].set_title('R', fontsize = fontsize)
    ax[1].set_title(r'$log_{10}(\vert \Delta E\vert)$', fontsize = fontsize)
    ax[2].set_title(r'$T_{comp}$ (s)', fontsize = fontsize)
    # ax[2].set_title(r'$log_{10}(\vert \Delta E_{bridge}\vert)$', fontsize = fontsize)
    # ax[2].set_title(r'$log_{10}(\vert \Delta E\vert) - log_{10}(\vert \Delta E_{prev}\vert)$', fontsize = fontsize)
    
    ax[0].set_yscale('symlog', linthresh = 1e0)
    b = ticker.SymmetricalLogLocator(base = 10, linthresh = 1e1)
    b.set_params(numticks = 4)
    ax[0].yaxis.set_major_locator(b)
    
    # ax[3].set_yscale('symlog', linthresh = 1e-1)
    ax[2].set_yscale('log')

    # For 1
    ax[0].set_ylim([-10e6, 10e4])
    ax[1].set_ylim([-4, 4])
    ax[2].set_ylim([1e-1, 3e0])

    # For hermite 2
    # ax[0].set_ylim([-10, 4])
    # ax[1].set_ylim([-10, 0])
    # ax[2].set_ylim([-15, -0.5])
    # ax[3].set_ylim([0.0001, 0.003])


    # For symple 2
    # ax[0].set_ylim([-30, 5])
    # ax[1].set_ylim([-12, 5])
    # ax[2].set_ylim([-30, -0.8])
    # ax[3].set_ylim([0.0001, 0.05])
    
    path = a.settings['Integration']['savefile'] + a.settings['Integration']['subfolder']
    plt.savefig(path + 'test_reward.png', dpi = 100)
    plt.show()


def plot_balance(a, reward, Eerror, Eerror_bridge, tcomp):
    """
    plot_reward: plot training parameters such as reward
    INPUTS:
        a: environment
        reward: rewards
        Eerror: energy error
        HuberLoss: huber loss
    """
    episodes = len(reward)-1
    x_episodes = np.arange(episodes)

    cumul_reward_perepisode = np.zeros(episodes)
    # avg_energy_perepisode = np.zeros(episodes)
    last_energy_perepisode = np.zeros(episodes)
    last_energy_bridge_perepisode = np.zeros(episodes)
    sum_tcomp_perepisode = np.zeros(episodes)
    nsteps_perpeisode = np.zeros(episodes)

    for i in range(episodes):
        nsteps_perpeisode[i] = len(reward[i])
        cumul_reward_perepisode[i] = sum(reward[i])
        sum_tcomp_perepisode[i] = sum(tcomp[i]) / nsteps_perpeisode[i]
        # avg_energy_perepisode[i] = np.mean(abs(np.array(Eerror[i])))
        last_energy_perepisode[i] = abs(Eerror[i][-1])
        last_energy_bridge_perepisode[i] = abs(Eerror_bridge[i][-1])

    
    f, ax = plt.subplots(1, 1, figsize = (10,6))
    plt.subplots_adjust(left=0.19, right=0.97, top=0.96, bottom=0.15, hspace = 0.3)
    fontsize = 18
    
    cm = plt.cm.get_cmap('RdYlBu')
    sc = ax.scatter(sum_tcomp_perepisode, last_energy_bridge_perepisode, c = x_episodes, cmap =cm)
    ax.set_yscale('log')
    plt.colorbar(sc)
    
    ax.set_ylabel(r'Episode $\Delta E_{bridge} $(s)', fontsize = fontsize)
    ax.set_xlabel('Episode avg computation time (s)', fontsize = fontsize)
    
    plt.savefig(a.settings['Training']['savemodel']+'tcomp_vs_energy_training.png', dpi = 100)
    plt.show()
