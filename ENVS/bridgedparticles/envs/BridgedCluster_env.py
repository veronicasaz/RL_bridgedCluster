"""
Bridged2Body_env: environment for integration of a 2 body problem using bridge

Author: Veronica Saz Ulibarrena
Last modified: 8-February-2024

Based on https://www.gymlibrary.dev/content/environment_creation/
pip install -e ENVS needed to create a package
"""

import gym 
import numpy as np
import matplotlib.pyplot as plt
import time
from setuptools import setup
import json

from pyDOE import lhs

from amuse.units import units, constants, nbody_system
from amuse.community.hermite.interface import Hermite
from amuse.community.ph4.interface import ph4
from amuse.community.symple.interface import symple
from amuse.community.huayno.interface import Huayno
from amuse.lab import Particles
from amuse.couple import bridge
from amuse.ext.orbital_elements import  generate_binaries

def plot_state(bodies):
    v = (bodies.vx**2 + bodies.vy**2 + bodies.vz**2).sqrt()
    plt.scatter(bodies.x.value_in(units.au),\
                bodies.y.value_in(units.au), \
                c=v.value_in(units.kms), alpha=0.5)
    plt.colorbar()
    plt.show()


def load_json(filepath):
    """
    load_json: load json file as dictionary
    """
    with open(filepath) as jsonFile:
        data = json.load(jsonFile)
    jsonFile.close()
    return data


class BridgedCluster_env(gym.Env):
    def __init__(self, render_mode = None):
        self.settings = load_json("./settings_integration_Cluster.json")

        self.n_bodies = self.settings['InitialConditions']['n_bodies']

        self._initialize_RL()


    def _initialize_RL(self):
        # STATE
        if self.settings["RL"]["state"] == 'cart':
            self.size = 4*self.n_bodies
            self.observation_space = gym.spaces.Box(low=np.array([-np.inf]*self.size), \
                                                high=np.array([np.inf]*self.size), \
                                                dtype=np.float64)
        # elif self.settings["RL"]["state"] == 'averaged':
            
        # ACTION
        # From a range of paramters
        if self.settings['Integration']['action'] == 'range':
            low = self.settings['Integration']['t_step_bridge'][0]
            high = self.settings['Integration']['t_step_bridge'][-1]
            n_actions = self.settings['Integration']['number_actions']
            self.actions = np.logspace(np.log10(low), np.log10(high), \
                                       num = n_actions, base = 10,
                                       endpoint = True)
        elif self.settings['Integration']['action'] == 'combinations':
            low = self.settings['Integration']['t_step_bridge'][0]
            high = self.settings['Integration']['t_step_bridge'][-1]
            n_actions = self.settings['Integration']['number_actions']
            actions_1 = np.logspace(np.log10(low), np.log10(high), \
                                       num = n_actions, base = 10,
                                       endpoint = True)
            low = self.settings['Integration']['t_step_integr'][0]
            high = self.settings['Integration']['t_step_integr'][-1]
            n_actions = self.settings['Integration']['number_actions']
            actions_2 = np.logspace(np.log10(low), np.log10(high), \
                                       num = n_actions, base = 10,
                                       endpoint = True)
            comb = np.meshgrid(actions_1, actions_2)
            self.actions = np.zeros((len(actions_1)* len(actions_2), 2))
            self.actions[:, 0] = comb[0].flatten() # bridge, tstep param
            self.actions[:, 1] = comb[1].flatten()

        
        self.action_space = gym.spaces.Discrete(len(self.actions)) 

        # Training parameters
        self.W = self.settings['Training']['weights']

    ## BASIC FUNCTIONS
    def reset(self):
        """
        reset: reset the simulation 
        INPUTS:
            seed: choose the random seed
            steps: simulation steps to be taken
            typereward: type or reward to be applied to the problem
            save_state: save state (True or False)
        OUTPUTS:
            state_RL: state vector to be passed to the RL
            info_prev: information vector of the previous time step (zero vector)
        """

        # Select units
        self.units()

        # TODO: CHECK FROM HERE DEPENDING ON SCRIPT
        # Same time step for the integrators and the bridge
        self.particles_global, self.particles_local = self._initial_conditions_bridge()

        self.grav_global = self._initialize_integrator(self.settings["Integration"]['t_step_global'], self.settings["Integration"]['integrator_global'])
        self.grav_local = self._initialize_integrator(self.settings ["Integration"]['t_step_local'], self.settings["Integration"]['integrator_local'])
        self.grav_global.particles.add_particles(self.particles_global)
        self.grav_local.particles.add_particles(self.particles_local)
            
        # Bridge creation
        self.grav_bridge = bridge.Bridge(use_threading=False)
        self.grav_bridge.add_system(self.grav_global, (self.grav_local,))
        self.grav_bridge.add_system(self.grav_local, (self.grav_global,))
        self.grav_bridge.timestep = self.actions[0] | self.units_time

        self.channel = [self.grav_global.particles.new_channel_to(self.particles_global), \
                        self.grav_local.particles.new_channel_to(self.particles_local)]

        particles_joined = self._join_particles_bridge([self.particles_global, self.particles_local])
        self.n_bodies_total = len(particles_joined)

        # Get initial energy and angular momentum. Initialize at 0 for relative error
        self.E_0, self.L_0 = self._get_info(particles_joined, initial = True)

        # Create state vector
        state_RL = self._get_state(particles_joined, self.E_0) # |r_i|, |v_i|
        self.iteration = 0
        self.t_cumul = 0.0 # cumulative time for integration
        self.t0_comp = time.time()

        # Plot trajectory
        if self.settings['Integration']['plot'] == True:
            plot_state(particles_joined)

        # Initialize variables to save simulation information
        if self.settings['Integration']['savestate'] == True:
            if steps == None:
                steps = self.settings['Integration']['max_steps']
            self.state = np.zeros((steps, self.n_bodies_total, 8)) # action, mass, rx3, vx3, 
            self.cons = np.zeros((steps, 5)) # action, E, Lx3, 
            self.comp_time = np.zeros(steps) # computation time
            self._savestate(0, 0, particles_joined, 0.0, 0.0, 0.0) # save initial state

        self.info_prev = [0.0, 0.0]
        return state_RL, self.info_prev
    
    def step(self, action):
        """
        step: take a step with a given action, evaluate the results
        INPUTS:
            action: integer corresponding to the action taken
        OUTPUTS:
            state: state of the system to be given to the RL algorithm
            reward: reward value obtained after a step
            terminated: True or False, whether to stop the simulation
            info: additional info to pass to the RL algorithm
        """

        check_step = self.settings['Integration']['check_step'] # final time for step integration
        self.iteration += 1
        self.t_cumul += check_step # add the previous simulation time
        t = (self.t_cumul) | self.units_time

        # Apply action
        self.grav_bridge.timestep = self.actions[action] | self.units_time

        # Integrate
        t0_step = time.time()
        self.grav_bridge.evolve_model(t)
        T = time.time() - t0_step

        for chan in range(len(self.channel)):
            self.channel[chan].copy()

        particles_joined = self._join_particles_bridge([self.particles_global, self.particles_local])
            
        # Get information for the reward
        info_error = self._get_info(particles_joined)
        if self.save_state_to_file == True:
            self._savestate(action, self.iteration, particles_joined, info_error[0], info_error[1], T) # save initial state
        
        # Information to evaluate
        state = self._get_state(particles_joined, info_error[0])
        reward = self._calculate_reward(info_error, self.info_prev, T, self.actions[action], self.W) # Use computation time for this step, including changing integrator
        self.reward = reward
        self.info_prev = info_error

        # finish experiment if max number of iterations is reached
        if (abs(info_error[0]) > 1e-4) or self.iteration == self.settings['Integration']['max_steps']:
            terminated = True
        else:
            terminated = False
            
        # Display information at each step
        if self.settings['Training']['display'] == True:
            self._display_info(info_error, reward, action)

        # Plot trajectory
        if self.settings['Integration']['plot'] == True and terminated == True:
            plot_state(particles_joined)

        info = dict()
        info['TimeLimit.truncated'] = False
        info['Energy_error'] = info_error[0]

        return state, reward, terminated, info
    
    
    ## ADDITIONAL FUNCTIONS NEEDED
    def _join_particles_bridge(self, particles_vector):
        """
        _join_particles_bridge: put all particles into a particle set
        INPUT: 
            particles_vector: array with all the particles sets involved
        OUTPUT:
            bodies: particle set with all bodies
        """
        bodies = Particles()
        for i in range(len(particles_vector)):
            bodies.add_particles(particles_vector[i])
        return bodies
    
    def _initialize_integrator(self, action, integrator_type):
        """
        _initialize_integrator: initialize chosen integrator with the converter and parameters
        INPUTS:
            action: choice of the action for a parameter that depends on the integrator. 
        Options:
            - Hermite: action is the time-step parameter
            - Ph4: action is the time-step parameter
            - Huayno: action is the time-step parameter
            - Symple: action is the time-step size
        OUTPUTS:
            g: integrator
        """
        if integrator_type == 'Hermite': 
            if self.settings['InitialConditions']['units'] == 'si':
                g = Hermite(self.converter)
            else:
                g = Hermite()
            # Collision detection and softening
            # g.stopping_conditions.timeout_detection.enable()
            g.parameters.epsilon_squared = 1e-9 | nbody_system.length**2
        elif integrator_type == 'Ph4': 
            if self.settings['InitialConditions']['units'] == 'si':
                g = ph4(self.converter)
            else:
                g = ph4()
            g.parameters.epsilon_squared = 1e-9 | nbody_system.length**2 # Softening
        elif integrator_type == 'Huayno': 
            if self.settings['InitialConditions']['units'] == 'si':
                g = Huayno(self.converter)
            else:
                g = Huayno()
            g.parameters.epsilon_squared = 1e-9 | nbody_system.length**2 # Softening
        elif integrator_type == 'Symple': 
            if self.settings['InitialConditions']['units'] == 'si':
                g = symple(self.converter, redirection ='none')
            else:
                g = symple(redirection ='none')
            g.initialize_code()
            
        return g 
    
    def _get_info(self, particles, initial = False): # change to include multiple energies
        """
        _get_info: get energy error, angular momentum error at current state
        OUTPUTS:
            Relative energy error, Relative Angular momentum error vector 
        """
        E_kin = particles.kinetic_energy().value_in(self.units_energy)
        E_pot = particles.potential_energy(G = self.G).value_in(self.units_energy)
        L = self.calculate_angular_m(particles)
        if initial == True:
            return E_kin + E_pot, L
        else:
            Delta_E = (E_kin + E_pot - self.E_0) / self.E_0
            Delta_L = (L - self.L_0) / self.L_0
            return Delta_E, Delta_L


    def _get_state(self, particles, E):  # TODO: change to include all particles?
        """
        _get_state: create the state vector
        Options:
            - norm: norm of the positions and velocities of each body and the masses
            - cart: 2D cartesian coordinates of the position and angular momentum plus the energy error
            - dis: distance between particles in position and momentum space plus the energy error

        OUTPUTS: 
            state: state array to be given to the reinforcement learning algorithm
        """
        particles_p_nbody = self.converter.to_generic(particles.position).value_in(nbody_system.length)
        particles_v_nbody = self.converter.to_generic(particles.velocity).value_in(nbody_system.length/nbody_system.time)
        particles_m_nbody = self.converter.to_generic(particles.mass).value_in(nbody_system.mass)

        if self.settings['Integration']['state'] == 'norm':
            state = np.zeros((self.n_bodies)*3) # m, norm r, norm v

            state[0:self.n_bodies] = particles_m_nbody
            state[self.n_bodies: 2*self.n_bodies]  = np.linalg.norm(particles_p_nbody, axis = 1)
            state[2*self.n_bodies: 3*self.n_bodies] = np.linalg.norm(particles_v_nbody, axis = 1)
       
        elif self.settings['Integration']['state'] == 'cart':
            state = np.zeros((self.n_bodies)*4+1) # all r, all v
            for i in range(self.n_bodies):
                state[2*i:2*i+2] = particles_p_nbody[i, 0:2]/10 # convert to 2D. Divide by 10 to same order as v
                state[2*self.n_bodies + 2*i: 2*self.n_bodies + 2*i+2] = particles_v_nbody[i, 0:2]
                state[-1] = -np.log10(abs(E))
        
        elif self.settings['Integration']['state'] == 'dist':
            state = np.zeros((self.n_bodies)*2) # dist r, dist v

            counter = 0
            for i in range(self.n_bodies):
                for j in range(i+1, self.n_bodies):
                    state[counter]  = np.linalg.norm(particles_p_nbody[i,:]-particles_p_nbody[j,:], axis = 0) /10
                    state[self.n_bodies+counter ] = np.linalg.norm(particles_v_nbody[i,:]-particles_v_nbody[j,:], axis = 0)
                    counter += 1

            state[-1] = -np.log10(abs(E))

        return state

    def _display_info(self, info, reward, action):
        """
        _display_info: display information at every step
        INPUTS:
            info: energy error and angular momentum vector
            reward: value of the reward for the given step
            action: action taken at this step
        """
        print("Iteration: %i/%i, E_E = %0.3E, Action: %i, Reward: %.4E"%(self.iteration, \
                                 self.settings['Integration']['max_steps'],\
                                 info[0],\
                                 action, \
                                 reward))
            
    def _savestate(self, action, step, particles, E, L, T):
        """
        _savestate: save state of the system to file
        INPUTS:
            action: action taken
            step: simulation step
            particles: particles set
            E: energy error
            L: angular momentum
        """
        self.state[step, :, 0] = action
        self.state[step, :, 1] = particles.mass.value_in(self.units_m)
        self.state[step, :, 2:5] = particles.position.value_in(self.units_l)
        self.state[step, :, 5:] = particles.velocity.value_in(self.units_l/self.units_t)
        self.cons[step, 0] = action
        self.cons[step, 1] = E
        self.cons[step, 2:] = L
        self.comp_time[step-1] = T

        np.save(self.settings['Integration']['savefile'] + self.settings['Integration']['subfolder'] +\
             '_state'+ self.settings['Integration']['suffix'], self.state)
        np.save(self.settings['Integration']['savefile'] + self.settings['Integration']['subfolder'] +\
             '_cons'+ self.settings['Integration']['suffix'], self.cons)
        np.save(self.settings['Integration']['savefile'] + self.settings['Integration']['subfolder'] +\
             '_tcomp' + self.settings['Integration']['suffix'], self.comp_time)
        
    def loadstate(self):
        """
        loadstate: load from file
        OUTPUTS:
            state: positions, masses and velocities of the particles
            cons: energy error, angular momentum
            tcomp: computation time
        """
        state = np.load(self.settings['Integration']['savefile'] + self.settings['Integration']['subfolder'] +\
                         '_state'+ self.settings['Integration']['suffix']+'.npy')
        cons = np.load(self.settings['Integration']['savefile'] + self.settings['Integration']['subfolder'] +\
                        '_cons'+ self.settings['Integration']['suffix']+'.npy')
        tcomp = np.load(self.settings['Integration']['savefile'] +  self.settings['Integration']['subfolder'] +\
                         '_tcomp'+ self.settings['Integration']['suffix'] +'.npy')
        return state, cons, tcomp
    
    def plot_orbit(self):
        """
        plot_orbit: plot orbits of the bodies
        """
        state, cons = self.loadstate()

        n_bodies = np.shape(state)[1]

        for i in range(n_bodies):
            plt.plot(state[:, i, 2], state[:, i, 3], marker= 'o', label = self.names[i])
        plt.axis('equal')
        plt.grid()
        plt.legend()
        plt.show()


    def calculate_angular_m(self, particles):
        """
        calculate_angular_m: return angular momentum (units m, s, kg)
        INPUTS: 
            particles: particle set with all bodies in the system
        OUTPUTS:
            L: angular momentum vector
        """
        L = 0
        for i in range(len(particles)):
            r = particles[i].position.value_in(self.units_l)
            v = particles[i].velocity.value_in(self.units_l/self.units_t)
            m = particles[i].mass.value_in(self.units_m)
            L += np.cross(r, m*v)
        return L