"""
Bridged2Body_env: environment for integration of a 2 body problem using bridge

Author: Veronica Saz Ulibarrena
Last modified: 5-March-2025

Based on https://www.gymlibrary.dev/content/environment_creation/
pip install -e ENVS needed to create a package
"""

import gym 
import numpy as np
import matplotlib.pyplot as plt
import time
import json

from amuse.units import units, constants, nbody_system
from amuse.community.hermite.interface import Hermite
from amuse.community.ph4.interface import ph4
from amuse.community.symple.interface import symple
from amuse.community.huayno.interface import Huayno
from amuse.community.bhtree.interface import BHTree
from amuse.community.fractalcluster.interface import new_fractal_cluster_model
from amuse.lab import Particles, new_powerlaw_mass_distribution
from amuse.ext.bridge import bridge
from amuse.ext.orbital_elements import get_orbital_elements_from_arrays
from amuse.ic import make_planets_oligarch

from env.InclusiveBridgeSep import Modified_Bridge

def plot_state(bodies):
    """
    plot_state: plot bodies at a given time
    INPUTS:
        bodies: particle set with the bodies
    """
    v = (bodies.vx**2 + bodies.vy**2 + bodies.vz**2).sqrt()
    plt.scatter(bodies.x.value_in(units.au),\
                bodies.y.value_in(units.au), \
                c=v.value_in(units.kms), alpha=0.5)
    plt.colorbar()
    plt.show()


def load_json(filepath):
    """
    load_json: load json file as dictionary
    INPUTS: 
        filepath: path to open
    OUTPUTS:
        data: json data as dict
    """
    with open(filepath) as jsonFile:
        data = json.load(jsonFile)
    jsonFile.close()
    return data

def get_orbital_elements_of_planetary_system(star, planets):
    """
    get_orbital_elements_of_planetary_system (unused):  get orbital elemetns from positions
    INPUTS: 
        star: central particle
        planets: orbiting particle set
    """
    total_masses = planets.mass + star.mass
    rel_pos = planets.position-star.position
    rel_vel = planets.velocity-star.velocity
    sma, ecc, true_anomaly,\
        inc, long_asc_node, arg_per_mat =\
            get_orbital_elements_from_arrays(rel_pos,
                                             rel_vel,
                                             total_masses,
                                             G=constants.G)
    planets.semimajor_axis = sma
    planets.eccentricity = ecc
    planets.inclination = inc

    planets = planets[np.isnan(planets.eccentricity)]
    planets.eccentricity = 10
    planets.semimajor_axis = 1.e+10 | units.au


class Cluster_env(gym.Env):
    def __init__(self, render_mode = None):
        """
        Cluster_env: cluster of stars with a planetary system orbiting 
        """
        self.settings = load_json("./settings_integration_Cluster.json")
        self.n_bodies = self.settings['InitialConditions']['n_bodies']
        self._initialize_RL()

    def _initialize_RL(self):
        """
        _initialize_RL: initialize RL algorithm
        """
        # STATE
        if self.settings["RL"]["state"] == 'cart':
            self.observation_space_n = 4*self.n_bodies+1
            
        elif self.settings["RL"]["state"] == 'potential':
            self.observation_space_n = 2
        elif self.settings["RL"]["state"] == 'dist':
            self.observation_space_n = 3

        self.observation_space = gym.spaces.Box(low=np.array([-np.inf]*self.observation_space_n), \
                                                high=np.array([np.inf]*self.observation_space_n), \
                                                dtype=np.float64)
            
        # ACTION
        # From a range of paramters
        if self.settings['RL']['action'] == 'range':
            low = self.settings['RL']['range_action'][0]
            high = self.settings['RL']['range_action'][-1]
            n_actions = self.settings['RL']['number_actions']
            self.actions = np.logspace(np.log10(low), np.log10(high), \
                                       num = n_actions, base = 10,
                                       endpoint = True)
            
            self.actions *= self.settings['RL']['t_step_param']

        self.action_space = gym.spaces.Discrete(len(self.actions)) 

        # Training parameters
        self.W = self.settings['RL']['weights']

    def _initial_conditions_bridge(self):
        """
        _initial_conditions_bridge: select initial conditions for the astrophysics system
        OUTPUTS:
            stars: particle set with the cluster of stars.
                Depending on the type of bridge, the common star is included or not
            planetary_system: particle set for the planetary system. 
            cluster: all bodies in the cluster and planetary system
        """
        np.random.seed(seed = self.settings['InitialConditions']['seed'])

        #################################
        if self.settings['InitialConditions']['bodies_in_system'] == 'random':
            nbodies = np.random.randint(5, self.settings['InitialConditions']['n_bodies'])
        else:
            nbodies = self.settings['InitialConditions']['n_bodies']
        # Stars
        masses = new_powerlaw_mass_distribution(nbodies,
                                            self.settings['InitialConditions']['ranges_mass'][0] |units.MSun,
                                            self.settings['InitialConditions']['ranges_mass'][1] |units.MSun, -2.35)
        Rcluster = self.settings['InitialConditions']['radius_cluster'] | units.pc
        self.converter = nbody_system.nbody_to_si(masses.sum(), Rcluster)
        stars = new_fractal_cluster_model(nbodies,
                                        fractal_dimension=1.6,
                                        convert_nbody=self.converter,
                                        random_seed = self.settings['InitialConditions']['seed'])

        stars.mass = masses
        stars.name = "star"
        stars.type = "star"
        stars.move_to_center()
        stars.scale_to_standard(self.converter, virial_ratio = self.settings['InitialConditions']['virial_ratio'])

        #################################
        # Star to get planets
        sun = stars[-1]
        sun.name = "Sun"

        #################################
        # Planetary system
        inner_radius_disk = self.settings['InitialConditions']['disk_radius'][0] | units.au
        outer_radius_disk = self.settings['InitialConditions']['disk_radius'][1] | units.au
        mass_disk = self.settings['InitialConditions']['mass_disk'] *sun.mass
        ps = make_planets_oligarch.new_system(sun.mass,
                                            sun.radius,
                                            inner_radius_disk,
                                            outer_radius_disk,
                                            mass_disk)
        planets = ps.planets[0]
        planets.name = "planet"
        planets.type = "planet"
        planets.position += sun.position
        planets.velocity += sun.velocity

        planetary_system = Particles()
        planetary_system.add_particle(sun)
        if self.settings['Training']['RemovePlanets'] == False:
            planetary_system.add_particles(planets)

        #################################
        # All together
        cluster = Particles()
        cluster.add_particles(stars)
        self.n_stars = len(cluster)
        self.index_planetarystar = self.n_stars-1
        
        if self.settings['Integration']["bridge"] == 'original':
            stars -= sun

        if self.settings['Training']['RemovePlanets'] == False:
            cluster.add_particles(planets)

        print("Planets: ", len(planetary_system)-1)

        return stars, planetary_system, cluster
        
    ## BASIC FUNCTIONS
    def reset(self):
        """
        reset: reset the simulation 
        OUTPUTS:
            state_RL: state vector to be passed to the RL
            info_prev: information vector of the previous time step (zero vector)
        """
        self.iteration = 0
        self.t_cumul = 0.0 # cumulative time for integration

        # Select units
        self.units()

        # Same time step for the integrators and the bridge
        self.particles_global, self.particles_local, \
        self.particles_joined = self._initial_conditions_bridge()
        self.particles_joined2 = self.particles_joined.copy()
        

        # TODO: tstep not implemented for now
        self.grav_global = self._initialize_integrator(self.settings["Integration"]['t_step_global'], self.settings["Integration"]['integrator_global'])
        self.grav_local = self._initialize_integrator(self.settings ["Integration"]['t_step_local'], self.settings["Integration"]['integrator_local'])
        self.grav_global.particles.add_particles(self.particles_global)
        self.grav_local.particles.add_particles(self.particles_local)

        # Bridge creation
        if self.settings['Integration']["bridge"] == 'original':
            self.grav_bridge = bridge()
        else:
            self.grav_bridge = Modified_Bridge()

        self.grav_bridge.add_system(self.grav_local, (self.grav_global,)) # particles without the sun
        self.grav_bridge.add_system(self.grav_global)

        self.channel = [self.grav_global.particles.new_channel_to(self.particles_joined),\
                        self.grav_local.particles.new_channel_to(self.particles_joined)
                        ]
        
        self.channel2 = [self.grav_global.particles.new_channel_to(self.particles_joined2),\
                         self.grav_local.particles.new_channel_to(self.particles_joined2)
                        ]
        
        self.channel_inverse = [self.particles_joined.new_channel_to(self.grav_global.particles),\
                                self.particles_joined.new_channel_to(self.grav_local.particles)
                        ]

        self.n_bodies_total = len(self.particles_joined)

        # Get initial energy and angular momentum. Initialize at 0 for relative error
        self.E_0_total = \
            self._get_info(self.particles_joined, initial = True)
        self.E_total_prev = self.E_0_total
        
        # Create state vector
        state_RL = self._get_state(self.particles_joined[0:self.n_stars], 1) # |r_i|, |v_i|

        # Initialize time
        self.grav_bridge.timestep = self.actions[0] | self.units_time
        self.check_step = self.settings['Integration']['check_step']

        # Plot trajectory
        if self.settings['Integration']['plot'] == True:
            plot_state(self.particles_joined)

        # Initialize variables to save simulation information
        if self.settings['Integration']['savestate'] == True:
            steps = self.settings['Integration']['max_steps'] + 1 # +1 to account for step 0
            self.state = np.zeros((steps, self.n_bodies_total, 9)) # action, mass, rx3, vx3, name
            self.cons = np.zeros((steps, 5)) # action, reward, E,  t
            self.comp_time = np.zeros(steps) # computation time
            self._savestate(0, 0, self.particles_joined, [1.0, 1.0],\
                             0.0, 0.0, 0) # save initial state

        self.info_prev = [0.0, 0.0]
        self.first_step = True # to begin with steps

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
        self.iteration += 1
        self.t_cumul += self.check_step # add the previous simulation time
        t = (self.t_cumul) | self.units_time

        # Apply action
        # Integrate
        timestep = self.actions[action]

        # start loop for hybrid
        counter_hybrid = 0
        if self.settings['Integration']['hybrid'] == True:
            t0_step = time.time()
            delta_e_good = False # see if energy error between consecutive steps is good
            counter_max = 4
            error_tol = 0.3
            action_i = action
            while delta_e_good == False and counter_hybrid < counter_max:

                delta_t0 = self.grav_bridge.time
                self.grav_bridge.evolve_model(t, timestep = timestep | self.units_time)
                
                for chan in range(len(self.channel2)):
                    self.channel2[chan].copy()

                # Get information for the reward
                if self.first_step == True:
                    self.E_1_total = \
                    self._get_info(self.particles_joined, initial = True)
                info_error = self._get_info(self.particles_joined2, save_step = False)
                if info_error[0][1] == 0:
                    error = np.log10(abs(info_error[0][0]))
                else:
                    error = np.log10(abs(info_error[0][0]))-np.log10(abs(info_error[0][1]))
                
                if error >error_tol and\
                    counter_hybrid < counter_max-1: #half an order of magnitude jump
                    # self.grav_bridge.time -= delta_t # redo time
                    if action_i == 0:
                        timestep = timestep/2       
                    else:
                        action_i -= 1
                        timestep = self.actions[action_i]
                    t+= self.check_step | self.units_time
                    for chan in range(len(self.channel_inverse)): #redo positions of particles
                        self.channel_inverse[chan].copy()
                    counter_hybrid += 1

                    # print(self.grav_global.particles[0])
                elif error> error_tol and counter_hybrid >= counter_max-1: # run no matter what
                    t+= self.check_step | self.units_time
                    if action_i == 0:
                        timestep = timestep/2       
                    else:
                        action_i -= 1
                        timestep = self.actions[action_i]
                    # self.grav_bridge.time -= delta_t # redo time
                    for chan in range(len(self.channel_inverse)): #redo positions of particles
                        self.channel_inverse[chan].copy()
                    self.grav_bridge.evolve_model(t, timestep = timestep | self.units_time)                    
                    counter_hybrid += 1

                else:
                    delta_e_good = True

        else:
            t0_step = time.time()
            self.grav_bridge.evolve_model(t, timestep = timestep | self.units_time)
            
        for chan in range(len(self.channel)):
            self.channel[chan].copy()

        T = time.time() - t0_step
        
        # Take energy error with respect to the first case
        if self.first_step == True:
            self.E_1_total = \
            self._get_info(self.particles_joined, initial = True)
        self.first_step = False

        info_error = self._get_info(self.particles_joined)
        state = self._get_state(self.particles_joined[0:self.n_stars], info_error[1])
        
        # Using total energy
        reward = self._calculate_reward(info_error[1], self.info_prev[1], T, self.actions[action], self.W) # Use computation time for this step, including changing integrator
        self.info_prev = info_error
        
        if self.settings['Integration']['savestate'] == True:
            self._savestate(action, self.iteration, self.particles_joined, \
                            [info_error[0][2], info_error[1]],\
                            T, reward, counter_hybrid) # save initial state
            
        # finish experiment if max number of iterations is reached
        if (abs(info_error[1]) > self.settings['Integration']['max_error_accepted']) or\
              self.iteration == self.settings['Integration']['max_steps']:
            terminated = True
        else:
            terminated = False
            
        # Display information at each step
        if self.settings['Training']['display'] == True:
            self._display_info(info_error, reward, action)

        # Plot trajectory
        if self.settings['Integration']['plot'] == True and terminated == True:
            plot_state(self.particles_joined)

        info = dict()
        info['TimeLimit.truncated'] = False
        info['Energy_error'] = info_error[1]
        info['Energy_error_rel'] = info_error[0][2]
        info['tcomp'] = T

        return state, reward, terminated, info
    
    def close(self):
        """
        close: close integrators
        """
        self.grav_global.stop()
        self.grav_local.stop()

    ## ADDITIONAL FUNCTIONS NEEDED
    def units(self):
        """
        units: choose standard units
        """
        # Choose set of units for the problem
        if self.settings['InitialConditions']['units'] == 'si':
            self.G = constants.G
            self.units_G = units.m**3 * units.kg**(-1) * units.s**(-2)
            self.units_energy = units.m**2 * units.s**(-2)* units.kg
            self.units_time = units.Myr

            self.units_t = units.s 
            self.units_l = units.m
            self.units_m = units.kg

        elif self.settings['Integration']['units'] == 'nbody':
            self.G = self.converter.to_nbody(constants.G)
            self.units_G = nbody_system.length**3 * nbody_system.mass**(-1) * nbody_system.time**(-2)
            self.units_energy = nbody_system.length**2 * nbody_system.time**(-2)* nbody_system.mass
            self.units_time = self.converter.to_nbody(1 | units.yr)
            self.units_t = nbody_system.time
            self.units_l = nbody_system.length
            self.units_m = nbody_system.mass

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
    
    def _initialize_integrator(self, tstep, integrator_type):
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
            g.parameters.dt_param = tstep
            # g.parameters.epsilon_squared = 1e-9 | nbody_system.length**2
        elif integrator_type == 'Ph4': 
            if self.settings['InitialConditions']['units'] == 'si':
                g = ph4(self.converter, number_of_workers = 1)
            else:
                g = ph4()
            g.parameters.timestep_parameter =  tstep
            # g.parameters.epsilon_squared = 1e-9 | nbody_system.length**2 # Softening
        elif integrator_type == 'Huayno': 
            if self.settings['InitialConditions']['units'] == 'si':
                g = Huayno(self.converter)
            else:
                g = Huayno()
            g.parameters.timestep_parameter =  tstep
            # g.parameters.epsilon_squared = 1e-9 | nbody_system.length**2 # Softening
        elif integrator_type == 'Symple': 
            if self.settings['InitialConditions']['units'] == 'si':
                g = symple(self.converter, redirection ='none')
            else:
                g = symple(redirection ='none')
            g.initialize_code()
            # g.parameters.timestep = tstep | self.units_time

        elif integrator_type == 'BHTree':
            if self.settings['InitialConditions']['units'] == 'si':
                g = BHTree(self.converter)
            else:
                g = BHTree()
            # g.parameters.timestep 

        return g 
    
    def _get_info(self, particles, initial = False, save_step = True):
        """
        _get_info: get energy error, angular momentum error at current state
        OUTPUTS:
            Step energy error
        """
        E_kin = particles.kinetic_energy().value_in(self.units_energy)
        E_pot = particles.potential_energy(G = self.G).value_in(self.units_energy)
        E_total = E_kin + E_pot
        
        if initial == True:
            return E_total
        else:
            Delta_E_rel = (E_total - self.E_0_total)/self.E_0_total
            Delta_E_prev_rel = (self.E_total_prev - self.E_0_total)/self.E_0_total
            if save_step == True:
                self.E_total_prev = E_total
            Delta_E_total = (E_total - self.E_0_total)/self.E_0_total
            Delta_E_rel_firststep = (E_total - self.E_1_total)/self.E_1_total
            return [Delta_E_rel, Delta_E_prev_rel, Delta_E_rel_firststep], \
                  Delta_E_total\
                    
    def _get_state(self, particles, E):  # TODO: change to include all particles?
        """
        _get_state: create the state vector
        Options:
            - norm: norm of the positions and velocities of each body and the masses
            - cart: 2D cartesian coordinates of the position and angular momentum plus the energy error
            - dis: distance between particles in position and momentum space plus the energy error
            - potential: potential at the position of the particle and energy error

        INPUTS: 
            particles: particle set with all bodies
            E: energy error
        OUTPUTS: 
            state: state array to be given to the reinforcement learning algorithm
        """
        particles_p_nbody = self.converter.to_generic(particles[0:self.n_stars].position).value_in(nbody_system.length)
        particles_v_nbody = self.converter.to_generic(particles[0:self.n_stars].velocity).value_in(nbody_system.length/nbody_system.time)
        particles_m_nbody = self.converter.to_generic(particles[0:self.n_stars].mass).value_in(nbody_system.mass)

        if self.settings['RL']['state'] == 'norm':
            state = np.zeros((self.n_bodies)*3) # m, norm r, norm v

            state[0:self.n_bodies] = particles_m_nbody
            state[self.n_bodies: 2*self.n_bodies]  = np.linalg.norm(particles_p_nbody, axis = 1)
            state[2*self.n_bodies: 3*self.n_bodies] = np.linalg.norm(particles_v_nbody, axis = 1)
       
        elif self.settings['RL']['state'] == 'cart':
            state = np.zeros((self.n_bodies)*4+1) # all r, all v
            for i in range(self.n_bodies):
                state[2*i:2*i+2] = particles_p_nbody[i, 0:2]/10 # convert to 2D. Divide by 10 to same order as v
                state[2*self.n_bodies + 2*i: 2*self.n_bodies + 2*i+2] = particles_v_nbody[i, 0:2]
                state[-1] = -np.log10(abs(E))
        
        elif self.settings['RL']['state'] == 'dist':
            state = np.zeros(3) # potential, mass, energy error

            distance = []
            for i in range(self.n_bodies):
                for j in range(i+1, self.n_bodies):
                    d  = np.linalg.norm(particles_p_nbody[i,:]-particles_p_nbody[j,:], axis = 0) /10
                    distance.append(d)

            min_distance = min(np.array(distance))

            pot = particles[self.index_planetarystar].potential()
            pot_nbody = self.converter.to_generic(pot).value_in(nbody_system.length**2/nbody_system.time**2)
            
            state[0] = min_distance
            state[1] = particles_m_nbody[self.index_planetarystar]
            state[2] = -np.log10(abs(E))

        elif self.settings['RL']['state'] == 'potential':
            state = np.zeros(2) # potential, mass, energy error

            pot = particles[self.index_planetarystar].potential()
            pot_nbody = self.converter.to_generic(pot).value_in(nbody_system.length**2/nbody_system.time**2)
            
            state[0] = pot_nbody
            state[1] = -np.log10(abs(E))

        return state
    
    def _calculate_reward(self, info, info_prev, T, action, W):
        """
        _calculate_reward: calculate the reward associated to a step
        INPUTS:
            info: energy error and change of angular momentum of iteration i
            info_prev: energy error and change of angular momentum of iteration i-1
            T: clock computation time
            action: action taken. Integer value
            W: weights for the terms in the reward function
        OUTPUTS:
            a: reward value
        """
        Delta_E = info
        Delta_E_prev= info_prev

        if Delta_E_prev == 0.0: # for the initial step
            return 0
        else:
            if self.settings['RL']['reward_f'] == 1:
                a = -W[0]* (np.log10(abs(Delta_E)/1e-10)/\
                         abs(np.log10(abs(Delta_E)))**3/self.iteration) +\
                         W[2]/abs(np.log10(action))
                return a

            
    def _display_info(self, info, reward, action):
        """
        _display_info: display information at every step
        INPUTS:
            info: energy error and angular momentum vector
            reward: value of the reward for the given step
            action: action taken at this step
        """
        print("Iteration: %i/%i, Delta E_E total = %0.3E, Action: %i, Reward: %.4E"%(self.iteration, \
                                 self.settings['Integration']['max_steps'],\
                                 info[0][2],\
                                 action, \
                                 reward))
            
    def _savestate(self, action, step, particles, E, T, R, hybrid_n):
        """
        _savestate: save state of the system to file
        INPUTS:
            action: action taken
            step: simulation step
            particles: particles set
            E: energy error
            T: computation time
            R: reward
            hybrid_n: number of iterations of the hybrid method
        """
        self.state[step, :, 0] = action
        self.state[step, :, 1] = particles.mass.value_in(self.units_m)
        self.state[step, :, 2:5] = particles.position.value_in(self.units_l)
        self.state[step, :, 5:8] = particles.velocity.value_in(self.units_l/self.units_t)

        particles_name_code = []
        for i in range(len(particles)):
            if particles[i].name == 'star':
                particles_name_code.append(0)
            else:
                particles_name_code.append(1)

        self.state[step, :, 8] = particles_name_code
        self.cons[step, 0] = action
        self.cons[step, 1] = R
        self.cons[step, 2] = E[0]
        self.cons[step, 3] = E[1]
        self.cons[step, 4] = hybrid_n
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


    def reset_withoutBridge(self):
        """
        reset_ithoutBridge: reset the simulation for a case with direct integration
        OUTPUTS:
            state_RL: state vector to be passed to the RL
            info_prev: information vector of the previous time step (zero vector)
        """
        self.iteration = 0
        self.t_cumul = 0.0 # cumulative time for integration

        # Select units
        self.units()

        # Same time step for the integrators and the bridge
        self.particles_global, self.particles_local, \
        self.particles_joined = self._initial_conditions_bridge()
        
        self.grav = self._initialize_integrator(self.settings["Integration"]['t_step_global'], self.settings["Integration"]['integrator_global'])
        self.grav.particles.add_particles(self.particles_joined)

        self.channel = self.grav.particles.new_channel_to(self.particles_joined)

        self.n_bodies_total = len(self.particles_joined)

        # Get initial energy and angular momentum. Initialize at 0 for relative error
        self.E_0_total = \
            self._get_info(self.particles_joined, initial = True)
        self.E_total_prev = self.E_0_total
        
        # Create state vector
        state_RL = self._get_state(self.particles_joined[0:self.n_stars], 1) # |r_i|, |v_i|

        # Initialize time
        self.check_step = self.settings['Integration']['check_step']

        # Plot trajectory
        if self.settings['Integration']['plot'] == True:
            plot_state(self.particles_joined)

        # Initialize variables to save simulation information
        if self.settings['Integration']['savestate'] == True:
            steps = self.settings['Integration']['max_steps'] + 1 # +1 to account for step 0
            self.state = np.zeros((steps, self.n_bodies_total, 9)) # action, mass, rx3, vx3, name
            self.cons = np.zeros((steps, 5)) # action, reward, E,  t
            self.comp_time = np.zeros(steps) # computation time
            self._savestate(0, 0, self.particles_joined, [1.0, 1.0],\
                             0.0, 0.0,0 ) # save initial state

        self.info_prev = [0.0, 0.0]
        
        self.first_step = True
            
        return state_RL, self.info_prev
    
    def step_withoutBridge(self, action):
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
        self.iteration += 1
        t0 = self.t_cumul 
        self.t_cumul += self.check_step # add the previous simulation time
        t = (self.t_cumul) | self.units_time

        # Apply action
        # Integrate
        t0_step = time.time()
        self.grav.evolve_model(t)
        self.channel.copy()
        T = time.time() - t0_step
            
        # Get information for the reward
        if self.first_step == True:
            self.E_1_total = \
            self._get_info(self.particles_joined, initial = True)
        info_error = self._get_info(self.particles_joined)
        state = self._get_state(self.particles_joined[0:self.n_stars], info_error[1])
        reward = self._calculate_reward(info_error[1], self.info_prev[1], T, self.actions[action], self.W) # Use computation time for this step, including changing integrator
        self.info_prev = info_error
        
        if self.settings['Integration']['savestate'] == True:
            self._savestate(action, self.iteration, self.particles_joined, \
                            [info_error[0][0], info_error[1]],\
                            T, reward, 0) # save initial state
            
        
        # finish experiment if max number of iterations is reached
        if (abs(info_error[1]) > self.settings['Integration']['max_error_accepted']) or\
              self.iteration == self.settings['Integration']['max_steps']:
            terminated = True
        else:
            terminated = False
            
        # Display information at each step
        if self.settings['Training']['display'] == True:
            self._display_info(info_error, reward, action)

        # Plot trajectory
        if self.settings['Integration']['plot'] == True and terminated == True:
            plot_state(self.particles_joined)

        info = dict()
        info['TimeLimit.truncated'] = False
        info['Energy_error'] = info_error[1]
        info['Energy_error_rel'] = info_error[0]
        info['tcomp'] = T

        return state, reward, terminated, info
    
    def close_withoutBridge(self):
        """
        close_withoutBridge: close environment
        """
        self.grav.stop()