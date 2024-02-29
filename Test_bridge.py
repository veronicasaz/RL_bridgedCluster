import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
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

def load_json(filepath):
    """
    load_json: load json file as dictionary
    """
    with open(filepath) as jsonFile:
        data = json.load(jsonFile)
    jsonFile.close()
    return data

def _add_bodies_inner(bodies, bodies_inner, settings):
    n_bodies = len(bodies)
    ranges = settings['Integration']['ranges_inner']
    ranges_np = np.array(list(ranges.values()))
    for bod in range(n_bodies):
        if bodies_inner[bod] > 0:
            np.random.seed(seed = 1)
            K = lhs(len(ranges), samples = bodies_inner[bod]) * (ranges_np[:, 1]- ranges_np[:, 0]) + ranges_np[:, 0] 

            for i in range(bodies_inner[bod]):
                sun, particle = generate_binaries(
                    bodies[bod].mass,
                    K[i, 0] | units.MSun,
                    K[i, 1] | units.au,
                    eccentricity = K[i, 2],
                    inclination = K[i, 3],
                    longitude_of_the_ascending_node = K[i, 5],
                    argument_of_periapsis = K[i, 4],
                    true_anomaly= K[i, 5],
                    G = constants.G)
                particle.name = "Particle_%i"%i

                # position around the main body
                particle.position += bodies[bod].position
                particle.velocity += bodies[bod].velocity
                particle.mass = K[i, 0] | units.MSun

                bodies.add_particles(particle)
        
    return bodies

def _join_particles_bridge(particles_vector):
    bodies = Particles()
    for i in range(len(particles_vector)):
        bodies.add_particles(particles_vector[i])
    return bodies

def _get_info(particles, units_energy, units, G, E_0, L_0, initial = False):
    E_kin = particles.kinetic_energy().value_in(units_energy)
    E_pot = particles.potential_energy(G = G).value_in(units_energy)
    L = calculate_angular_m(particles, units)
    if initial == True:
        return E_kin + E_pot, L
    else:
        Delta_E = (E_kin + E_pot - E_0) / E_0
        Delta_L = (L - L_0) / L_0
        return Delta_E, Delta_L
    
def calculate_angular_m(particles, units):
    """
    calculate_angular_m: return angular momentum (units m, s, kg)
    INPUTS: 
        particles: particle set with all bodies in the system
    OUTPUTS:
        L: angular momentum vector
    """
    units_m, units_l, units_t = units
    L = 0
    for i in range(len(particles)):
        r = particles[i].position.value_in(units_l)
        v = particles[i].velocity.value_in(units_l/units_t)
        m = particles[i].mass.value_in(units_m)
        L += np.cross(r, m*v)
    return L
    
def _savestate(state, cons, units, savefile, action, step, particles, E, L):
    """
    _savestate: save state of the system to file
    INPUTS:
        action: action taken
        step: simulation step
        particles: particles set
        E: energy error
        L: angular momentum
    """
    units_m, units_l, units_t = units
    state[step, :, 0] = action
    state[step, :, 1] = particles.mass.value_in(units_m)
    state[step, :, 2:5] = particles.position.value_in(units_l)
    state[step, :, 5:] = particles.velocity.value_in(units_l/units_t)
    cons[step, 0] = action
    cons[step, 1] = E
    cons[step, 2:] = L

    np.save(savefile + '_state', state)
    np.save(savefile + '_cons', cons)
    return state

def run():
    # GENERAL SETTINGS
    savefile = "./Testing_results/0_testBridge/"
    settings = load_json("./settings_integration_3Body.json")
    n_bodies = 3
    t_step_integr = [1e-6, 1e-6]
    t_step_bridge = 1e-3
    check_step = 1e-1
    steps = 100
    
    # INITIAL CONDITIONS
    G = constants.G
    units_G = units.m**3 * units.kg**(-1) * units.s**(-2)
    units_energy = units.m**2 * units.s**(-2)* units.kg
    units_time = units.yr

    units_t = units.s 
    units_l = units.m
    units_m = units.kg
    
    ranges = settings['Integration']['ranges_coords']
    ranges_np = np.array(list(ranges.values()))
    np.random.seed(seed = 1)
    K = np.random.uniform(low = ranges_np[:, 0], high = ranges_np[:, 1])

    r = np.array([[0, 0, 0],
                    [K[0], K[1], 0],
                    [K[2], K[3], 0.0]])
    v = np.array([[0, 10.0, 0],
                    [-10, 0, 0],
                    [0, 0, 0.0]])

    bodies_global = Particles(n_bodies-1)
    bodies_local = Particles(1)
    
    bodies_global.mass = (1.0, 1.0) | units.MSun # equal mass
    bodies_global[0].position = r[0] | units.au
    bodies_global[0].velocity = v[0] | units.kms
    bodies_global[1].position = r[1] | units.au
    bodies_global[1].velocity = v[1] | units.kms
        
    bodies_local.mass = (1.0) | units.MSun # equal mass
    bodies_local[0].position = r[2] | units.au
    bodies_local[0].velocity = v[2] | units.kms

    bodies_inner = settings['Integration']['n_bodies_inner']
    
    bodies_global = _add_bodies_inner(bodies_global, bodies_inner[0:n_bodies-1], settings) # first times
    bodies_local = _add_bodies_inner(bodies_local, [bodies_inner[n_bodies-1]], settings)
        
    converter = nbody_system.nbody_to_si(bodies_global.mass.sum() + bodies_local.mass.sum(), 1 | units.au)

    # INITIALIZE BRIDGE AND INTEGRATORS
    grav_global = Hermite(converter)
    grav_global.stopping_conditions.timeout_detection.enable()
    grav_global.parameters.epsilon_squared = 1e-9 | nbody_system.length**2
    grav_global.parameters.dt_param = t_step_integr[0] 
    
    grav_local = Hermite(converter)
    grav_local.stopping_conditions.timeout_detection.enable()
    grav_local.parameters.epsilon_squared = 1e-9 | nbody_system.length**2
    grav_local.parameters.dt_param = t_step_integr[1] 

    grav_global.particles.add_particles(bodies_global)
    grav_local.particles.add_particles(bodies_local)

    grav_bridge = bridge.Bridge(use_threading=False)
    grav_bridge.add_system(grav_global, (grav_local,))
    grav_bridge.add_system(grav_local, (grav_global,))
    grav_bridge.timestep = t_step_bridge | units_time

    channel = [grav_global.particles.new_channel_to(bodies_global), \
                grav_local.particles.new_channel_to(bodies_local)]

    particles_joined = _join_particles_bridge([bodies_global, bodies_local])
    n_bodies_total = len(particles_joined)
    E_0, L_0 = _get_info(particles_joined, units_energy, [units_m, units_l, units_t], G, 0,0,  initial = True)
            
    state = np.zeros((steps, n_bodies_total, 8))
    cons = np.zeros((steps, 5))
    _savestate(state, cons, [units_m, units_l, units_t], savefile, 0,0, particles_joined, 0.0, 0.0)
    info_prev = [0.0, 0.0]

    t_cumul = 0.0
    for step in range(steps):
        print("Step: %i/%i"%(step, steps))
        grav_bridge.timestep = t_step_bridge | units_time
        t_cumul += check_step
        t = (t_cumul) | units_time

        t0_step = time.time()
        grav_bridge.evolve_model(t)
        T = time.time() - t0_step

        for chan in range(len(channel)):
            channel[chan].copy()
        
        particles_joined = _join_particles_bridge([bodies_global, bodies_local])
        info_error = _get_info(particles_joined, units_energy, [units_m, units_l, units_t], G, E_0, L_0)

        _savestate(state, cons, [units_m, units_l, units_t], savefile, 0, step, particles_joined, info_error[0], info_error[1]) # save initial state

def plot():
    savefile = "./Testing_results/0_testBridge/"
    state = np.load(savefile + '_state.npy')
    cons = np.load(savefile+'_cons.npy')

    # Calculate the energy errors
    E_E = abs(cons[1:, 1]) # absolute relative energy error

    steps = np.shape(E_E)[0]
    print(steps)
    x_axis = np.arange(0, steps, 1)
    y_axis = E_E
    
    fig = plt.figure(figsize = (10,15))
    gs1 = matplotlib.gridspec.GridSpec(4, 1, 
                                    left=0.1, wspace=0.4, hspace = 1.5, right = 0.93,
                                    top = 0.9, bottom = 0.04)
    
    ax0 = fig.add_subplot(gs1[0:2, :])
    ax1 = fig.add_subplot(gs1[2:, :])
    colors = ['black', 'red', 'green', 'blue', 'orange']
    
    n_planets = np.shape(state)[1]
    name_planets = (np.arange(np.shape(state[0])[1])+1).astype(str)
    for j in range(n_planets):
        x = state[0:steps, j, 2]/1.496e11
        y = state[0:steps, j, 3]/1.496e11
        m = state[0, j, 1]
        size_marker = np.log10(m)/10

        ax0.scatter(x[0], y[0], s = 20*size_marker,\
                   c = colors[j%len(colors)], \
                    label = "Particle "+ name_planets[j])
        ax0.plot(x[1:], y[1:], marker = None, 
                    markersize = size_marker, \
                    linestyle = '-',\
                    color = colors[j%len(colors)], \
                    alpha = 0.1)
        
        ax0.scatter(x[1:], y[1:], marker = 'o', s = size_marker, \
                    c = colors[j%len(colors)])  
        
        ax0.legend()



    ax1.plot(x_axis, y_axis, color = 'black', linestyle = '-', label = ' ')
    ax1.set_xlabel('Steps', fontsize = 15)
    ax1.set_ylabel('Energy Error', fontsize = 15)
    ax1.set_yscale('log')

    plt.show()

if __name__ == '__main__':
    run()
    plot()