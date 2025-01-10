"""
Bridged2Body_env: environment for integration of a 2 body problem using bridge

Author: Veronica Saz Ulibarrena
Last modified: 8-February-2024

Based on https://www.gymlibrary.dev/content/environment_creation/
pip install -e ENVS needed to create a package
"""

import numpy as np
import threading


from amuse.units import units, constants, nbody_system
from amuse.units.quantities import sign
from amuse.ext.bridge import bridge
from amuse.units import quantities
import matplotlib.pyplot as plt

from amuse.lab import Particles, new_powerlaw_mass_distribution


def kick_system(system, get_gravity, dt):
    parts=system.particles.copy()
    ax,ay,az=get_gravity(parts.radius,parts.x,parts.y,parts.z)
    parts.vx=parts.vx+dt*ax
    parts.vy=parts.vy+dt*ay
    parts.vz=parts.vz+dt*az
    channel=parts.new_channel_to(system.particles)
    channel.copy_attributes(["vx","vy","vz"])   
#    parts.copy_values_of_all_attributes_to(system.particles)
    
class Modified_Bridge(bridge):
    def __init__(self,verbose=False,method=None, use_threading=True, time=None):
        # super(Modified_Bridge, self).__init__()
        self.units_energy = units.m**2 *units.kg*units.s**(-2)
        self.G = constants.G

        self.systems=list()
        self.partners=dict()
        self.time_offsets=dict()
        if time is None:
            time=quantities.zero
        self.time=time
        self.do_sync=dict()
        self.verbose=verbose
        self.timestep=None
        self.method=method
        self.use_threading=use_threading

    def add_system(self, interface,  partners=list(),do_sync=True):
        """
        add a system to bridge integrator  
        """
        if hasattr(interface,"model_time"):
            self.time_offsets[interface]=(self.time-interface.model_time)
        else:
            self.time_offsets[interface]=quantities.zero     
        self.systems.append(interface)
        for p in partners:
            if not hasattr(p,"get_gravity_at_point"):
                return -1
        self.partners[interface]=partners
        self.do_sync[interface]=do_sync  
        return 0
    
    def evolve_model(self,tend,timestep=None):
        """
        evolve combined system to tend, timestep fixes timestep
        """
        if timestep is None:
            if self.timestep is None:
                timestep=tend-self.time
            else:
                timestep=self.timestep
        
        timestep=sign(tend-self.time)*abs(timestep)
                
        if self.method==None:
          return self.evolve_joined_leapfrog(tend,timestep)
        else:
          return self.evolve_simple_steps(tend,timestep) 
        
    def evolve_joined_leapfrog(self,tend,timestep):
        self.find_common_particles()
        
        first=True
        self._drift_time=self.time
        self._kick_time=self.time
        for x in self.systems:
            if self.partners[x]:
                DeltaE_0 = self.get_info_error(x)

        # E1 = DeltaE_0

        # self.save_partner = self.partners[self.systems[0]][0].particles.copy()
        while sign(timestep)*(tend - self.time) > sign(timestep)*timestep/2:      #self.time < (tend-timestep/2):
            # one at a time
            self.synchronize_particles()
            
            for x in self.systems:
                if self.partners[x]:
                    # print("******************")
                    # print(x.particles)
                    # print(self.partners[x][0].particles)
                    # Delta_E, E = self.get_info_error(x, DeltaE_0)
                    # print("Energy error drift cluster %3E"%Delta_E)
                    self.get_updated_particles(x)
                    Delta_E, E = self.get_info_error(x, DeltaE_0)
                    print("Energy error update one system %3E"%Delta_E)
                    
                    # print('=======Kick===========')
                    # if first:      
                    #     self.kick_one_system(x, timestep/2)
                    # else:
                    self.kick_one_system(x, timestep)
                    Delta_E, E = self.get_info_error(x, DeltaE_0)
                    print("Energy error kick one system %3E"%Delta_E)
                    print("************************")
                    # Delta_E, E = self.get_info_error(x, DeltaE_0)
                    # print("Energy error kick one system %3E"%Delta_E)
                    
                    # print('=======Drift===========')
                    self.drift_one_system(x, self.time+timestep)
                    # Delta_E, E1 = self.get_info_error(x, DeltaE_0)
                    # print("Energy error drift one system %3E"%Delta_E)
                    # print((E-E1)/DeltaE_0 )

                    # print('=======Update===========')
                    self.update_particles(x)
                    # Delta_E, E = self.get_info_error(x, DeltaE_0)
                    # print("Energy error update particles %3E"%Delta_E)
                else:
                    # print(x.particles)
                    self.drift_one_system(x, self.time+timestep)
            # print((E-E1, E1-DeltaE_0,  DeltaE_0-DeltaE_0)/DeltaE_0)
            # E1 = E

            first = False    
            self.time=self.time+timestep

        for x in self.systems:
            if self.partners[x]:
                self.get_updated_particles(x)
        return 0
        

    def plot_particles(self, particles):
        colors = ['black', 'red', 'green', 'yellow', 'orange', 'pink', 'blue']
        for body in range(5):
            x = particles[body].position[0].value_in(units.au)
            y = particles[body].position[1].value_in(units.au)
            plt.scatter(x, y, color = colors[body%len(colors)])
        # plt.xlim((-3050, -2950))
        # plt.ylim((-9300, -9000))
        plt.show()

    def get_info_error(self, x, DeltaE_0 = None):
        particles_grouped = Particles()
        particles_grouped.add_particles(x.particles)
      
        for y in self.partners[x]:
            particles_grouped.add_particles(y.particles[0:-1])

        # print(particles_grouped)

        # self.plot_particles(particles_grouped)

        DeltaE = particles_grouped.kinetic_energy().value_in(self.units_energy)+\
                 particles_grouped.potential_energy(G = self.G).value_in(self.units_energy)
        if DeltaE_0 != None:
            DeltaE_rel = (DeltaE-DeltaE_0)/DeltaE_0
            return DeltaE_rel, DeltaE
        else:
            return DeltaE
        
    def find_common_particles(self):
        self.particle_pairs = dict()
        self.channel_particlepairs = dict()
        for x in self.systems:
            for y in self.partners[x]:
                key = np.intersect1d(x.particles.key, y.particles.key)
                index1 = np.where(x.particles.key == key)[0][0]
                index2 = np.where(y.particles.key == key)[0][0]
                self.particle_pairs[x] = [index1, index2]
            self.mass_common = x.particles[index1].mass

    def get_center_of_mass(self, x):
        center_r = x.particles[0].position *0
        center_v = x.particles[0].velocity *0
        self.total_mass = np.sum(x.particles.mass)
        for i in range(len(x.particles)):
            center_r += (x.particles[i].position * x.particles[i].mass)/self.total_mass
            center_v += (x.particles[i].velocity * x.particles[i].mass)/self.total_mass
        return center_r, center_v

    
    def get_updated_particles(self, x):
        for y in  self.partners[x]:
            # self.channel_particlepairs[x].copy()

            # self.part_partner = y.particles.copy()
            index1 = self.particle_pairs[x][0]
            index2 = self.particle_pairs[x][1]

            # Get difference in center of mass movement
            # center_0_r, center_0_v = self.get_center_of_mass(x)
            # center_1_r = y.particles[index2].position
            # center_1_v = y.particles[index2].velocity

            # diff_position = center_1_r - center_0_r
            # diff_velocity =  center_1_v - center_0_v

            # Get difference ignoring planets
            x.particles[index1].mass = y.particles[index2].mass
            diff_position = y.particles[index2].position - x.particles[index1].position 
            diff_velocity = y.particles[index2].velocity - x.particles[index1].velocity

            x.particles.position += diff_position
            x.particles.velocity += diff_velocity

    def update_particles(self, x):
        "x is the system to update"

        for y in  self.partners[x]:
            index1 = self.particle_pairs[x][0]
            index2 = self.particle_pairs[x][1]
            
            # center of mass approach
            # center_0_r, center_0_v = self.get_center_of_mass(x)
            # y.particles[index2].mass = self.total_mass # center of mass of the planetary system
            # y.particles[index2].position = center_0_r
            # y.particles[index2].velocity = center_0_v

            # Moves particle twice because of kick
            # y.particles[index2].mass = x.particles[index1].mass
            # y.particles[index2].position = x.particles[index1].position
            # y.particles[index2].velocity = x.particles[index1].velocity
            # y.particles.add_particle(self.part)

            # Updates with itsef in previous version
            y.particles[index2].mass = self.part_mass
            y.particles[index2].position = self.part_pos
            y.particles[index2].velocity = self.part_vel

            diff_x = x.particles[index1].position - y.particles[index2].position
            diff_y = x.particles[index1].velocity - y.particles[index2].velocity

            # take back movement of the center of mass
            # x.particles[index2].mass = self.part_mass
            x.particles[index1].position -= diff_x
            x.particles[index1].velocity -= diff_y


    def remove_common(self, x, y):
        self.part_mass =  y.particles[self.particle_pairs[x][1]].mass
        self.part_pos = y.particles[self.particle_pairs[x][1]].position  
        self.part_vel = y.particles[self.particle_pairs[x][1]].velocity
        # y.particles -= self.part

        # x.particles[self.particle_pairs[x][0]].mass *= 0  # change so that it has no potential
        # x.particles[self.particle_pairs[x][0]].position *= 0 
        # x.particles[self.particle_pairs[x][0]].velocity *= 0 

        y.particles[self.particle_pairs[x][1]].mass *= 0  # change so that it has no potential
        y.particles[self.particle_pairs[x][1]].position *= 0 
        y.particles[self.particle_pairs[x][1]].velocity *= 0 

    def restore_central_star(self, x):
        x.particles[self.particle_pairs[x][0]].velocity = self.part_vel

    def synchronize_particles(self):
        for x in self.systems:
            if self.do_sync[x]:
                if hasattr(x,"synchronize_model"):
                    if(self.verbose): print(x.__class__.__name__,"is synchronizing", end=' ')
                    x.synchronize_model()   
                    if(self.verbose):  print(".. done")
                
    
    def kick_systems(self,dt):
        for x in self.systems:
            if self.do_sync[x]:
                if hasattr(x,"synchronize_model"):
                    if(self.verbose): print(x.__class__.__name__,"is synchronizing", end=' ')
                    x.synchronize_model()    
                    if(self.verbose):  print(".. done")
        for x in self.systems:
            if hasattr(x,"particles") and len(x.particles)>0:
                for y in self.partners[x]:
                    if x is not y:
                        self.remove_common(x, y)
                        if(self.verbose):  print(x.__class__.__name__,"receives kick from",y.__class__.__name__, end=' ')
                        kick_system(x,y.get_gravity_at_point,dt)
                        if(self.verbose):  print(".. done")
        return 0
    
    def drift_systems(self,tend):
        threads=[]
        for x in self.systems:
            if hasattr(x,"evolve_model"):
                offset=self.time_offsets[x]
                if(self.verbose):
                    print("evolving", x.__class__.__name__, end=' ')
                threads.append(threading.Thread(target=x.evolve_model, args=(tend-offset,)) )

        if self.use_threading:
            for x in threads:
                x.start()            
            for x in threads:
                x.join()
        else:
            for x in threads:
                x.run()
        if(self.verbose): 
            print(".. done")
        return 0
    

    def kick_one_system(self,x, dt):
        if hasattr(x,"particles") and len(x.particles)>0:
            for y in self.partners[x]:
                if x is not y:
                    if(self.verbose):  print(x.__class__.__name__,"receives kick from",y.__class__.__name__, end=' ')
                    self.remove_common(x, y)            
                    kick_system(x, y.get_gravity_at_point, dt)
                    self.restore_central_star(x)
                    if(self.verbose):  print(".. done")
        return 0
    
    def drift_one_system(self,x, tend):
        threads=[]
        if hasattr(x, "evolve_model"):
            offset=self.time_offsets[x]
            if(self.verbose):
                print("evolving", x.__class__.__name__, end=' ')
            x.set_time = self.time
            threads.append(threading.Thread(target=x.evolve_model, args=(tend-offset,)) )
            
        if self.use_threading:
            for x in threads:
                x.start()            
            for x in threads:
                x.join()
        else:
            for x in threads:
                x.run()
        if(self.verbose): 
            print(".. done")
        return 0