import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

np.random.seed(0)

class Spring:
	def __init__(self, num_particles=2, box_size=5., loc_std=.5, vel_norm=.5,
				 interaction_strength=.1, noise_var=0.):
		self.num_particles = num_particles
		self.box_size = box_size
		self.loc_std = loc_std
		self.vel_norm = vel_norm
		self.interaction_strength = interaction_strength
		self.noise_var = noise_var
		
		self.spring_prob = [0.5, 0.0, 0.5]
		self._spring_types = np.array([0., 0.5, 1.])
		self._delta_T = 0.001
		self._max_F = 0.1 / self._delta_T
		
		self.positions = []
		self.velocities = []
		self.edges = None
	
	def _clamp(self, loc, vel):
		'''
		:param loc: 2xN location at one time stamp
		:param vel: 2xN velocity at one time stamp
		:return: location and velocity after hiting walls and returning after
			elastically colliding with walls
		'''
		assert (np.all(loc < self.box_size * 3))
		assert (np.all(loc > -self.box_size * 3))
		
		over = loc > self.box_size
		loc[over] = 2 * self.box_size - loc[over]
		assert (np.all(loc <= self.box_size))
		
		# assert(np.all(vel[over]>0))
		vel[over] = -np.abs(vel[over])
		
		under = loc < -self.box_size
		loc[under] = -2 * self.box_size - loc[under]
		# assert (np.all(vel[under] < 0))
		assert (np.all(loc >= -self.box_size))
		vel[under] = np.abs(vel[under])
		
		return loc, vel
	
	def get_init_pos_velocity(self):
		init_position = np.random.randn(2, self.num_particles) * self.loc_std
		init_velocity = np.random.randn(2, self.num_particles)
		# Compute magnitude of this velocity vector and format to right shape
		v_norm = np.linalg.norm(init_velocity, axis=0)
		# print(np.sqrt((init_velocity ** 2).sum(axis=0)).reshape(1, -1))
		# Scale the magnitude ?
		init_velocity = init_velocity * self.vel_norm / v_norm
		return init_position, init_velocity
	
	def get_force(self, edges, current_positions):
		'''
		:param edges: Adjacency matrix representing mutual causality
		:param current_positions: current coordinates of all particles
		:return: F
		'''
		force_matrix = - self.interaction_strength * edges
		np.fill_diagonal(force_matrix, 0)
		x_cords, y_cords = current_positions[0, :], current_positions[1, :]
		x_diffs = np.subtract.outer(x_cords, x_cords).reshape(1, self.num_particles, self.num_particles)
		y_diffs = np.subtract.outer(y_cords, y_cords).reshape(1, self.num_particles, self.num_particles)
		# F1/F2 = (m1*d2)/(m2*d1)
		# F2 = F * D
		F = (force_matrix.reshape(1, self.num_particles, self.num_particles) *
			 np.concatenate((x_diffs, y_diffs))).sum(axis=-1)
		F[F > self._max_F] = self._max_F
		F[F < -self._max_F] = -self._max_F
		return F
	
	def generate_edges(self):
		# Sample nxn springs _spring_types which each holding a probability spring_prob
		edges = np.random.choice(self._spring_types, size=(self.num_particles, self.num_particles), p=self.spring_prob)
		# Establish symmetry causal interaction
		edges = np.tril(edges) + np.tril(edges, -1).T
		# Nullify self interaction or causality
		np.fill_diagonal(edges, 0)
		return edges
	
	def sample_trajectory(self, T=10000, sample_freq=10):
		
		# Initialize causality between particles.
		self.edges = self.generate_edges()
		edges = self.edges
		
		# Initialize the first position and velocity from a distribution
		init_position, init_velocity = self.get_init_pos_velocity()
		
		# Adding initial position and velocity of particles to trajectory.
		init_position, init_velocity = self._clamp(init_position, init_velocity)
		self.positions.append(init_position)
		self.velocities.append(init_velocity)
		# self.tr.add_snap_shot(self._clamp(init_position, init_velocity))
		
		# Compute initial forces between particles.
		init_force_between_particles = self.get_force(edges, init_position)
		
		# Compute new velocity.
		'''
		F = m * (dv/dt), for unit mass
		dv = dt * F
		velocity - current_velocity = dt * F
		velocity = current_velocity + (self._delta_T * F)
		'''
		get_velocity = lambda init_velocity, forces: init_velocity + (self._delta_T * forces)
		
		velocity = get_velocity(init_velocity, init_force_between_particles)
		current_position = init_position
		
		for i in range(1, T):
			
			# Compute new position based on current velocity and positions.
			new_position = current_position + (self._delta_T * velocity)
			new_position, velocity = self._clamp(new_position, velocity)
			
			# Adding new position and velocity of particles to trajectory.
			if i % sample_freq == 0:
				self.positions.append(new_position)
				self.velocities.append(velocity)
				#self.tr.add_snap_shot((new_position, velocity))
			
			# Compute forces between particles
			force_between_particles = self.get_force(edges, new_position)
			
			# Compute new velocity based on current velocity and forces between particles.
			new_velocity = velocity + (self._delta_T * force_between_particles)
			
			# Update velocity and position
			velocity = new_velocity
			current_position = new_position
			
			# Add noise to observations
			current_position += np.random.randn(2, self.num_particles) * self.noise_var
			velocity += np.random.randn(2, self.num_particles) * self.noise_var
			
		return self.positions, self.velocities, self.edges
	
	def get_energy(self):
		'''
		Total Energy = Kinetic Energy (K) + Potential Energy (U)
		Kinetic Energy (K) = (1/2) * m * velocity^2 : unit mass m
		Potential Energy (U) = m * g * h: h is distance, g is field, unit mass m
		:return: energy
		'''
		
		# Compute Kinetic Energy at each snap shot
		ek = lambda velocity: 0.5 * (velocity ** 2).sum()
		kinetic_energies = [ek(velocities) for velocities in self.velocities]
		
		# Compute Potential Energy at each snap shot
		potential_energies = []
		for position in positions:
			U = 0
			_pos = position.T
			for particle_index in range(0, self.num_particles):
				position_fill_mat = np.full(_pos.shape, _pos[particle_index])
				distances = np.sqrt(np.square(_pos - position_fill_mat).sum(axis=1))
				U += 0.5 * self.interaction_strength * np.dot(self.edges[particle_index], distances ** 2) / 2
			potential_energies.append(U)
		
		total_energy = np.add(kinetic_energies, potential_energies)
		return total_energy
	
	def plot(self):
		plt.figure()
		axes = plt.gca()
		axes.set_xlim([-5., 5.])
		axes.set_ylim([-5., 5.])
		positions = [position for position in self.positions]
		positions = np.asarray(positions)
		for i in range(positions.shape[-1]):
			plt.plot(positions[:, 0, i], positions[:, 1, i])
			plt.plot(positions[0, 0, i], positions[0, 1, i], 'd')
		plt.figure()
		energies = self.get_energy()
		plt.plot(energies)
		plt.show()
	
	def plot_sns(self):
		positions = [position for position in self.positions]
		positions = np.asarray(positions)
		entries = []
		for i in range(positions.shape[-1]):
			#plt.plot(positions[:, 0, i], positions[:, 1, i])
			#plt.plot(positions[0, 0, i], positions[0, 1, i], 'd')
			for time_step in range(positions.shape[-1]):
				entries.append({'time_step': time_step,
								'particle': i,
								'x_dim': positions[time_step, 0, i],
								'y_dim': positions[time_step, 1, i]})
		dframe = pd.DataFrame(entries)
		sns.lineplot(data=dframe, x='time_step', y='average_reward', hue='epsilon')
		#energies = self.get_energy()
		#plt.plot(energies)
		#plt.show()
		
if __name__ == '__main__':
	sim = Spring(num_particles=5)
	t = time.time()
	positions, velocities, edges = sim.sample_trajectory(T=5000, sample_freq=100)
	print("Simulation time: {}".format(time.time() - t))
	sim.get_energy()
	sim.plot_sns()