import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



class Spring:
	def __init__(self, num_particles=2, interaction_strength=.1, dynamics='static', min_steps=50, max_steps=200):
		np.random.seed(0)
		self.num_particles = num_particles
		self.interaction_strength = interaction_strength
		self.dynamics = dynamics
		self.min_steps = min_steps
		self.max_steps = max_steps

		self.box_size = 5.
		self.loc_std = .5
		self.vel_norm = .5
		self.noise_var = 0.
		
		self.spring_prob = [0.5, 0.0, 0.5]
		self._spring_types = np.array([0., 0.5, 1.])
		self._delta_T = 0.001
		self._max_F = 0.1 / self._delta_T
		
		self.positions = []
		self.velocities = []
		self.edges = []
		self.edge_counter = self.get_edge_counter(self.min_steps, self.max_steps+1)

		self.columns = [f'particle_{i}' for i in range(self.num_particles)]
	
	def _clamp(self, loc, vel):
		"""
		:param loc: 2xN location at one time stamp
		:param vel: 2xN velocity at one time stamp
		:return: location and velocity after hiting walls and returning after
			elastically colliding with walls
		"""
		assert (np.all(loc < self.box_size * 3))
		assert (np.all(loc > -self.box_size * 3))
		
		over = loc > self.box_size
		loc[over] = 2 * self.box_size - loc[over]
		assert (np.all(loc <= self.box_size))

		vel[over] = -np.abs(vel[over])
		
		under = loc < -self.box_size
		loc[under] = -2 * self.box_size - loc[under]

		assert (np.all(loc >= -self.box_size))
		vel[under] = np.abs(vel[under])
		
		return loc, vel

	def get_edge_counter(self, min_steps, max_steps):
		counter = np.random.choice(list(range(min_steps, max_steps)), size=(self.num_particles, self.num_particles))
		counter = np.tril(counter) + np.tril(counter, -1).T
		np.fill_diagonal(counter, 0)
		return counter


	def get_init_pos_velocity(self):
		"""
		This function samples position and velocity from a distribution.
		These position and velocity will be used as
		initial position and velocity for all particles.
		:return: initial position and velocity
		"""
		init_position = np.random.randn(2, self.num_particles) * self.loc_std
		init_velocity = np.random.randn(2, self.num_particles)

		# Compute magnitude of this velocity vector and format to right shape
		v_norm = np.linalg.norm(init_velocity, axis=0)

		# Scale by magnitude ?
		init_velocity = init_velocity * self.vel_norm / v_norm

		return init_position, init_velocity
	
	def get_force(self, _edges, current_positions):
		"""
		:param _edges: Adjacency matrix representing mutual causality
		:param current_positions: current coordinates of all particles
		:return: net forces acting on all particles.
		"""
		force_matrix = - self.interaction_strength * _edges
		np.fill_diagonal(force_matrix, 0)
		x_cords, y_cords = current_positions[0, :], current_positions[1, :]
		x_diffs = np.subtract.outer(x_cords, x_cords).reshape(1, self.num_particles, self.num_particles)
		y_diffs = np.subtract.outer(y_cords, y_cords).reshape(1, self.num_particles, self.num_particles)
		force_matrix = force_matrix.reshape(1, self.num_particles, self.num_particles)
		_force = (force_matrix * np.concatenate((x_diffs, y_diffs))).sum(axis=-1)
		_force[_force > self._max_F] = self._max_F
		_force[_force < -self._max_F] = -self._max_F
		return _force
	
	def generate_edges(self):
		"""
		This function generates causality graph where particles are treated as nodes.
		:return: causality graph represented as edges where particles
		"""
		# Sample nxn springs _spring_types which each holding a probability spring_prob
		_edges = np.random.choice(self._spring_types, size=(self.num_particles, self.num_particles), p=self.spring_prob)

		# Establish symmetry causal interaction
		_edges = np.tril(_edges) + np.tril(_edges, -1).T

		# Nullify self interaction or causality
		np.fill_diagonal(_edges, 0)

		return _edges
	
	def sample_trajectory(self, total_time_steps=10000, sample_freq=10):

		# Data frame columns and index for particles
		columns = [f'particle_{i}' for i in range(self.num_particles)]
		index = ['x_cordinate', 'y_cordinate']

		# Initialize causality between particles.
		_edges = self.generate_edges()

		# Initialize the first position and velocity from a distribution
		init_position, init_velocity = self.get_init_pos_velocity()
		
		# Adding initial position and velocity of particles to trajectory.
		init_position, init_velocity = self._clamp(init_position, init_velocity)
		_position = pd.DataFrame(init_position, columns=columns, index=index)
		_velocity = pd.DataFrame(init_velocity, columns=columns, index=index)
		_edge = pd.DataFrame(_edges, columns=columns, index=columns)
		self.positions.append(_position)
		self.velocities.append(_velocity)
		self.edges.append(_edge)
		
		# Compute initial forces between particles.
		init_force_between_particles = self.get_force(_edges, init_position)
		
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

		edges_counter = self.edge_counter

		for i in range(1, total_time_steps):
			
			# Compute new position based on current velocity and positions.
			new_position = current_position + (self._delta_T * velocity)
			new_position, velocity = self._clamp(new_position, velocity)
			
			# Adding new position and velocity of particles to trajectory.
			if i % sample_freq == 0:
				_position = pd.DataFrame(new_position, columns=columns, index=index)
				_velocity = pd.DataFrame(velocity, columns=columns, index=index)
				_edge = pd.DataFrame(_edges, columns=columns, index=columns)
				self.positions.append(_position)
				self.velocities.append(_velocity)
				self.edges.append(_edge)

			# If causal graph is not static, flip causal edges when counter turns zero
			if self.dynamics != 'static':
				edges_counter -= 1
				change_mask = np.where(edges_counter == 0, 1, 0)
				if np.any(change_mask):
					new_edges = np.where(_edges == 0, 1.0, 0)
					_edges = np.where(change_mask == 1, new_edges, _edges)
					new_counter = self.get_edge_counter(self.min_steps, self.max_steps+1)
					edges_counter = np.where(change_mask == 1, new_counter, edges_counter)

			# Compute forces between particles
			force_between_particles = self.get_force(_edges, new_position)
			
			# Compute new velocity based on current velocity and forces between particles.
			new_velocity = velocity + (self._delta_T * force_between_particles)
			
			# Update velocity and position
			velocity = new_velocity
			current_position = new_position
			
			# Add noise to observations
			current_position += np.random.randn(2, self.num_particles) * self.noise_var
			velocity += np.random.randn(2, self.num_particles) * self.noise_var

		# Compute energy of the system
		kinetic_energies, potential_energies, total_energies = self.get_energy()
		# construct data frame
		trajectory = {
			'positions': self.positions,
			'velocity': self.velocities,
			'edges': self.edges,
			'kinetic_energy': kinetic_energies,
			'potential_energy': potential_energies,
			'total_energy': total_energies,
		}
		return pd.DataFrame(trajectory)
	
	def get_energy(self):
		'''
		Total Energy = Kinetic Energy (K) + Potential Energy (U)
		Kinetic Energy (K) = (1/2) * m * velocity^2 : unit mass m
		Potential Energy (U) = m * g * h: h is distance, g is field, unit mass m
		:return: energy
		'''
		
		# Compute Kinetic Energy for each snap shot
		# Kinetic energy = (1/2) m * v^2, here assume a unit mass
		ek = lambda velocity: 0.5 * (velocity ** 2).sum(axis=0)
		kinetic_energies = [ek(_velocities) for _velocities in self.velocities]
		kinetic_energies = [pd.DataFrame({'kinetic_energy': ke, 'particles': self.columns}).set_index('particles') for ke in kinetic_energies]

		# Compute Potential Energy at each snap shot
		# potential energy = m * g * d, here assume a unit mass
		# g represents interaction strength and h represents distance.
		potential_energies = []
		for time_step, position in enumerate(self.positions):
			_pos = position.T.to_numpy()
			_u = []
			for particle_index in range(0, self.num_particles):
				particle_pos = position[f'particle_{particle_index}'].T.to_numpy()
				position_fill_mat = np.full(_pos.shape, particle_pos)
				distances = np.sqrt(np.square(position_fill_mat - _pos).sum(axis=1))
				pe = np.dot(self.edges[time_step][f'particle_{particle_index}'], distances ** 2)
				_u.append(0.5 * self.interaction_strength * pe)
			potential_energies.append(pd.DataFrame({'potential_energy': _u, 'particles': self.columns}).set_index('particles'))


		# Compute total energy of the system
		total_energies = []
		for time_step in range(len(potential_energies)):
			total_en = kinetic_energies[time_step]['kinetic_energy'] + potential_energies[time_step]['potential_energy']
			total_energies.append(pd.DataFrame({'total_energy': total_en, 'particles': self.columns}).set_index('particles'))
		kinetic_energies = [ken.T for ken in kinetic_energies]
		potential_energies = [pen.T for pen in potential_energies]
		total_energies = [ten.T for ten in total_energies]
		return kinetic_energies, potential_energies, total_energies
	
	def plot(self):
		"""
		This function plots position and energy over time.
		:return:
		"""
		plt.figure()
		axes = plt.gca()
		axes.set_xlim([-5., 5.])
		axes.set_ylim([-5., 5.])
		positions = [position for position in self.positions]
		positions = np.asarray(positions)
		for i in range(positions.shape[-1]):
			print(positions[:, 0, i], positions[:, 1, i])
			plt.plot(positions[:, 0, i], positions[:, 1, i])
			plt.plot(positions[0, 0, i], positions[0, 1, i], 'd')
		plt.figure()
		energies = self.get_energy()
		plt.plot(energies)
		plt.show()


	def create_gif(self):
		"""
		This function generates a gif to visualize the trajectory of the particles.
		:return:
		"""
		import os
		import glob
		from PIL import Image

		fp_in = "/Users/naveenmysore/Documents/plots/timestep_*.png"
		fp_out = "/Users/naveenmysore/Documents/plots/dyari.gif"

		positions = [position for position in self.positions]
		positions = np.asarray(positions)

		for time_step in range(0, positions.shape[0]):
			fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False, sharex=False)
			axes[0].set_title('Position')
			axes[1].set_title('Spring')

			fig.suptitle(f'DYARI- timestep {time_step}')
			entries = []
			for particle_id in range(0, positions.shape[-1]):
				data = {'particle': particle_id,
						'x_dim': positions[time_step, 0, particle_id],
						'y_dim': positions[time_step, 1, particle_id]}
				entries.append(data)
			dframe = pd.DataFrame(entries)

			pl = sns.scatterplot(data=dframe, x='x_dim', y='y_dim', hue='particle', ax=axes[0])

			plh = sns.heatmap(self.edges[time_step], vmin=0, vmax=1, ax=axes[1])

			pl.set_ylim(-5.0, 5.0)
			pl.set_xlim(-5.0, 5.0)

			plt.savefig(f"/Users/naveenmysore/Documents/plots/timestep_{time_step}.png")
			plt.clf()

		# ref: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
		img, *imgs = [Image.open(f"/Users/naveenmysore/Documents/plots/timestep_{i}.png") for i in range(0, len(self.positions))]
		img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=10, loop=0)

		# delete all png files.
		for f in glob.glob(fp_in):
			os.remove(f)


if __name__ == '__main__':
	sim = Spring(num_particles=2, dynamics='periodic', min_steps=50, max_steps=200)
	t = time.time()
	data_frame = sim.sample_trajectory(total_time_steps=1000, sample_freq=10)
	#sim.plot()
	sim.create_gif()
	print("Simulation time: {}".format(time.time() - t))
	# sim.create_gif()
