import time
import argparse
import numpy as np
from synthetic_sim import Spring

parser = argparse.ArgumentParser()

parser.add_argument('--num-train', type=int, default=50,
                    help='Number of training simulations to generate.')
parser.add_argument('--num-test', type=int, default=10000,
                    help='Number of test simulations to generate.')
parser.add_argument('--num-valid', type=int, default=10000,
                    help='Number of validation simulations to generate.')

parser.add_argument('--length', type=int, default=5000,
                    help='Length of trajectory.')
parser.add_argument('--sample-freq', type=int, default=100,
                    help='How often to sample the trajectory.')
parser.add_argument('--n-balls', type=int, default=5,
                    help='Number of balls in the simulation.')


def generate_data(args, sim, mode):
    print(f"Generating {args.num_train} {mode} simulations")
    suffix = f'_springs_{str(args.n_balls)}'
    save_path_position = f'samples/loc_{mode}_{suffix}.npy'
    save_path_velocity = f'samples/vel_{mode}_{suffix}.npy'
    save_path_edges = f'samples/edges_{mode}_{suffix}.npy'

    all_positions = []
    all_velocities = []
    all_edges = []

    for i in range(args.num_train):
        positions, velocities, edges = sim.sample_trajectory(total_time_steps=args.length,
                                                             sample_freq=args.sample_freq)
        all_positions.append(positions)
        all_velocities.append(velocities)
        all_edges.append(edges)

    np.save(save_path_position, all_positions)
    np.save(save_path_velocity, all_velocities)
    np.save(save_path_edges, all_edges)


def main():
    args = parser.parse_args()
    sim = Spring(num_particles=5)

    generate_data(args, sim, 'train')
    generate_data(args, sim, 'test')
    generate_data(args, sim, 'val')


if __name__ == "__main__":
    main()
