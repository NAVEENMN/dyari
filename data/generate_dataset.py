import time
import argparse
import numpy as np
import pandas as pd
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


def generate_data_schema():
    schema = {
        'label': ['positions', 'velocity', 'edges'],
        'descriptions': ['postions of all particles in x and y cordinates',
                         'velocity of all particles in x and y cordinates',
                         'Causal relationship between particles'],
        'dimensions': [('2', 'num_of_particles'), ('2', 'num_of_particles'), ('num_of_particles', 'num_of_particles')]
    }

    df = pd.DataFrame(schema).set_index('label')


def generate_data(args, dynamics):

    print(f"Generating {args.num_train} {dynamics} simulations")
    trajectories = []
    particles = []
    dynamics = []

    for i in range(args.num_train):
        t = time.time()
        sim = Spring(num_particles=5)
        data_frame = sim.sample_trajectory(total_time_steps=args.length,
                                           sample_freq=args.sample_freq)
        trajectories.append(data_frame)
        particles.append(5)
        dynamics.append('periodic')
        print(f"Simulation {i}, time: {time.time() - t}")

    data = {
        'trajectories': trajectories,
        'particles': particles,
        'dynamics': dynamics,
        'simulation_id': [f'simulation_{i}' for i in range(args.num_train)]
    }

    df = pd.DataFrame(data).set_index('simulation_id')
    df.to_pickle('samples/dyari.pkl')
    print(f"Simulations saved to samples/dyari.csv")


def main():
    args = parser.parse_args()
    generate_data(args=args, dynamics='periodic')


if __name__ == "__main__":
    main()
