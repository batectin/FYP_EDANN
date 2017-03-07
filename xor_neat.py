"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function
from sklearn.externals import joblib
import os
import neat
import numpy
import visualize

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


def load_data():
    #############
    # LOAD DATA #
    #############

    print('... loading data')

    # Load the dataset
    f = numpy.loadtxt("breastcancer.data", delimiter=",", dtype=None)

    numpy.random.shuffle(f)
    dataset_input = f[0:20, 3:12]
    dataset_input = list(tuple(map(tuple, dataset_input)))

    dataset_output = f[0:20, 2:3]
    for i in dataset_output:
        i[0] = 0 if i[0] == 2 else 1
    dataset_output = list(tuple(map(tuple, dataset_output)))

    rval = [dataset_input, dataset_output]
    return rval


# Load dataset
datasetX = load_data()[0]
datasetY = load_data()[1]


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 30.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(datasetX, datasetY):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2

def display(winner, config):
    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    for xi, xo in zip(datasetX, datasetY):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(50))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 1000)

    # Save the model the pickle file
    joblib.dump(winner, "neat_model.pkl")

    display(winner, config)
    node_names = {0: 'Breast Cancer'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


    #
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)
def cont(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-1899')
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(100))
    winner = p.run(eval_genomes, 500)

    display(winner, config)
    node_names = {0: 'Breast Cancer'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

def load(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Load model
    winner = joblib.load("neat_model.pkl")

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    error = 0
    for xi, xo in zip(datasetX, datasetY):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
        error += (xo[0] - output[0]) ** 2

    print("\nRoot mean square error: %f" % error)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    load(config_path)
    # cont(config_path)
