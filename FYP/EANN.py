"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function

import os
import timeit

import neat
import sys

from math import ceil
from sklearn.externals import joblib

import visualize
from FYP.model import evaluate

# Parameters for target ANN.
dataset = 'breastcancer.data'
n_in = 4
n_circuit = 1
n_hidden_node = [8, 3, 4]
n_out = 3
drop_type = 3
n_epochs = 70
sparsity = 3
varied_coef = 1
learning_rate = 0.01
momentum = 0.4
batch_size = 27
probability = [0.1,
               0.2,
               0.3,
               0.4,
               0.5,
               0.6,
               0.7,
               0.8,
               0.9]

# Parameters for developmental ANN
n_hidden_unit = 2 * n_in // 3 + n_out
max_added_node = 2


def eval(hidden_unit, hidden_circuit):
    return evaluate(n_hidden_node=hidden_unit, n_circuit=hidden_circuit,
                    learning_rate=learning_rate, n_epochs=n_epochs, momentum=momentum, batch_size=batch_size,
                    dataset=dataset,
                    drop_type=drop_type, probability=probability[0],
                    sparsity=sparsity, varied_coef=varied_coef
                    )


def eval_genomes(genomes, config):
    global n_hidden_node
    global n_circuit
    valid_error = eval(n_hidden_node, n_circuit)
    standing_error = min(valid_error) * 100
    best_fitness = 0.0
    for genome_id, genome in genomes:
        n_hidden_node_tmp = list(n_hidden_node)
        n_circuit_tmp = n_circuit
        genome.fitness = 100.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        # for xi, xo in zip(datasetX, datasetY):
        #     output = net.activate(xi)
        #     genome.fitness -= (output[0] - xo[0]) ** 2
        xi = tuple(valid_error) + (len(n_hidden_node),) + (sum(n_hidden_node),) + (n_circuit,)
        output = net.activate(xi)

        # Analyze developmental signals and make changes on target ANN
        # signal_type = output.index(max(output))
        # if signal_type == 0:
        #     genome.fitness -= standing_error
        # else:
        #     if signal_type == 1:  # Increase 1 node at the first layer
        #         n_hidden_node_tmp[0] += 1
        #     elif (signal_type == 2) and (n_hidden_node_tmp[0] != 0):  # Decrease 1 node at the first layer
        #         n_hidden_node_tmp[0] -= 1
        #     elif signal_type == 3:  # Increase 1 layer to the right
        #         n_hidden_node_tmp.append(n_hidden_node_tmp[0])
        #     elif (signal_type == 4) and (len(n_hidden_node_tmp) != 1):  # Remove the last layer
        #         n_hidden_node_tmp.pop()
        #     elif signal_type == 5:  # Increase number of circuit by 1
        #         i = n_circuit_tmp
        #         while (n_hidden_node_tmp[-1] % i) != 0:
        #             i += 1
        #         n_circuit_tmp = i
        #         if n_circuit_tmp == n_hidden_node_tmp[-1]:
        #             n_circuit_tmp = 1
        #
        #     elif (signal_type == 6) and (n_circuit_tmp > 1):  # Decrease number of circuit by 1
        #         n_circuit_tmp -= 1
        #     valid_error = eval(n_hidden_node_tmp, n_circuit_tmp)
        #     genome.fitness -= min(valid_error) * 100
        #     # print('Genome fitness: {}'.format(genome.fitness))
        #     # print('Signal type: {}--------Error: {:8.5f}---------Best fitness: {}'
        #     #       .format(signal_type, min(valid_error) * 100, best_fitness))

        layer_action = ceil(output[0] * 3) - 2
        if layer_action == -2: layer_action = -1
        node_action = ceil(output[2] * 3) - 2
        if node_action == -2: node_action = -1
        circuit_action = ceil(output[5] * 3) - 2
        if circuit_action == -2: circuit_action = -1
        node_number = ceil(output[3] * max_added_node)
        if node_number == 0: node_number = 1
        layer_position = ceil(output[1] * len(n_hidden_node)) - 1
        if layer_position == -1: layer_position = 0

        if layer_action == -1:
            if len(n_hidden_node_tmp) > 1:  # decrease 1 layer
                n_hidden_node_tmp = n_hidden_node[:layer_position] + n_hidden_node[layer_position + 1:]
            else:
                genome.fitness -= 10  # "punish" if decrease a layer having less than X nodes
        elif layer_action == 1:  # increase 1 layer
            n_hidden_node_tmp.insert(layer_position, n_hidden_unit)

        node_position = ceil(output[4] * len(n_hidden_node_tmp)) - 1    # decide which layer after the layer action
        if node_position == -1: node_position = 0

        if node_action == -1:
            if n_hidden_node_tmp[node_position] - node_number > 0:  # decrease X node(s) at layer L
                n_hidden_node_tmp[node_position] -= node_number
            else:
                genome.fitness -= 20  # "punish" if this action is decided on a 1-layer NN
        elif node_action == 1:  # increase X node(s) at layer L
            n_hidden_node_tmp[node_position] += node_number


        if circuit_action == -1:
            if n_circuit_tmp > 1:
                n_circuit_tmp -= 1
            else:
                genome.fitness -= 10  # "punish" if the current number of circuit is 1
        elif circuit_action == 1:
            i = n_circuit_tmp + 1
            while ((n_hidden_node_tmp[-1] % i) != 0) and (n_hidden_node_tmp[-1] > i):
                i += 1
            for k in n_hidden_node_tmp:
                if i > k:
                    i = 1
                    break
            n_circuit_tmp = i
        valid_error = eval(n_hidden_node_tmp, n_circuit_tmp)
        genome.fitness -= min(valid_error) * 100
        print('Old: {}\tNew: {}\t\t\t\t\t Signals: {} {} {} {} {} {}\t\t\tFitness: {}'
              .format(n_hidden_node, n_hidden_node_tmp, layer_action, layer_position, node_action,
                      node_number, node_position, circuit_action, genome.fitness))
        if best_fitness < genome.fitness:
            best_decision = [layer_action, layer_position, node_action, node_number, node_position, circuit_action]
            best_fitness = genome.fitness
            best_hidden_node = list(n_hidden_node_tmp)
            best_circuit = n_circuit_tmp
    n_hidden_node = list(best_hidden_node)
    n_circuit = best_circuit
    print('\n===================SET OF SIGNALS [L] [Lpo] [N] [Nnum] [Npo] [C]: {}'
          .format(best_decision))
    print('New topology: {}\tNumber of circuit: {}'.format(best_hidden_node, best_circuit))
    f = open('Result_breastcancer.txt', 'a')
    f.write('\n===================SET OF SIGNALS [L] [Lpo] [N] [Nnum] [Npo] [C]: {}\n'
            .format(best_decision))
    f.write('New topology: {}\tNumber of circuit: {}\n'.format(best_hidden_node, best_circuit))
    f.write('Fitness: {}\n'.format(best_fitness))


def display(winner, config):
    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    # print('\nOutput:')
    # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    # for xi, xo in zip(datasetX, datasetY):
    #     output = winner_net.activate(xi)
    #     print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))


def run(config_file):
    start_time = timeit.default_timer()
    open('Result_breastcancer.txt', 'w')
    print('Initial topology: {}\tNumber of circuit: {}'.format(n_hidden_node, n_circuit))
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
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to x generations.
    winner = p.run(eval_genomes, 1000)

    # Save the model the pickle file
    joblib.dump(winner, "EANN_model.pkl")

    # display(winner, config)
    # node_names = {0: 'EANN'}
    visualize.draw_net(config, winner, True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    end_time = timeit.default_timer()
    print('The code for file EANN.py ran for {:8.1f}s'.format(end_time - start_time), file=sys.stderr)


def cont(config_file):
    start_time = timeit.default_timer()
    # open('Result_breastcancer.txt', 'w')
    print('Initial topology: {}\tNumber of circuit: {}'.format(n_hidden_node, n_circuit))
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-59')
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))
    winner = p.run(eval_genomes, 10)

    # Save the model the pickle file
    joblib.dump(winner, "EANN_model.pkl")

    visualize.draw_net(config, winner, True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    end_time = timeit.default_timer()
    print('The code for file EANN.py ran for {:8.1f}hour(s)'.format((end_time - start_time)/3600), file=sys.stderr)

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
    # for xi, xo in zip(datasetX, datasetY):
    #     output = winner_net.activate(xi)
    #     print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
    #     error += (xo[0] - output[0]) ** 2

    print("\nRoot mean square error: %f" % error)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-MLP')
    cont(config_path)
    # cont(config_path)
    # eval([8,10], 4)
