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
dataset = 'car.data'
n_in = 6
n_circuit = 1
n_hidden_node = [9, 8]
n_out = 4
drop_type = 3
n_epochs = 200
sparsity = 3
varied_coef = 1
learning_rate = 0.01
momentum = 0.4
batch_size = 100
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

        layer_action = ceil(output[0] * 3) - 2
        if layer_action == -2: layer_action = -1
        node_action = ceil(output[2] * 3) - 2
        if node_action == -2: node_action = -1
        circuit_action = ceil(output[5] * 3) - 2
        if circuit_action == -2: circuit_action = -1
        node_number = ceil(output[3] * max_added_node)
        layer_position = ceil(output[1] * len(n_hidden_node)) - 1
        if layer_position == -1: layer_position = 0

        if layer_action == -1:
            if len(n_hidden_node_tmp) > 1:  # decrease 1 layer
                n_hidden_node_tmp = n_hidden_node[:layer_position] + n_hidden_node[layer_position + 1:]
            else:
                genome.fitness -= 10  # "punish" if decrease a layer having less than X nodes
        elif layer_action == 1:  # increase 1 layer
            n_hidden_node_tmp.insert(layer_position, n_hidden_unit)

        node_position = ceil(output[4] * len(n_hidden_node_tmp)) - 1
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
        print('Old: {}\tNew: {} cir {}\t\t\t\tSignals: {} {} {} {} {} {}\t\tFitness: {}'
              .format(n_hidden_node, n_hidden_node_tmp, n_circuit_tmp, layer_action, layer_position, node_action,
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

    f = open('Result_car.txt', 'a')
    f.write('\n===================SET OF SIGNALS [L] [Lpo] [N] [Nnum] [Npo] [C]: {}\n'
            .format(best_decision))
    f.write('New topology: {}\tNumber of circuit: {}\n'.format(best_hidden_node, best_circuit))
    f.write('Fitness: {}\n'.format(best_fitness))
    f.close()


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

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 100)

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
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-41')
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))
    winner = p.run(eval_genomes, 10)

    # Save the model the pickle file
    joblib.dump(winner, "EANN_model.pkl")

    display(winner, config)
    visualize.draw_net(config, winner, True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


def load(config_file):
    initial_topology = [8, 10]
    initial_circuit = 2
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Load model
    winner = joblib.load("EANN_model.pkl")
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    valid_error = eval(initial_topology, initial_circuit)
    inp = tuple(valid_error[100:]) + (len(initial_topology),) + (sum(initial_topology),) + (initial_circuit,)
    output = winner_net.activate(inp)

    layer_action = ceil(output[0] * 3) - 2
    if layer_action == -2: layer_action = -1
    node_action = ceil(output[2] * 3) - 2
    if node_action == -2: node_action = -1
    circuit_action = ceil(output[5] * 3) - 2
    if circuit_action == -2: circuit_action = -1
    node_number = ceil(output[3] * max_added_node)
    layer_position = ceil(output[1] * len(n_hidden_node)) - 1
    if layer_position == -1: layer_position = 0

    if (layer_action == -1) and (len(initial_topology) > 1):
        initial_topology = initial_topology[:layer_position] + initial_topology[layer_position + 1:]

    elif layer_action == 1:  # increase 1 layer
        initial_topology.insert(layer_position, n_hidden_unit)

    node_position = ceil(output[4] * len(initial_topology)) - 1
    if node_position == -1: node_position = 0

    if node_action == -1:
        if initial_topology[node_position] - node_number > 0:  # decrease X node(s) at layer L
            initial_topology[node_position] -= node_number

    elif node_action == 1:  # increase X node(s) at layer L
        initial_topology[node_position] += node_number

    if circuit_action == -1:
        if initial_circuit > 1:
            initial_circuit -= 1
    elif circuit_action == 1:
        i = initial_circuit + 1
        while ((initial_topology[-1] % i) != 0) and (initial_topology[-1] > i):
            i += 1
        for k in initial_topology:
            if i > k:
                i = 1
                break
        initial_circuit = i

    print('\nAdjusted to: {} {}'.format(initial_topology, initial_circuit))
    eval(initial_topology, initial_circuit)

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-MLP')
    load(config_path)
    # cont(config_path)
    # eval([8,10], 4)
