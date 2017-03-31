"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function

import os
import random
import timeit

import neat
import sys

from math import ceil
from sklearn.externals import joblib

import visualize
from FYP.model import evaluate

# Parameters for target ANN.
dataset = ['winequality.data', 'breastcancer.data', 'iris.data', 'car.data', 'g_credit.csv',
           'balance-scale.csv', 'segment.csv', 'diabetes.csv']
dataset_inout = [(11, 6), (9, 2), (4, 3), (6, 4), (24, 2), (4, 3), (19, 7), (8, 2)]
batch_size = [40, 10, 10, 30, 10, 40, 10, 10]
n_in = 9
# n_circuit = 1
# n_hidden_node = []
n_out = 2
drop_type = 3
n_epochs = 100
sparsity = 3
varied_coef = 1
learning_rate = [0.01, 0.01, 0.01, 0.01, 0.005, 0.01, 0.01, 0.01]
momentum = 0.1
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
n_hidden_unit = 2 * (n_in + n_out) // 3
max_added_node = 3
n_stretch = 1
n_target_ANN = 6
threshold_layer = 3
threshold_node_upper = 15
threshold_node_lower = 3
threshold_circuit = 2

def eval(hidden_unit, hidden_circuit, index):
    n_in, n_out = dataset_inout[index]
    return evaluate(n_hidden_node=hidden_unit, n_circuit=hidden_circuit,
                    learning_rate=learning_rate[index], n_epochs=n_epochs, momentum=momentum,
                    batch_size=batch_size[index],
                    dataset='Dataset/'+dataset[index],
                    drop_type=drop_type, probability=probability[0],
                    sparsity=sparsity, varied_coef=varied_coef,
                    n_in=n_in, n_out=n_out)

n_hidden_node = []
n_circuit = []
valid_errors = []
index = []
for target_index in range(n_target_ANN):
    n_hidden_node.append([])
    n_layer = random.randint(1, threshold_layer)
    for layer in range(n_layer):
        n_hidden_node[target_index].append(random.randint(threshold_node_lower, threshold_node_upper))
    n_circuit.append(random.randint(1, threshold_circuit))
    index.append(random.randint(0, len(dataset) - 1))
    valid_errors.append(eval(n_hidden_node[target_index], n_circuit[target_index], index[target_index]))

n_hidden_node.append([])
n_circuit.append(0)
index.append(0)
valid_errors.append([])


def eval_genomes(genomes, config):
    global n_hidden_node
    global n_circuit

    # Init variables and dependencies
    best_fitness = -100.0
    target_index = n_target_ANN - 1
    n_layer = random.randint(1, threshold_layer)
    n_hidden_node[target_index] = []
    for layer in range(n_layer):
        n_hidden_node[target_index].append(random.randint(threshold_node_lower, threshold_node_upper))
    n_circuit[target_index] = random.randint(1, threshold_circuit)
    index[target_index] = random.randint(0, len(dataset) - 1)
    valid_errors[target_index] = eval(n_hidden_node[target_index], n_circuit[target_index], index[target_index])

    for genome_id, genome in genomes:
        average_error = 0
        genome.fitness = 0.0
        for target_index in range(n_target_ANN):
            best_stretch_error = 100
            n_hidden_node_tmp = list(n_hidden_node[target_index])
            valid_error = list(valid_errors[target_index])
            n_circuit_tmp = n_circuit[target_index]
            if n_circuit_tmp > min(n_hidden_node_tmp):
                n_circuit_tmp = min(n_hidden_node_tmp)

            net = neat.nn.FeedForwardNetwork.create(genome, config)
            xi = tuple(valid_error) + (len(n_hidden_node_tmp),) + (sum(n_hidden_node_tmp),) + (n_circuit_tmp,)
            output = net.activate(xi)

            # Analyze developmental signals and make changes on target ANN

            layer_action = ceil(output[0] * 3) - 2
            if layer_action == -2: layer_action = -1
            node_action = ceil(output[2] * 3) - 2
            if node_action == -2: node_action = -1
            circuit_action = ceil(output[5] * 3) - 2
            if circuit_action == -2: circuit_action = -1
            node_number = ceil(output[3] * max_added_node)
            if node_number == 0: node_number = 1
            layer_position = ceil(output[1] * len(n_hidden_node_tmp)) - 1
            if layer_position == -1: layer_position = 0

            if layer_action == -1:
                if len(n_hidden_node_tmp) > 1:  # decrease 1 layer
                    n_hidden_node_tmp = n_hidden_node[target_index][:layer_position] + \
                                        n_hidden_node[target_index][layer_position + 1:]
                    # else:
                    #     genome.fitness -= 10  # "punish" if decrease a layer having less than X nodes
            elif layer_action == 1:  # increase 1 layer
                n_hidden_node_tmp.insert(layer_position, n_hidden_unit)

            node_position = ceil(output[4] * len(n_hidden_node_tmp)) - 1  # decide which layer after the layer action
            if node_position == -1: node_position = 0

            if node_action == -1:
                if (n_hidden_node_tmp[node_position] - node_number) > 0:  # decrease X node(s) at layer L
                    n_hidden_node_tmp[node_position] -= node_number
                    # else:
                    #     genome.fitness -= 20  # "punish" if this action is decided on a 1-layer NN
            elif node_action == 1:  # increase X node(s) at layer L
                n_hidden_node_tmp[node_position] += node_number

            if circuit_action == -1:
                if n_circuit_tmp > 1:
                    n_circuit_tmp -= 1
                    # else:
                    #     genome.fitness -= 10  # "punish" if the current number of circuit is 1
            elif circuit_action == 1:
                if n_circuit_tmp + 1 <= n_hidden_node_tmp[-1]: n_circuit_tmp += 1
                if n_circuit_tmp > min(n_hidden_node_tmp): n_circuit_tmp = min(n_hidden_node_tmp)

            stretch_error = 0
            for i in range(n_stretch):
                stretch_error += min(eval(n_hidden_node_tmp, n_circuit_tmp, index[target_index]))
            stretch_error /= n_stretch
            average_error += (stretch_error - min(valid_error))
            if best_stretch_error > stretch_error:
                decision = [layer_action, layer_position, node_action, node_number, node_position, circuit_action]
            print('Old: {}\tNew: {}\tCircuit: {}\t\t\t\t\t Signals: {} {} {} {} {} {}\t\t\tDelta error: {}\tDataset: {}'
                  .format(n_hidden_node[target_index], n_hidden_node_tmp, n_circuit_tmp, layer_action, layer_position,
                          node_action,
                          node_number, node_position, circuit_action, (stretch_error - min(valid_error)) * 100,
                          dataset[index[target_index]]))

        genome.fitness += (average_error * 100 / n_target_ANN)
        print('Fitness: {}\n'.format(genome.fitness))
        if best_fitness < genome.fitness:
            best_decision = decision
            best_fitness = genome.fitness
            best_hidden_node = list(n_hidden_node_tmp)
            best_circuit = n_circuit_tmp
        genome.decision = decision

    # Lemarkism
    # n_hidden_node = list(best_hidden_node)
    # n_circuit = best_circuit
    print('\n===================SET OF SIGNALS [L] [Lpo] [N] [Nnum] [Npo] [C]: {}'
          .format(best_decision))
    f = open('Result_breastcancer.txt', 'a')
    f.write('{} {} {} {}\n'.format(len(best_hidden_node), sum(best_hidden_node), best_circuit, best_fitness))

    g = open('Result_for_visualize.txt', 'a')
    g.write('{}\n'.format(best_decision))


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
    winner = p.run(eval_genomes, 200)

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
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-248')
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))
    winner = p.run(eval_genomes, 50)

    # Save the model the pickle file
    joblib.dump(winner, "EANN_model.pkl")
    visualize.draw_net(config, winner, True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    end_time = timeit.default_timer()
    print('The code for file EANN.py ran for {:8.1f}hour(s)'.format((end_time - start_time) / 3600), file=sys.stderr)


def load(config_file):
    n_hidden_node_tmp = [9, 9, 9]
    n_circuit_tmp = 1

    print('Initial topology: {} {}'.format(n_hidden_node_tmp, n_circuit_tmp))
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Load model
    winner = joblib.load("EANN_model.pkl")

    error = eval(n_hidden_node_tmp, n_circuit_tmp)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    inp = tuple(error) + (len(n_hidden_node_tmp),) + (sum(n_hidden_node_tmp),) + (n_circuit_tmp,)
    output = winner_net.activate(inp)

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
    elif layer_action == 1:  # increase 1 layer
        n_hidden_node_tmp.insert(layer_position, n_hidden_unit)

    node_position = ceil(output[4] * len(n_hidden_node_tmp)) - 1  # decide which layer after the layer action
    if node_position == -1: node_position = 0

    if node_action == -1:
        if n_hidden_node_tmp[node_position] - node_number > 0:  # decrease X node(s) at layer L
            n_hidden_node_tmp[node_position] -= node_number
    elif node_action == 1:  # increase X node(s) at layer L
        n_hidden_node_tmp[node_position] += node_number

    if circuit_action == -1:
        if n_circuit_tmp > 1:
            n_circuit_tmp -= 1
    elif circuit_action == 1:
        if n_circuit_tmp + 1 <= n_hidden_node_tmp[-1]: n_circuit_tmp += 1

    print('\nAdjusted to: {} {}'.format(n_hidden_node_tmp, n_circuit_tmp))
    eval(n_hidden_node_tmp, n_circuit_tmp)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-MLP')
    cont(config_path)
    # valid1 = 0
    # valid2 = 0
    # for i in range(10):
    #     tmp = min(eval([8,8], 2)[0])
    #     tmp2 = min(eval([1,1,1,1,1,1], 1)[0])
    #     print(tmp*100, ' ', tmp2*100)
    #     valid1 += tmp
    #     valid2 += tmp2
    # valid1 /= 10
    # valid2 /= 10
    # print('Result: {}% {}%\nDifference: {}%'.format(valid1*100, valid2*100, abs(valid1-valid2)*100))
    # eval([8, 10], 1, 0)
