import neuralnet as nn


def read(file_name):
    f = open(file_name, 'r')
    lines = f.readlines()
    depth = int(lines[0].replace('\n', ''))
    topology = []
    for i in range(1, depth + 1):
        topology.append(int(lines[i].replace('\n', '')))
    net = nn.NeuralNet(topology)
    skip = 0
    for l in range(depth - 1):
        for i in range(topology[l] + 1):
            line = lines[depth + 1 + i + skip].replace('\n', '')
            weights = line.split(' ')
            for j in range(topology[l + 1]):
                net.theta[l][i, j] = float(weights[j])
        skip += topology[l] + 1
    f.close()
    return net
