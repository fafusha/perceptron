import numpy as np
import matplotlib.pyplot as plt


def training_matrix(n, lbound, ubound):
    '''Gerates a training matrix for a two input function

    Parameters:
        n: number if entries
        lbound: lower bound of a matrix entry
        ubound: upper bound of a matrix entry

    Returns:
        out: generated training matrix

    '''
    out = np.zeros((n,3))

    for i in range(n):
        out[i, 0] = 1
        out[i, 1] = np.random.randint(low = lbound, high = ubound)
        out[i, 2] = np.random.randint(low = lbound, high = ubound)

    return out


def predict(x, w):
    '''Perceptron function

    Parameters:
        x: perceptron input
        w: coefficient vector

    Returns:
        1/0

    '''

    # summation function
    sum = (w.T).dot(x)

    # activation function
    if sum > 0:
        return 1
    else:
        return 0


def target(x):
    '''Target function for >

    Parameters:
        x: perceptron input

    Returns:
        1/0

    '''
    return x[1] > x[2]


def train(examples):
    '''Perceptron training function

    Parameters:
        examples:training matrix

    Returns:
        w: coefficient vector
    '''
    w = np.array([0,0,0])
    perfect = False
    while not perfect:
        perfect = True
        for e in examples:
            if predict(e, w) != target(e):
                perfect = False
                if predict(e, w) == 0:
                    w = w + e
                elif predict(e, w) == 1:
                    w = w - e
    return w




def analyze(low, high, lbound, ubound, numtest, numtrain):
    """Calculates sample accuracy

    Parameters:
        low: min number of training samples
        high: max number of training examples
        lbound: lower number bound
        ubound: upper number bound
        numtest: number of tests
        numtrain: number of trainings

    Returns:
        outx1: x-coords
        outy1: Individiual sample accuracy
        outx2: x-cords
        outy2: Average sample accuracy

    """
    outy1 = []
    outy2 = []

    outx1 = []
    for x in range(low, high + 1):
        outx1.extend([x]*numtrain)

    outx2 = [x for x in range(low, high + 1)]

    test = training_matrix(numtest, lbound, ubound)

    for n in range(low, high + 1):
        avgtotal = 0

        for i in range(numtrain):

            tmat = training_matrix(high, lbound, ubound) # training matrix
            examples = tmat[:n]
            w = train(examples)
            avgtemp = 0

            for e in test:
                if predict(e, w) == target(e):
                    avgtemp += 1

            avgtemp = avgtemp / numtest * 100
            avgtotal = avgtotal + avgtemp
            outy1.append(avgtemp)

        avgtotal = avgtotal / numtrain
        outy2.append(avgtotal)

    return outx1, outy1, outx2, outy2


x1, y1, x2, y2 = analyze(1, 100, -100, 100, 250, 10)

plt.plot(x2, y2, label = 'Average sample accuracy', c = 'cyan')
plt.scatter(x1, y1, label = 'Individual sample accuracy', s = 5)


plt.xlabel('Number of training samples')
plt.ylabel('Prediction accuracy, %')
plt.title('Relation between number of training samples and\n'+
          'prediction accuracy for perceptron training algorithm\n'+
          ' for > function. ')
plt.legend()
plt.show()

print(x1, '\n', y1, '\n', x2, '\n', y2)
