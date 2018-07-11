from numpy import *
import matplotlib.pyplot as plt

#y=mx+b
def sse(x, y, m, b): #Sum of Squared Errors
    sse = 0
    for i in range(len(x)):
        sse += (y[i]-m*x[i]-b) ** 2
    return sse/float(len(x))

def gradient_step(m, b, x, y, learning_rate):
    m_gradient = 0
    b_gradient = 0
    N = float(len(x))
    for i in range(0, len(x)):
        #partial derrivatives
        m_gradient += -2/N * x[i] * (y[i] - m * x[i] - b)
        b_gradient += -2/N * (y[i] - m * x[i] - b)
    new_m = m - (learning_rate * m_gradient)
    new_b = b - (learning_rate * b_gradient)
    return [new_m, new_b]

def gradient_descent(init_b, init_m, x, y, learning_rate, n_iterations):
    m = init_m
    b = init_b
    N = float(len(x))
    for i in range(n_iterations):
        m, b = gradient_step(m, b, x, y, learning_rate)
    return [m, b]

def run():
    #hyperparameters
    init_m = 0
    init_b = 0
    learning_rate = 0.0001
    n_iterations = 1000
    points = genfromtxt('data.csv', delimiter=',')
    x = []
    y = []
    for i in points:
        x.append(i[0])
        y.append(i[1])
    print('Initial values: \nm={0}, b={1}, \nlearning rate = {2}, \nn of iterations = {3}, \nsum of squared errors = {4}'.format(init_m, init_b, learning_rate, n_iterations, sse(x, y, init_m, init_b)))
    print('Starting...')
    m, b = gradient_descent(init_b, init_m, x, y, learning_rate, n_iterations)
    print("After {0} iterations m = {1}, b = {2}, sum of squared errors = {3}".format(n_iterations, m, b, sse(x, y, m, b)))
    plt.plot(x, y, 'ko')
    plt.plot([0, 120], [m*0+b, m*120+b])
    plt.axis([-10, 110, -10, 120])
    plt.show()

if __name__ == '__main__':
    run()
