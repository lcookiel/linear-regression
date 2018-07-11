from numpy import *
import matplotlib.pyplot as plt

#y=mx+b
def sse(x, y, m, b): #Sum of Squared Errors
    sse = 0
    for i in range(len(x)):
        sse += (y[i]-m*x[i]-b) ** 2
    return sse/float(len(x))

def least_squares(x, y):
    mean_x = mean(x)
    mean_y = mean(y)
    x_and_mean_x_difference = []
    y_and_mean_y_difference = []
    m_numerator = 0
    m_denominator = 0
    for i in range(len(x)):
        m_numerator += (x[i]-mean_x) * (y[i]-mean_y)
        m_denominator += (x[i]-mean_x) ** 2
    m = m_numerator / m_denominator
    b = mean_y - m * mean_x
    return [m, b]

def run():
    points = genfromtxt('data.csv', delimiter=',')
    x = []
    y = []
    for i in points:
        x.append(i[0])
        y.append(i[1])
    plt.plot(x, y, 'ko')
    print('Starting...')
    m, b = least_squares(x, y)
    print("m = {0}, b = {1}, sum of squared errors = {2}".format(m, b, sse(x, y, m, b)))
    plt.plot([0, 120], [m*0+b, m*120+b])
    plt.axis([-10, 110, -10, 120])
    plt.show()

if __name__ == '__main__':
    run()
