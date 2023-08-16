import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('housing.csv')
data.dropna(inplace=True)

data = data.drop(['longitude','latitude','ocean_proximity'],axis = 1)

train_data = data.columns = [' housing_median_age','total_rooms','total_bedrooms','population','households','median_income','median_house_value']

def normalize(x,mean=np.zeros(1),std=np.zeros(1)):
    x = np.array(x)
    if len(mean.shape) == 1 or len(std.shape) == 1:
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0, ddof=1)

    x = (x - mean)/std
    return x, mean, std

x_norm , mu , sigma = normalize(data[[' housing_median_age','total_rooms','total_bedrooms','population','households','median_income']])

data[' housing_median_age'] = x_norm[:,0]
data['total_rooms'] = x_norm[:,1]
data['total_bedrooms'] = x_norm[:,2]
data['population'] = x_norm[:,3]
data['households'] = x_norm[:,4]
data['median_income'] = x_norm[:,5]

def cost(x,y,theta):
    m = y.shape[0]
    h = x.dot(theta)
    J = (1/(2*m))*(np.sum((h - y)**2))
    return J

def gradient_descent(x, y, theta, alpha, num_iters):
    m = y.shape[0]
    J_history = np.zeros(shape=(num_iters, 1))

    for i in range(0, num_iters):
        h = x.dot(theta)
        diff_hy = h - y

        delta = (1/m) * (diff_hy.T.dot(x))
        theta = theta - (alpha * delta.T)
        J_history[i] = cost(x, y, theta)

    return theta, J_history

m = data.shape[0]
x = np.hstack((np.ones((m, 1)), x_norm))
y = np.array(data.median_house_value).reshape(-1,1)
theta = np.zeros(shape=(x.shape[1],1))

alpha = [0.3, 0.1, 0.03, 0.01]
colors = ['b','r','g','c']
num_iters = 50

for i in range(0, len(alpha)):
    theta = np.zeros(shape=(x.shape[1],1))
    theta, J_history = gradient_descent(x, y, theta, alpha[i], num_iters)
    plt.plot(range(len(J_history)), J_history, colors[i], label='Alpha {}'.format(alpha[i]))
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.title('Selecting learning rates')
plt.legend()
plt.show()

iterations = 2500
alpha = 0.1
theta, _ = gradient_descent(x, y, theta, alpha, iterations)

age = float(input('How old you want the house: '))
rooms = float(input('How many rooms you want: '))
bed_rooms = float(input('How many bedrooms you want: '))
population = float(input('How crowded you want the location: '))
households = float(input('How many households do you want: '))
income = float(input('How much is your income: '))

age = (age-mu[0])/sigma[0]
rooms = (rooms-mu[1])/sigma[1]
bed_rooms = (bed_rooms-mu[2])/sigma[2]
population = (population-mu[4])/sigma[4]
households = (households-mu[4])/sigma[4]
income = (income-mu[5])/sigma[5]


input_data = np.array([1, age, rooms, bed_rooms, population, households, income])
Selling_price = input_data.dot(theta)

print('The amount you have to pay is:', Selling_price)
