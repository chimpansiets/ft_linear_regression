from math import sqrt
import matplotlib.pyplot as plt

def mean(values):
	return (sum(values) / float(len(values)))

def variance(values, mean):
	return sum([(x - mean)**2 for x in values])

def covariance(x, mean_x, y, mean_y):
	covar = 0.0
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar

def coefficients(dataset):
	x = [row[0] for row in dataset]
	y = [row[1] for row in dataset]
	x_mean, y_mean = mean(x), mean(y)
	b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
	b0 = y_mean - b1 * x_mean
	return [b0, b1]

def simple_linear_regression(train, test):
	predictions = list()
	b0, b1 = coefficients(train)
	print("Theta0: {}, Theta1: {}".format(b0, b1))
	for row in test:
		yhat = b0 + b1 * row[0]
		predictions.append(yhat)
	return predictions

def rmse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)

def evaluate_algorithm(dataset, algorithm):
	test_set = list()
	for row in dataset:
		row_copy = list(row)
		row_copy[-1] = None
		test_set.append(row_copy)
	predicted = algorithm(dataset, test_set)
	# print(predicted)
	actual = [row[-1] for row in dataset]
	rmse = rmse_metric(actual, predicted)
	return rmse

if __name__ == "__main__":
	f = open("data.csv", "r")

	data = []
	for line in f.readlines():
		if line != 'km,price\n':
			data.append(list(map(int, line.split(','))))
	rmse = evaluate_algorithm(data, simple_linear_regression)
	print('RMSE: %.3f' % rmse)

	# Plot part
	line = [8500 + (-0.02 * x) for x in range(10000, 250000, 1000)]
	plt.scatter([x[0] for x in data], [x[1] for x in data])
	plt.ylabel('Price')
	plt.xlabel('Mileage')
	plt.plot(range(10000, 250000, 1000), line, color='red')
	plt.show()