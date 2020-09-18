
if __name__ == "__main__":
	print("Please enter a mileage for your car: ", end='')
	mileage = int(input())
	theta0, theta1 = 8500, -0.02144
	print("The predicted price for your car would be: ", float(theta0 + (theta1 * mileage)))