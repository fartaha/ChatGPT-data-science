import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pymc3 as pm

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
with pm.Model() as iris_model:
    # Define the priors
    alpha = pm.Normal('alpha', mu=0, sigma=10, shape=4)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=(4, 4))
    sigma = pm.HalfNormal('sigma', 10)
    
    # Define the likelihood
    mu = pm.Deterministic('mu', alpha + pm.math.dot(X_train, beta))
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_train)
    
    # Run the inference
    trace = pm.sample(3000, tune=1000, chains=2)
