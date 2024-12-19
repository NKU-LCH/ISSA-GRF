import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from geopy.distance import geodesic
from sklearn.metrics import pairwise_distances
import joblib  # For model saving and loading
from scipy.stats import gaussian_kde

# Define the Geographical Random Forest model
class GeographicalRandomForest(BaseEstimator, RegressorMixin):
    def __init__(self, n_neighbors=25, n_estimators=100, random_state=42):
        self.n_neighbors = n_neighbors
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = {}

    def fit(self, X, y):
        self.X_train_ = X.copy()  # Store training data for use in predict
        # Compute geographic distance matrix using geodesic distance
        geodist_matrix = pairwise_distances(
            X[['lat', 'lon']], 
            metric=lambda u, v: geodesic(u, v).km
        )
        for i in range(len(X)):
            # Find indices of the nearest neighbors (excluding the point itself)
            neighbors_idx = np.argsort(geodist_matrix[i])[1:self.n_neighbors + 1]
            X_neighbors = X.iloc[neighbors_idx].drop(['lat', 'lon'], axis=1)
            y_neighbors = y.iloc[neighbors_idx]
            # Initialize and train the Random Forest model for the current region
            model = RandomForestRegressor(
                n_estimators=self.n_estimators, 
                random_state=self.random_state
            )
            model.fit(X_neighbors, y_neighbors)
            self.models[i] = model
        return self

    def predict(self, X):
        predictions = np.zeros(len(X))
        # Compute geographic distance between test points and training points
        geodist_matrix = pairwise_distances(
            X[['lat', 'lon']], 
            self.X_train_[['lat', 'lon']], 
            metric=lambda u, v: geodesic(u, v).km
        )
        for i in range(len(X)):
            # Find the nearest training index
            nearest_train_idx = np.argmin(geodist_matrix[i])
            model = self.models.get(nearest_train_idx, None)
            if model is not None:
                # Predict using the corresponding regional model
                predictions[i] = model.predict(
                    X.iloc[[i]].drop(['lat', 'lon'], axis=1)
                )
            else:
                # Fallback to a global model if regional model is not found
                predictions[i] = np.nan
        return predictions

# Define the fitness function for optimization (Lower MSE is better)
def fitness_function(params, X_train, y_train, X_test, y_test):
    n_neighbors = max(1, int(params[0]))
    n_estimators = max(10, int(params[1]))
    # Initialize and train the GRF model with given hyperparameters
    model = GeographicalRandomForest(
        n_neighbors=n_neighbors, 
        n_estimators=n_estimators
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Calculate Mean Squared Error as the fitness score
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Define the Levy flight step size generator using Mantegna's algorithm
def levy_flight(Lambda):
    sigma_u = (
        (np.math.gamma(1 + Lambda) * 
         np.sin(np.pi * Lambda / 2)) / 
        (np.math.gamma((1 + Lambda) / 2) * Lambda * 
         2 ** ((Lambda - 1) / 2))
    ) ** (1 / Lambda)
    u = np.random.normal(0, sigma_u)
    v = np.random.normal(0, 1)
    step = u / (np.abs(v) ** (1 / Lambda))
    return step

# Define the Opposite-Based Learning strategy
def generate_opposite(solution, lb, ub):
    return lb + ub - solution

# Improved Sparrow Search Algorithm (ISSA) with Opposite-Based Learning and Levy Flight
def ISSA(
    pop_size, 
    max_iter, 
    lb, 
    ub, 
    dim, 
    X_train, 
    y_train, 
    X_test, 
    y_test, 
    Lambda=1.5,
    OBL_rate=0.5
):
    lb = np.array(lb)
    ub = np.array(ub)
    # Initialize population within bounds
    X = np.random.rand(pop_size, dim) * (ub - lb) + lb
    # Evaluate initial fitness
    fitness = np.array([
        fitness_function(ind, X_train, y_train, X_test, y_test) for ind in X
    ])
    # Initialize personal bests
    pFit = fitness.copy()
    pX = X.copy()
    # Identify the global best
    best_idx = np.argmin(fitness)
    bestX = X[best_idx].copy()
    best_fit = fitness[best_idx]

    for t in range(max_iter):
        print(f'Iteration {t+1}/{max_iter}: Best Fitness = {best_fit:.4f}')
        for i in range(pop_size):
            # Opposite-Based Learning with a probability defined by OBL_rate
            if np.random.rand() < OBL_rate:
                opposite = generate_opposite(X[i], lb, ub)
                opposite_fit = fitness_function(
                    opposite, X_train, y_train, X_test, y_test
                )
                # Select the better between current solution and its opposite
                if opposite_fit < fitness[i]:
                    X[i] = opposite
                    fitness[i] = opposite_fit

            # Decide whether to use Levy flight or not
            if np.random.rand() < 0.8:
                # Exploration phase: Levy flight
                step_size = levy_flight(Lambda)
                # Update position based on Levy flight
                X[i] = bestX + step_size * (X[i] - bestX)
            else:
                # Exploitation phase: Adjust towards personal best
                X[i] = pX[i] + np.random.rand(dim) * (bestX - X[i])

            # Ensure the new position is within bounds
            X[i] = np.clip(X[i], lb, ub)
            # Evaluate fitness of the new position
            current_fit = fitness_function(
                X[i], X_train, y_train, X_test, y_test
            )
            # Update personal best if improvement is found
            if current_fit < fitness[i]:
                fitness[i] = current_fit
                pX[i] = X[i].copy()
                # Update global best if necessary
                if current_fit < best_fit:
                    best_fit = current_fit
                    bestX = X[i].copy()

        # Optionally, implement convergence criteria or additional logging here

    return bestX, best_fit

# Main function to execute the workflow
def main():
    # Load the dataset (Ensure the path is correct)
    data = pd.read_excel('demo.xlsx')
    # Define features and target
    X = data.drop(['MPB'], axis=1)
    y = data['MPB']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define ISSA parameters
    SearchAgents_no = 20       # Increased population size for better exploration
    Max_iteration = 50         # Increased iterations for better convergence
    dim = 2                    # Number of hyperparameters to optimize
    lb = [5, 10]               # Lower bounds for n_neighbors and n_estimators
    ub = [50, 200]             # Upper bounds for n_neighbors and n_estimators

    # Perform ISSA optimization to find the best hyperparameters
    best_params, best_fitness = ISSA(
        pop_size=SearchAgents_no,
        max_iter=Max_iteration,
        lb=lb,
        ub=ub,
        dim=dim,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        Lambda=1.5,
        OBL_rate=0.5  # Probability of applying Opposite-Based Learning
    )
    # Extract optimized hyperparameters
    n_neighbors, n_estimators = int(best_params[0]), int(best_params[1])
    print(f'Optimized Hyperparameters: n_neighbors={n_neighbors}, n_estimators={n_estimators}')

    # Train the final GRF model with optimized hyperparameters
    final_model = GeographicalRandomForest(
        n_neighbors=n_neighbors, 
        n_estimators=n_estimators
    )
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    # Visualization using Gaussian KDE for point density
    xy = np.vstack([y_test, y_pred])
    z = gaussian_kde(xy)(xy)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=y_test, 
        y=y_pred, 
        hue=z, 
        palette='viridis', 
        alpha=0.7, 
        s=50
    )
    plt.plot(
        [y_test.min(), y_test.max()], 
        [y_test.min(), y_test.max()], 
        'k--', lw=2
    )
    plt.xlabel('Actual MP Concentration', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted MP Concentration', fontsize=14, fontweight='bold')
    plt.title('Actual vs Predicted MP Concentration', fontsize=16, fontweight='bold')
    plt.legend(title='Density', loc='upper left', fontsize=12)
    plt.show()

    # Calculate evaluation metrics
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R²: {r2:.4f}')

    # Detailed Scatter Plot with Color Mapping based on Error
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        y_test, 
        y_pred, 
        c=np.abs(y_pred - y_test), 
        cmap='viridis', 
        alpha=0.7
    )
    ax.plot(
        [y_test.min(), y_test.max()], 
        [y_test.min(), y_test.max()], 
        'k--', lw=2
    )
    ax.set_xlabel('Actual MP Concentration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted MP Concentration', fontsize=14, fontweight='bold')
    ax.set_title('Actual vs Predicted MP Concentration with Error Mapping', fontsize=16, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Absolute Error', fontsize=12)
    plt.show()

    # Save the optimized model to disk
    joblib.dump(
        final_model, 
        'issa_grf.pkl'
    )
    print('Optimized GRF model saved successfully.')

if __name__ == "__main__":
    main()
