import numpy as np
import matplotlib.pyplot as plt


class SOM:
    def __init__(self, grid_x, grid_y, input_dim, learning_rate=0.5, radius=1.0, decay=0.99):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.radius = radius
        self.decay = decay
        self.weights = np.random.rand(grid_x, grid_y, input_dim)

    def _get_bmu(self, x):
        """
        Find the Best Matching Unit (BMU) for input vector x
        """
        bmu_idx = None
        min_dist = np.inf
        for i in range(self.grid_x):
            for j in range(self.grid_y):
                w = self.weights[i, j]
                dist = np.linalg.norm(x - w)
                if dist < min_dist:
                    min_dist = dist
                    bmu_idx = (i, j)
        return bmu_idx

    def _update_weights(self, x, bmu_idx, iteration, max_iterations):
        """
        Update the weights of the SOM grid
        """
        lr = self.learning_rate * (1 - iteration / max_iterations)  # Decaying learning rate
        radius_decay = self.radius * (1 - iteration / max_iterations)  # Decaying radius

        for i in range(self.grid_x):
            for j in range(self.grid_y):
                distance_to_bmu = np.linalg.norm(np.array([i, j]) - np.array(bmu_idx))
                if distance_to_bmu <= radius_decay:
                    influence = np.exp(-distance_to_bmu / (2 * (radius_decay ** 2)))
                    self.weights[i, j] += lr * influence * (x - self.weights[i, j])

    def train(self, data, num_iterations):
        """
        Train the SOM using the input data
        """
        for iteration in range(num_iterations):
            x = data[np.random.randint(0, len(data))]
            bmu_idx = self._get_bmu(x)
            self._update_weights(x, bmu_idx, iteration, num_iterations)

    def visualize(self):
        """
        Visualize the SOM grid
        """
        fig, ax = plt.subplots()
        for i in range(self.grid_x):
            for j in range(self.grid_y):
                ax.scatter(self.weights[i, j][0], self.weights[i, j][1], color='blue')
        plt.show()


# Generate some random data (2D data for this example)
data = np.random.rand(100, 2)
print(data[0])

# Create and train the SOM
som = SOM(grid_x=10, grid_y=10, input_dim=2, learning_rate=0.5, radius=1.0, decay=0.99)
som.train(data, num_iterations=1000)

# Visualize the SOM
som.visualize()