import numpy as np


def generate_points_from_model(sampled_points, num_points, noise_level, randomizer):
    # Assume sampled_points is an array of [x, y, z] coordinates
    points = np.zeros((num_points, 7), dtype=np.float32)
    if num_points > len(sampled_points):
        raise ValueError("Number of points requested exceeds the number of sampled points")

    # Use the sampled points for position, apply noise only to z
    points[:, :2] = sampled_points[:num_points, :2]
    points[:, 2] = sampled_points[:num_points, 2] + randomizer.uniform(-noise_level, noise_level, num_points)

    # Random color and size
    points[:, 3:6] = randomizer.rand(num_points, 3)  # color (RGB)
    points[:, 6] = randomizer.uniform(20.5, 20.9, num_points)  # size

    return points


class Randomizer:
    def uniform(self, low, high, size=None):
        return np.random.uniform(low, high, size)

    def rand(self, *args):
        return np.random.rand(*args)

    def seed(self, seed_value):
        np.random.seed(seed_value)

def generate_points(num_points, noise_level, window_size, randomizer):
    # Assume sampled_points is an array of [x, y, z] coordinates
    points = np.zeros((num_points, 7), dtype=np.float32)
    if num_points > len(sampled_points):
        raise ValueError("Number of points requested exceeds the number of sampled points")

    # Use the sampled points for position, apply noise only to z
    points[:, :2] = sampled_points[:num_points, :2]
    points[:, 2] = sampled_points[:num_points, 2] + randomizer.uniform(-noise_level, noise_level, num_points)

    # Random color and size
    points[:, 3:6] = randomizer.rand(num_points, 3)  # color (RGB)
    points[:, 6] = randomizer.uniform(0.01, 0.05, num_points)  # size

    return points