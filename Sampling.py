import numpy as np


class Sampling:
    def __init__(self, dimension):
        self.dimension = dimension

    def generate_isotropic_points(self, count):
        points = np.random.uniform(0, 1, count)
        points = np.repeat(points[:, np.newaxis], self.dimension, axis=1)
        return points

    def generate_disturbed_isotropic_points(self, base_points, count, std=0.1):
        selected_indices = np.random.choice(base_points.shape[0], size=count)
        disturbed_points = base_points[selected_indices, :] + np.random.normal(
            0, std, (count, self.dimension)
        )
        disturbed_points = np.clip(
            disturbed_points, 0, 1
        )  # Clip to keep within valid range
        return disturbed_points

    def generate_onehot_points(self, count):
        assert (
            count >= self.dimension
        ), "Count must be at least the dimension to cover all one-hot vectors."
        onehot_points = np.eye(self.dimension)
        return onehot_points

    def generate_disturbed_onehot_points(self, base_points, count, std=0.1):
        selected_indices = np.random.choice(
            base_points.shape[0], size=count, replace=True
        )
        disturbed_points = base_points[selected_indices, :] + np.random.normal(
            0, std, (count, self.dimension)
        )
        disturbed_points = np.clip(
            disturbed_points, 0, 1
        )  # Clip to keep within valid range
        return disturbed_points

    def generate_adjusted_nd_points(self, n_points, exclude_onehot=True):
        grid_size = int(np.ceil(np.power(n_points / 0.5, 1 / self.dimension)))
        grid_points = np.indices([grid_size] * self.dimension).reshape(
            self.dimension, -1
        ).T / (grid_size - 1)
        onehot_filter = (
            ~np.apply_along_axis(self.is_onehot, 1, grid_points)
            if exclude_onehot
            else np.ones(len(grid_points), dtype=bool)
        )
        sum_to_one_filter = np.sum(grid_points, axis=1) != 1
        isotropic_filter = ~np.all(grid_points == grid_points[:, [0]], axis=1)
        mask = onehot_filter & sum_to_one_filter & isotropic_filter
        filtered_points = grid_points[mask]
        if len(filtered_points) > n_points:
            indices = np.random.choice(len(filtered_points), n_points, replace=False)
            filtered_points = filtered_points[indices]
        return filtered_points

    def is_onehot(self, point):
        return (np.sum(point == 1) == 1) and (np.sum(point == 0) == len(point) - 1)

    def generate_combined_points(
        self, total_points, prop_isotropic, prop_onehot, include_onehot=True
    ):
        if include_onehot:
            onehot_points = self.generate_onehot_points(
                self.dimension
            )  # Always generates `dimension` one-hot vectors
            total_points -= self.dimension
            assert (
                total_points > 0
            ), f"Total points must be greater than number of dimension {self.dimension} since you included one-hot points. Increase total points or set 'include_onehot' to False."
        isotropic_count = int(prop_isotropic * total_points)
        onehot_count = int(prop_onehot * total_points)
        grid_count = total_points - isotropic_count - onehot_count

        isotropic_points = self.generate_isotropic_points(isotropic_count // 2)
        disturbed_isotropic_points = self.generate_disturbed_isotropic_points(
            isotropic_points, isotropic_count // 2
        )

        disturbed_onehot_points = self.generate_disturbed_onehot_points(
            onehot_points, onehot_count // 2
        )  # Use updated method

        grid_points = self.generate_adjusted_nd_points(grid_count)

        all_points = np.vstack(
            (
                isotropic_points,
                disturbed_isotropic_points,
                onehot_points,
                disturbed_onehot_points,
                grid_points,
            )
        )
        return all_points
