# camera.py
import numpy as np
from utils import normalize, perspective
class Camera:
    def __init__(
        self, position, target, up_vector, fov, aspect_ratio, near_plane, far_plane
    ):
        self.position = np.array(position, dtype=np.float32)
        self.target = np.array(target, dtype=np.float32)
        self.up_vector = np.array(up_vector, dtype=np.float32)
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.update_view_matrix()
        self.update_projection_matrix()

    def update_view_matrix(self):
        f = normalize(self.target - self.position)
        s = normalize(np.cross(f, self.up_vector))
        u = np.cross(s, f)

        mat = np.identity(4)
        mat[0, 0:3] = s
        mat[1, 0:3] = u
        mat[2, 0:3] = -f
        mat[0:3, 3] = -np.dot(mat[0:3, 0:3], self.position)
        self.view_matrix = mat

    def update_projection_matrix(self):
        f = 1.0 / np.tan(self.fov / 2.0)
        self.projection_matrix = np.array(
            [
                [f / self.aspect_ratio, 0, 0, 0],
                [0, f, 0, 0],
                [
                    0,
                    0,
                    (self.far_plane + self.near_plane)
                    / (self.near_plane - self.far_plane),
                    (2 * self.far_plane * self.near_plane)
                    / (self.near_plane - self.far_plane),
                ],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )

    def move(self, direction, amount):
        """
        Move the camera in a direction by a certain amount.

        :param direction: The direction to move the camera (forward, right, or up).
        :type direction: str
        :param amount: The amount to move the camera by.
        :type amount: float
        """
        forward_vector = normalize(self.target - self.position)
        right_vector = normalize(np.cross(forward_vector, self.up_vector))

        if direction == "forward":
            self.position += forward_vector * amount
        elif direction == "backward":
            self.position -= forward_vector * amount
        elif direction == "right":
            self.position += right_vector * amount
        elif direction == "left":
            self.position -= right_vector * amount
        elif direction == "up":
            self.position += self.up_vector * amount
        elif direction == "down":
            self.position -= self.up_vector * amount

        # After moving, the target needs to be updated to keep the camera direction consistent
        self.target = self.position + forward_vector

        self.update_view_matrix()

    def rotate(self, axis, angle):
        """
        Rotate the camera around an axis by a certain angle.

        :param axis: The axis to rotate the camera around (yaw or pitch).
        :type axis: str
        :param angle: The angle to rotate the camera by, in radians.
        :type angle: float
        """
        if axis == "yaw":
            rotation_matrix = rotate_y(angle)
            direction = self.target - self.position
            rotated_direction = np.dot(
                direction.reshape(1, -1), rotation_matrix[:3, :3]
            ).reshape(-1)
            self.target = self.position + rotated_direction
        elif axis == "pitch":
            # This requires more complex handling to prevent gimbal lock and ensure proper up vector orientation
            pass  # Placeholder for pitch implementation
        self.update_view_matrix()

