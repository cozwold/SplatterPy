# utils.py
import glfw
from OpenGL.GL import *
import numpy as np


def check_gl_error():
    error = glGetError()
    if error != GL_NO_ERROR:
        print(f"OpenGL error: {error}")


def perspective(fov, aspect, near, far):
    """
    Create a perspective projection matrix based on the field of view, aspect ratio, and near/far planes.

    :param fov: Field of view angle in the y direction, in radians.
    :type fov: float
    :param aspect: Aspect ratio, defined as view space width divided by height.
    :type aspect: float
    :param near: Distance from the viewer to the near clipping plane (always positive).
    :type near: float
    :param far: Distance from the viewer to the far clipping plane (always positive).
    :type far: float
    :returns: A perspective projection matrix.
    :rtype: numpy.ndarray
    """

    f = 1.0 / np.tan(fov / 2.0)
    return np.array(
        [
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0],
        ],
        dtype=np.float32,
    )


def look_at(eye, center, up):
    """
    Define a view matrix with an eye point, a reference point indicating the center of the scene, and an UP vector.

    :param eye: Position of the camera, in world space.
    :type eye: numpy.ndarray
    :param center: Position where the camera is looking at, in world space.
    :type center: numpy.ndarray
    :param up: Up vector, in world space (typically, this is the y-axis).
    :type up: numpy.ndarray
    :returns: A viewing matrix that transforms world coordinates to the camera's coordinate space.
    :rtype: numpy.ndarray
    """
    f = normalize(center - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)

    mat = np.identity(4)
    mat[0, 0:3] = s
    mat[1, 0:3] = u
    mat[2, 0:3] = -f
    mat[0:3, 3] = -np.dot(mat[0:3, 0:3], eye)
    return mat


def normalize(v):
    """
    Normalize a vector.

    This function returns the unit vector of the input vector. If the vector has zero length,
    the function will return the original vector.

    :param v: The vector to normalize.
    :type v: numpy.ndarray
    :returns: The normalized vector.
    :rtype: numpy.ndarray
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def rotate_y(angle):
    """
    Create a rotation matrix around the y-axis.

    :param angle: The rotation angle in radians.
    :type angle: float
    :returns: A rotation matrix that rotates a vector by `angle` radians around the y-axis.
    :rtype: numpy.ndarray
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], dtype=np.float32
    )