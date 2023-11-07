import glfw
from OpenGL.GL import *
import numpy as np
import ctypes

from data_gen import generate_points, generate_points_from_model
from procedural_data_gen import generate_voxel_terrain, Randomizer
import trimesh

# Vertex Shader: Processes each vertex's position, color, and size; outputs to the geometry shader.
vertex_shader_source = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in float aSize;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
out vec3 vertexColor;
out float vertexSize;

void main()
{
    // Main function for vertex shader that transforms vertices and passes color and size to geometry shader.
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    vertexColor = aColor;
    vertexSize = aSize;
}
"""

# Geometry Shader: Receives vertices from the vertex shader; emits vertices forming a triangle strip.
geometry_shader_source = """
#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;
in vec3 vertexColor[];  // Receive color from vertex shader
in float vertexSize[];  // Receive size from vertex shader
out vec3 geoColor;  // Pass color to fragment shader
out vec2 TexCoords;

void main() {
    geoColor = vertexColor[0];  // Pass along color to fragment shader
    vec4 position = gl_in[0].gl_Position;
    float size = vertexSize[0];

    gl_Position = position + vec4(-size, -size, 0, 0);
    TexCoords = vec2(0, 0);
    EmitVertex();

    gl_Position = position + vec4(size, -size, 0, 0);
    TexCoords = vec2(1, 0);
    EmitVertex();

    gl_Position = position + vec4(-size, size, 0, 0);
    TexCoords = vec2(0, 1);
    EmitVertex();

    gl_Position = position + vec4(size, size, 0, 0);
    TexCoords = vec2(1, 1);
    EmitVertex();
}
"""


# Fragment Shader: Calculates the color of each pixel by applying a Gaussian function for soft-edge rendering.
fragment_shader_source = """
#version 330 core
out vec4 FragColor;
in vec3 geoColor;  
in vec2 TexCoords;

uniform float amplitude;  // Declare the amplitude uniform variable

void main()
{
    vec2 coord = TexCoords * 2.0 - 1.0;  // Normalize to [-1, 1]
    float dist_sq = dot(coord, coord);  // Distance squared from center
    vec3 center = geoColor;  // Use the color of the point as the center color
    float gaussian = amplitude * exp(-dist_sq / (2.0 * 0.2 * 0.2));  // Simplified Gaussian equation
    float alpha = clamp(gaussian, 0.0, 1.0);  // Clamp Gaussian to [0, 1] for alpha
    FragColor = vec4(geoColor * gaussian, alpha);  // Apply Gaussian to color
}

"""

def check_gl_error():
    error = glGetError()
    if error != GL_NO_ERROR:
        print(f'OpenGL error: {error}')
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
    return np.array([
        [f/aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
        [0, 0, -1, 0]
    ], dtype=np.float32)
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
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

# Initialize GLFW
if not glfw.init():
    raise Exception("glfw can not be initialized!")


def framebuffer_size_callback(window, width, height):
    """
    Callback function for when the window's framebuffer size is changed.

    This function is registered with GLFW to be called whenever the window is resized. It updates
    the viewport to the new window size and recalculates the projection matrix.

    :param window: The window that received the event.
    :type window: glfw.Window
    :param width: The new width of the framebuffer.
    :type width: int
    :param height: The new height of the framebuffer.
    :type height: int
    """
    glViewport(0, 0, width, height)
    aspect_ratio = width / height
    projection_matrix = perspective(fov, aspect_ratio, near_plane, far_plane)
    glUniformMatrix4fv(projection_location, 1, GL_TRUE, projection_matrix)
    # Update the points data with new window size
    vertices = generate_voxel_terrain(width, length, height_scale, randomizer, octaves, persistence, lacunarity)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

# Create a window
window = glfw.create_window(720, 480, "OpenGL Window", None, None)

# Check if window was created
if not window:
    glfw.terminate()
    raise Exception("glfw window can not be created!")

# Set window's position
glfw.set_window_pos(window, 400, 200)

# Set the framebuffer size callback
glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)


# Make the context current
glfw.make_context_current(window)

# Enable Blending
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)  # Standard alpha blending

# Enable Depth Testing (needed to maintain z-order)
glEnable(GL_DEPTH_TEST)  # Enable depth testing
glDepthFunc(GL_LESS)  # Accept fragment if it's closer to the camera than the former one

# Disable Depth Testing (if necessary)
glDisable(GL_DEPTH_TEST)
# Create the vertex shader
vertex_shader = glCreateShader(GL_VERTEX_SHADER)
glShaderSource(vertex_shader, vertex_shader_source)
glCompileShader(vertex_shader)

# Check for shader compile errors
success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
if not success:
    info_log = glGetShaderInfoLog(vertex_shader)
    raise RuntimeError(f"ERROR::SHADER::VERTEX::COMPILATION_FAILED\n{info_log}")

# Create shaders
geometry_shader = glCreateShader(GL_GEOMETRY_SHADER)
glShaderSource(geometry_shader, geometry_shader_source)
glCompileShader(geometry_shader)
# Check for shader compile errors
success = glGetShaderiv(geometry_shader, GL_COMPILE_STATUS)
if not success:
    info_log = glGetShaderInfoLog(geometry_shader)
    raise RuntimeError(f"ERROR::SHADER::GEOMETRY::COMPILATION_FAILED\n{info_log}")


# Check for shader compile errors
success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
if not success:
    info_log = glGetShaderInfoLog(vertex_shader)
    raise RuntimeError(f"ERROR::SHADER::VERTEX::COMPILATION_FAILED\n{info_log}")

fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
glShaderSource(fragment_shader, fragment_shader_source)
glCompileShader(fragment_shader)

# Check for shader compile errors
success = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
if not success:
    info_log = glGetShaderInfoLog(fragment_shader)
    raise RuntimeError(f"ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n{info_log}")

# Link shaders
shader_program = glCreateProgram()
glAttachShader(shader_program, vertex_shader)
glAttachShader(shader_program, geometry_shader)  # Attach geometry shader
glAttachShader(shader_program, fragment_shader)
glLinkProgram(shader_program)


# Check for linking errors
success = glGetProgramiv(shader_program, GL_LINK_STATUS)
if not success:
    info_log = glGetProgramInfoLog(shader_program)
    raise RuntimeError(f"ERROR::SHADER::PROGRAM::LINKING_FAILED\n{info_log}")

glDeleteShader(vertex_shader)
glDeleteShader(fragment_shader)
glDeleteShader(geometry_shader)

# Replace this with your array of points
initial_window_size = (720, 480)
num_points = 100 # number of points you want to sample


# Initialize randomizer
randomizer = Randomizer()
randomizer.seed(42)  # Seed for reproducibility

# Generate terrain points
width = 200  # Width of the terrain
length = 200  # Length of the terrain
height_variation = 50.0  # Maximum variation in height
height_scale = 40.0  # Scale for the height variations


octaves = 4         # Number of detail layers
persistence = 0.5   # How much detail amplitudes decrease per octave
lacunarity = 2.0    # How much detail frequency increases per octave

# Replace 'width', 'length', 'height_scale', and 'randomizer' with your actual values
vertices = generate_voxel_terrain(width, length, height_scale, randomizer, octaves, persistence, lacunarity)





VBO = glGenBuffers(1)
VAO = glGenVertexArrays(1)

glBindVertexArray(VAO)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(0))  # position
glEnableVertexAttribArray(0)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(12))  # color
glEnableVertexAttribArray(1)
glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(24))  # size
glEnableVertexAttribArray(2)
glBindBuffer(GL_ARRAY_BUFFER, 0)
glBindVertexArray(0)
# Create the perspective projection matrix
fov = np.radians(25.0)
aspect_ratio = 720.0 / 480.0
near_plane = 0.1
far_plane = 10000.0
projection_matrix = perspective(fov, aspect_ratio, near_plane, far_plane)

# Create the view matrix
camera_pos = np.array([0.0, 110.0, 0.0], dtype=np.float32)
camera_target = np.array([100.0, 0.0, 50.0], dtype=np.float32)
camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
view_matrix = look_at(camera_pos, camera_target, camera_up)

# Create a model matrix
model_matrix = np.identity(4, dtype=np.float32)

# Get the uniform locations
projection_location = glGetUniformLocation(shader_program, "projection")
view_location = glGetUniformLocation(shader_program, "view")
model_location = glGetUniformLocation(shader_program, "model")

# Angle of rotation
angle = 0.0

# Render loop
while not glfw.window_should_close(window):
    # Input
    if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
        glfw.set_window_should_close(window, True)

    # Update the rotation angle
    angle += 0.0  # This will rotate more slowly

    # Render
    # Set clear color to transparent
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    # Activate shader program

    # Inside your render loop
    glUseProgram(shader_program)
    glUniform1f(glGetUniformLocation(shader_program, "size"), 0.05)
    glUniform1f(glGetUniformLocation(shader_program, "sigma"), 0.2)
    glUniform3f(glGetUniformLocation(shader_program, "center"), 0.0, 0.0, 0.0)
    glUniform1f(glGetUniformLocation(shader_program, "amplitude"), 1.0)

    # Update the model matrix with a new rotation
    model_matrix = rotate_y(angle)

    # Pass the matrices to the shader
    glUniformMatrix4fv(projection_location, 1, GL_TRUE, projection_matrix)
    glUniformMatrix4fv(view_location, 1, GL_TRUE, view_matrix)
    glUniformMatrix4fv(model_location, 1, GL_TRUE, model_matrix)

    # Draw the points
    # Draw the points
    glBindVertexArray(VAO)
    glDrawArrays(GL_POINTS, 0, len(vertices))
    check_gl_error()  # Check for errors after drawing
    glBindVertexArray(0)
    # Swap the screen buffers
    glfw.swap_buffers(window)

    # Poll for and process events
    glfw.poll_events()

# Properly clean/delete all of the resources that were allocated
glDeleteVertexArrays(1, [VAO])
glDeleteBuffers(1, [VBO])
glDeleteProgram(shader_program)

glfw.terminate()
