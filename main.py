# main.py
import glfw
from OpenGL.GL import *
import numpy as np
import ctypes
from camera import Camera
from data_gen import generate_points, generate_points_from_model
from procedural_data_gen import generate_voxel_terrain, Randomizer
import trimesh
from utils import rotate_y, check_gl_error, normalize
from shaders import VERTEX_SHADER_SOURCE, GEOMETRY_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE

# Initialize mouse state variables
lastX, lastY = 1920 / 2, 1080 / 2  # Assuming a 1920x1080 window size
first_mouse = True  # This will help to check if the mouse has moved for the first time
yaw, pitch = -90.0, 0.0  # Initialize yaw and pitch



# Mouse callback function
# Mouse callback function
def mouse_callback(window, xpos, ypos):
    global lastX, lastY, first_mouse, yaw, pitch
    if first_mouse:  # Check if the mouse has moved for the first time
        lastX, lastY = xpos, ypos
        first_mouse = False

    # Calculate the offset movement between the last and current frame
    xoffset = xpos - lastX
    yoffset = lastY - ypos  # Reversed since y-coordinates range from bottom to top
    lastX, lastY = xpos, ypos

    sensitivity = 0.1  # Change this value to increase/decrease the camera's sensitivity
    xoffset *= sensitivity
    yoffset *= sensitivity

    yaw += xoffset
    pitch += yoffset

    # Constrain the pitch angle to prevent the camera from flipping
    if pitch > 89.0:
        pitch = 89.0
    elif pitch < -89.0:
        pitch = -89.0

    # Calculate the new front vector based on the updated yaw and pitch
    front = np.array(
        [
            np.cos(np.radians(yaw)) * np.cos(np.radians(pitch)),
            np.sin(np.radians(pitch)),
            np.sin(np.radians(yaw)) * np.cos(np.radians(pitch)),
        ]
    )

    # Update the camera's direction based on the new front vector
    camera.target = camera.position + normalize(front)
    camera.update_view_matrix()


# Mouse button callback function
def mouse_button_callback(window, button, action, mods):
    global first_mouse
    if button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
        first_mouse = True  # Reset the first_mouse flag
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    if button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.RELEASE:
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)


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
    camera.update_projection_matrix()
    glUniformMatrix4fv(projection_location, 1, GL_TRUE, camera.projection_matrix)
    # Update the points data with new window size
    vertices = generate_voxel_terrain(
        width, length, height_scale, randomizer, octaves, persistence, lacunarity
    )
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)


# Create a window
window = glfw.create_window(1920, 1080, "OpenGL Window", None, None)

# Check if window was created
if not window:
    glfw.terminate()
    raise Exception("glfw window can not be created!")

# Set window's position
glfw.set_window_pos(window, 400, 200)

# Make the context current
glfw.make_context_current(window)

# Set the mouse callbacks
glfw.set_cursor_pos_callback(window, mouse_callback)
glfw.set_mouse_button_callback(window, mouse_button_callback)

# Set the framebuffer size callback
glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)

# Now, you can set input mode for the cursor
glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)


# Enable Blending
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)  # Standard alpha blending

# Enable Depth Testing (needed to maintain z-order)
glEnable(GL_DEPTH_TEST)  # Enable depth testing
glDepthFunc(GL_LESS)  # Accept fragment if it's closer to the camera than the former one

# Disable Depth Testing (if necessary)
#glDisable(GL_DEPTH_TEST)
# Create and compile vertex shader
vertex_shader = glCreateShader(GL_VERTEX_SHADER)
glShaderSource(vertex_shader, VERTEX_SHADER_SOURCE)
glCompileShader(vertex_shader)

# Check for shader compile errors
success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
if not success:
    info_log = glGetShaderInfoLog(vertex_shader)
    raise RuntimeError(f"ERROR::SHADER::VERTEX::COMPILATION_FAILED\n{info_log}")

# Create and compile geometry shader
geometry_shader = glCreateShader(GL_GEOMETRY_SHADER)
glShaderSource(geometry_shader, GEOMETRY_SHADER_SOURCE)
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

# Create and compile fragment shader
fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
glShaderSource(fragment_shader, FRAGMENT_SHADER_SOURCE)
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
num_points = 100  # number of points you want to sample


# Initialize randomizer
randomizer = Randomizer()
randomizer.seed(42)  # Seed for reproducibility

# Generate terrain points
width = 200  # Width of the terrain
length = 200  # Length of the terrain
height_variation = 50.0  # Maximum variation in height
height_scale = 20.0  # Scale for the height variations


octaves = 4  # Number of detail layers
persistence = 0.5  # How much detail amplitudes decrease per octave
lacunarity = 2.0  # How much detail frequency increases per octave

# Replace 'width', 'length', 'height_scale', and 'randomizer' with your actual values
vertices = generate_voxel_terrain(
    width, length, height_scale, randomizer, octaves, persistence, lacunarity
)


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
fov = np.radians(100.0)
aspect_ratio = 720.0 / 480.0
near_plane = 0.1
far_plane = 10000.0


# Create the camera object
camera = Camera(
    position=[0.0, 10.0, 0.0],
    target=[0.0, 0.0, 00.0],
    up_vector=[0.0, 1.0, 0.0],
    fov=np.radians(100.0),
    aspect_ratio=720.0 / 480.0,
    near_plane=0.1,
    far_plane=10000.0,
)

# Get the uniform locations
projection_location = glGetUniformLocation(shader_program, "projection")
view_location = glGetUniformLocation(shader_program, "view")
model_location = glGetUniformLocation(shader_program, "model")

# Now it's safe to call update_projection_matrix() because 'camera' has been defined
camera.update_projection_matrix()


# Angle of rotation
angle = 0.0
last_frame = glfw.get_time()


# Render loop
while not glfw.window_should_close(window):
    # Calculate delta time
    current_frame = glfw.get_time()
    delta_time = current_frame - last_frame
    last_frame = current_frame

    camera_speed = 5.0 * delta_time

    # Input
    if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
        glfw.set_window_should_close(window, True)

    # Camera movement
    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        camera.move("forward", camera_speed)
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        camera.move("forward", -camera_speed)
    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
        camera.move("right", -camera_speed)
    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
        camera.move("right", camera_speed)
    if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
        camera.move("up", camera_speed)
    if glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS:
        camera.move("up", -camera_speed)

    # Update the camera's aspect ratio and recalculate matrices
    camera.aspect_ratio = (
        glfw.get_framebuffer_size(window)[0] / glfw.get_framebuffer_size(window)[1]
    )
    camera.update_view_matrix()
    camera.update_projection_matrix()
    # Update the rotation angle
    angle += 0.0  # This will rotate more slowly

    # Render
    # Set clear color to transparent
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Update the camera's aspect ratio and recalculate matrices
    camera.aspect_ratio = (
        glfw.get_framebuffer_size(window)[0] / glfw.get_framebuffer_size(window)[1]
    )
    camera.update_view_matrix()
    camera.update_projection_matrix()
    # Pass the matrices to the shader
    # Right before setting the uniform:
    glUseProgram(shader_program)  # Make sure you're using the correct shader program

    # And when calling glUniformMatrix4fv:
    glUniformMatrix4fv(projection_location, 1, GL_FALSE, camera.projection_matrix)
    glUniformMatrix4fv(view_location, 1, GL_TRUE, camera.view_matrix)
    # Activate shader program

    # Inside your render loop
    glUniform1f(glGetUniformLocation(shader_program, "size"), 0.05)
    glUniform1f(glGetUniformLocation(shader_program, "sigma"), 0.2)
    glUniform3f(glGetUniformLocation(shader_program, "center"), 0.0, 0.0, 0.0)
    glUniform1f(glGetUniformLocation(shader_program, "amplitude"), 1.0)

    # Update the model matrix with a new rotation
    model_matrix = rotate_y(angle)

    # Pass the matrices to the shader
    glUniformMatrix4fv(projection_location, 1, GL_TRUE, camera.projection_matrix)
    glUniformMatrix4fv(view_location, 1, GL_TRUE, camera.view_matrix)
    glUniformMatrix4fv(model_location, 1, GL_TRUE, model_matrix)

    # Draw the points
    # Draw the points
    glBindVertexArray(VAO)
    glDepthMask(GL_FALSE)  # Disable depth writing
    glDrawArrays(GL_POINTS, 0, len(vertices))
    glDepthMask(GL_TRUE)  # Re-enable depth writing

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
