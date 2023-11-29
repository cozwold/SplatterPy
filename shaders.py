
# Vertex Shader: Processes each vertex's position, color, and size; outputs to the geometry shader.
VERTEX_SHADER_SOURCE = """
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
GEOMETRY_SHADER_SOURCE = """
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
FRAGMENT_SHADER_SOURCE = """
#version 330 core
out vec4 FragColor;
in vec3 geoColor;  
in vec2 TexCoords;

uniform float amplitude;  // Declare the amplitude uniform variable

void main()
{
    vec2 coord = TexCoords - vec2(0.5, 0.5);  // Adjust TexCoords
    float dist_sq = dot(coord, coord);  // Distance squared from center
    vec3 center = geoColor;  // Use the color of the point as the center color
    float gaussian = amplitude * exp(-dist_sq / (2.0 * 0.2 * 0.2));  // Simplified Gaussian equation
    float alpha = clamp(gaussian, 0.0, 1.0);  // Clamp Gaussian to [0, 1] for alpha
    FragColor = vec4(geoColor * gaussian, alpha);  // Apply Gaussian to color
}

"""