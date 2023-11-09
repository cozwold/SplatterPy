import numpy as np

# Permutation table. This is just a random jumble of all 8-bit numbers,
# repeated twice to avoid wrapping the index at 255 for each lookup.
# Create a full permutation array with all numbers from 0 to 255
permutation = np.random.RandomState(seed=0).permutation(256)

# Double the permutation array to avoid wrapping index at 255
perm = np.concatenate([permutation, permutation])


# Gradient vectors for each of the 12 edge midpoints in a cube.
grad3 = np.array([[1,1,0],[-1,1,0],[1,-1,0],[-1,-1,0],
                  [1,0,1],[-1,0,1],[1,0,-1],[-1,0,-1],
                  [0,1,1],[0,-1,1],[0,1,-1],[0,-1,-1]], dtype=np.float32)

def lerp(t, a, b):
    "Linear interpolation"
    return a + t * (b - a)

def fade(t):
    "6t^5 - 15t^4 + 10t^3"
    return t * t * t * (t * (t * 6 - 15) + 10)

def grad(hash, x, y, z):
    "Grad converts h to the right gradient vector and return the dot product with (x,y,z)"
    g = grad3[hash % 12]
    return g[0]*x + g[1]*y + g[2]*z

def perlin(x, y, z):
    "The main function that generates Perlin noise for a coordinate"
    # Determine grid cell coordinates
    X = int(np.floor(x)) & 255
    Y = int(np.floor(y)) & 255
    Z = int(np.floor(z)) & 255
    # Relative x, y, z of point in grid cell
    x -= np.floor(x)
    y -= np.floor(y)
    z -= np.floor(z)
    # Compute fade curves for x, y, z
    u = fade(x)
    v = fade(y)
    w = fade(z)
    # Hash coordinates of the 8 cube corners
    A = perm[X] + Y
    AA = perm[A] + Z
    AB = perm[A + 1] + Z
    B = perm[X + 1] + Y
    BA = perm[B] + Z
    BB = perm[B + 1] + Z
    # Add blended results from 8 corners of the cube
    res = lerp(w, lerp(v, lerp(u, grad(perm[AA], x, y, z), grad(perm[BA], x-1, y, z)),
                           lerp(u, grad(perm[AB], x, y-1, z), grad(perm[BB], x-1, y-1, z))),
                   lerp(v, lerp(u, grad(perm[AA+1], x, y, z-1), grad(perm[BA+1], x-1, y, z-1)),
                           lerp(u, grad(perm[AB+1], x, y-1, z-1), grad(perm[BB+1], x-1, y-1, z-1))))
    # We bound it to 0 - 1 (theoretical min/max before is -1 - 1)
    return (res + 1.0) / 2.0
def perlin_octaves(x, y, z, octaves, persistence, lacunarity):
    total = 0
    frequency = 1
    amplitude = 1
    max_value = 0  # Used for normalizing result to 0.0 - 1.0
    for i in range(octaves):
        total += perlin(x * frequency, y * frequency, z * frequency) * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    return total / max_value

def generate_voxel_terrain(width, length, height_scale, randomizer, octaves, persistence, lacunarity):
    depth = 2
    num_points = width * length * depth
    points = np.zeros((num_points, 7), dtype=np.float32)  # [x, y, z, r, g, b, size]

    index = 0
    for x in range(width):
        for z in range(length):
            # Get the surface height for this x, z coordinate
            surface_y = height_scale * perlin_octaves(x * 0.1, 0, z * 0.1, octaves, persistence, lacunarity)
            for y in range(int(surface_y), int(surface_y) - depth, -1):  # Extend from the surface down to 'depth'
                # Here we could use Perlin noise again for variation in the volume
                # or simply fill the volume under the terrain surface
                points[index, :3] = [x, y, z]
                # Assign color and size based on your criteria, for now, it's random
                points[index, 3:6] = randomizer.rand(3)
                points[index, 6] = randomizer.uniform(1.0, 1.5)
                index += 1

    return points




class Randomizer:
    def uniform(self, low, high, size=None):
        return np.random.uniform(low, high, size)

    def rand(self, *args):
        return np.random.rand(*args)

    def seed(self, seed_value):
        np.random.seed(seed_value)
