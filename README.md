[![Python Versions](https://img.shields.io/pypi/pyversions/pythonworley?style=plastic)](https://pypi.org/project/pythonworley/)
[![PyPI](https://img.shields.io/pypi/v/pythonworley?style=plastic)](https://pypi.org/project/pythonworley/)
[![License](https://img.shields.io/pypi/l/pythonworley?style=plastic)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/pythonworley/badge/?version=latest)](https://pythonworley.readthedocs.io/en/latest/?badge=latest)

# Pythonworley

## Worley noise in python -- procedural generative art tool to seamlessly tile texture patterns in any dimensions
<br>

## Worley noise is a function introduced by Steven Worley in 1996. Worley noise is an extension of the Voronoi tessellation.
#


# Installation
```
pip install pythonworley
```

More examples and animations can be found at:

https://github.com/timpyrkov/procedural-art/

https://www.instagram.com/timpyrkov/

# Worley noise centers and Voronoi tesselation

Worley noise and Voronoi tesselation work on any random centers. Yet, placement of random centers placed in a regular grid gives an easy way to make a pattern which can be seamlessly tiled in any direction.

```
import pylab as plt
from scipy import spatial
from pythonworley import noisecoords, worley

# Set grid shape for randomly seeded gradients
shape = (8,4)

# Generate grid noise and set boundary=True 
# to pad it with periodic boundary points
noise = noisecoords(*shape, boundary=True, seed=0)

# Flatten X, Y coordinates and generate Voronoi tesselation
coords = noise.reshape(2,-1).T
vor = spatial.Voronoi(coords)
vert = vor.vertices
edge = vor.ridge_vertices
face = vor.regions


plt.figure(figsize=(12,6), facecolor="grey")

# Fill faces with random colors
rand = np.random.uniform(0.2, 0.8, len(face))
color = plt.get_cmap("Greys")(rand)
for i, f in enumerate(face):
    if len(f) and min(f) > 0:
        v = vert[f]
        plt.fill(v[:,0], v[:,1], c=color[i])

# Draw edges
for e in edge:
    if min(e) > 0:
        v = vert[e]
        plt.plot(v[:,0], v[:,1], c="black", lw=12)

# Plot centers
plt.scatter(*noise, c="black", s=200)

# Set xlim and ylim to hide periodic boundary padding points
plt.xlim(0, shape[0])
plt.ylim(0, shape[1])
plt.axis("off")
plt.show()
```
![](https://github.com/timpyrkov/pythonworley/blob/master/media/img_voronoi.jpg?raw=true)


# Cellular noise

Worley noise produces cellular noise pattern when colored dark to light with the distance to noise centers increasing.

```
import pylab as plt
from pythonworley import worley

# Set grid shape for randomly seeded gradients
shape = (4,4)

# Set density - output shape will be dens * shape = (256,256)
dens = 64

# Generate noise and centers
w, c = worley(shape, dens=dens, seed=0)

# Worley noise is an array of distances to the Nth closest neighbour centers.
# Let's select the first (the smallest). 
# Then transpose, because plt.imshow will treat axis 0 as the "Y".
w = w[0].T

# Concatenate to test that noise tiles seamlessly
w = np.concatenate([w] * 2, axis=1)

plt.figure(figsize=(12,6))
plt.imshow(w, cmap=plt.get_cmap("Greys_r"))
plt.plot([256,256], [0,256], "--", lw=3, color="black")
# plt.scatter(*c, c="white")
plt.axis("off")
plt.show()

```
![](https://github.com/timpyrkov/pythonworley/blob/master/media/img_tile.jpg?raw=true)


# Bubble pettern

Worley noise produces bubble pattern when colored light to dark as with the distance to noise centers increasing.

```
dens = 64
shape = (8,4)
w, c = worley(shape, dens=dens, seed=0)
w = w[0].T

plt.figure(figsize=(12,6))
plt.imshow(w, cmap=plt.get_cmap("Greys"))
# plt.scatter(*c, c="black")
plt.axis("off")
plt.show()
```
![](https://github.com/timpyrkov/pythonworley/blob/master/media/img_bubble.jpg?raw=true)


# Cobblestone pattern

Worley noise can produce ccobblestone pavement pattern. To do that we take the difference between the smallest and the second smallest distances to nise centers.

```
dens = 64
shape = (8,4)
w, c = worley(shape, dens=dens, seed=0)
w = w[1].T - w[0].T

plt.figure(figsize=(12,6))
plt.imshow(w, cmap=plt.get_cmap("Greys_r"))
# plt.scatter(*c, c="black")
plt.axis("off")
plt.show()
```
![](https://github.com/timpyrkov/pythonworley/blob/master/media/img_stone.jpg?raw=true)


# Generate procedural stars

* When generating a random procedural star field, a problem is how keep stars apart. Unfortunately, random placememt often tends to put some stars too close to each other.

* An elegant solution is to place stars based on the regular grid noise. Then shrink the size of each star to keep it within its grid cell. In this example we do not use Worley noise itself, but we use the grid noise which is similar to what we used to generate Worley noise centers above.

```
shape = (20,10)

# Make rectangular grid
x, y = np.arange(shape[0]), np.arange(shape[1])
x, y = np.meshgrid(x, y, indexing="ij")

# Generate noise: random displacements r at random angles phi
np.random.seed(0)
phi = np.random.uniform(0, 2 * np.pi, x.shape)
r = np.random.uniform(0, 0.5, x.shape)

# Shrink star size to keep it within its cell.
# Also, we want more small stars - for the background effect.
# To do that we rescale displacements: r -> 1/2 - 0.001 / r.
r = np.clip(0.5 - 1e-3 / r, 0, None)
size = 200 * (0.5 - r) - 0.4

# Convert r and phi to cartesian coordinates using the Euler formula.
z = r * np.exp(1j * phi)
dx, dy = z.real, z.imag
x, y = x + dx, y + dy

plt.figure(figsize=(12,6), facecolor="black")
plt.scatter(x, y, c="white", s=size)
plt.axis("off")
plt.show()
```
![](https://github.com/timpyrkov/pythonworley/blob/master/media/img_stars.png?raw=true)


# Generate procedural cityline

Again, in this example we do not use Worley noise itself. Instead we generate a random cityline based on the grid noise which is similar to what we used to generate Worley noise centers above.

```
# Let's werite a function to plot a building as a set of bar plots
def plot_building(center, width, height, floor_color, window_color, floor=3, basement=0):
    nfloor = int(height)
    if nfloor > floor:
        colors = [[window_color, floor_color] for i in range(nfloor - 1)]
        colors = [floor_color] * 2 + list(itertools.chain(*colors))
        heights = floor * np.arange(1, 1 + 2 * nfloor)[::-1] + basement
        centers = np.ones((2 * nfloor)) * center
        plt.bar(centers, heights, width=width, color=colors)


nparams = 3 # Position, width, and height per a building block
nblocks = 6 # Number of blocks per line
nlines = 3 # Number of lines
w = 20 # Average block width

# Generate grid noise
np.random.seed(0)
rand = np.random.uniform(0, w, (nlines, nparams, nblocks))

# Plot blocks line by line
plt.figure(figsize=(18,6), facecolor="w")
for i in range(nlines):
    darkness = (i + np.arange(2) + 1) / (nlines + 1)
    floor_color, window_color = plt.get_cmap("Greys")(darkness)
    center, width, height = rand[i]
    center += 3 * w * np.arange(nblocks) + w * i
    width += w
    for j in range(nblocks):
        plot_building(center[j], width[j], height[j], 
                      floor_color, window_color, basement=i)
plt.axis("off")
plt.show()
```
![](https://github.com/timpyrkov/pythonworley/blob/master/media/img_city.png?raw=true)


# Documentation

[https://pythonworley.readthedocs.io](https://pythonworley.readthedocs.io)