import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class Euler_integrator:
    def __init__(self, time_step, bodies):
        self.time_step = time_step
        self.bodies = bodies

    def compute_gravity_step(self):
        # Calculate pairwise distances between bodies
        r = np.sum((self.bodies[:, :3] - self.bodies[:, :3][:, np.newaxis])**2, axis=2)
        r = np.sqrt(r)
        # Set diagonal elements of distance matrix to infinity so they are not considered
        np.fill_diagonal(r, np.inf)

        # Compute pairwise gravitational forces
        G_const = 6.67408e-11  # m3 kg-1 s-2
        forces = G_const * self.bodies[:, 3][:, np.newaxis] * self.bodies[:, 3] / r**3
        # Sum forces in each dimension
        acceleration = np.sum(forces * (self.bodies[:, :3] - self.bodies[:, :3][:, np.newaxis]), axis=0)

        # Update velocities and positions
        self.bodies[:, 3:6] += acceleration * self.time_step
        self.bodies[:, :3] += self.bodies[:, 3:6] * self.time_step

class Render:
    def __init__(self, bodies, fig):
        self.bodies = bodies
        self.fig = fig
        self.ax = self.fig.add_subplot(1,1,1, projection='3d')

        # Set plot limits based on maximum body position in any dimension
        max_range = np.max(np.abs(self.bodies[:, :3]))
        self.ax.set_xlim([-max_range, max_range])    
        self.ax.set_ylim([-max_range, max_range])
        self.ax.set_zlim([-max_range, max_range])

    def update(self, *args):
        self.ax.clear()
        self.ax.scatter(self.bodies[:, 0], self.bodies[:, 1], self.bodies[:, 2])

# Initialize bodies
bodies = np.array([[1, 0, 0, 0, 1, 0, "Sun"],
                   [0, 0, 5.2, 0, 0, 0, "Earth"],
                   [0, 1, 0, 0, 0, 0, "Moon"]])

# Set time step and initialize integrator
time_step = 1  # s
integrator = Euler_integrator(time_step, bodies)

# Set up animation
fig = plt.figure()
render = Render(bodies, fig)
ani = FuncAnimation(fig, render.update, frames=range(100), interval=100)
plt.show()
