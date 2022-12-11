import numpy as np
import vispy.scene
from vispy.scene import visuals
from vispy.app.timer import Timer

class point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class body:
    def __init__(self, location, mass, velocity, name):
        self.location = location
        self.mass = mass
        self.velocity = velocity
        self.name = name

class Euler_integrator:
    def __init__(self, time_step, bodies):
        self.time_step = time_step
        self.bodies = bodies

    def calculate_single_body_acceleration(self, body_index):
        G_const = 6.67408e-11  # m3 kg-1 s-2

        # Use NumPy arrays to store locations, masses, and accelerations
        locations = np.array([b.location for b in self.bodies])
        masses = np.array([b.mass for b in self.bodies])
        accelerations = np.zeros_like(locations)

        # Calculate the distance between the target body and each external body
        r = locations[body_index] - locations
        r_sq = np.sum(r ** 2, axis=1)
        r_mag = np.sqrt(r_sq)

        # Calculate the acceleration on the target body due to each external body
        tmp = G_const * masses / r_sq ** (3 / 2)
        acceleration = np.sum(tmp[:, np.newaxis] * r, axis=0)

        return acceleration

    def compute_velocity(self):
        for body_index, target_body in enumerate(self.bodies):
            acceleration = self.calculate_single_body_acceleration(body_index)
            target_body.velocity.x += acceleration.x * self.time_step
            target_body.velocity.y += acceleration.y * self.time_step
            target_body.velocity.z += acceleration.z * self.time_step

            target_body.location.x += target_body.velocity.x * self.time_step
            target_body.location.y += target_body.velocity.y * self.time_step
            target_body.location.z += target_body.velocity.z * self.time_step

class Render:
    def __init__(self, bodies, time_step):
        # Create a Vispy canvas and add a view
        self.canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
        self.view = self.canvas.central_widget.add_view()
        self.bodies = bodies
        # Generate data for the scatter plot
        self.pos = np.empty((0, 3))
        self.symbols = []
        self.colors = []
        for body in bodies:
            self.pos = np.vstack((self.pos, [body.location.x, body.location.y, body.location.z]))
            self.symbols.append('o')
            self.colors.append([1, 1, 1, 1])

        # Create the scatter plot
        self.scatter = visuals.Markers()
        self.scatter.set_data(self.pos, edge_width=0, face_color=self.colors, size=5, symbol=self.symbols)

        # Add the scatter plot to the view
        self.view.add(self.scatter)

        # Set the camera to 'turntable'
        self.view.camera = 'turntable'

        # Add a colored 3D axis for orientation
        axis = visuals.XYZAxis(parent=self.view.scene)

        # Create an Euler integrator
        self.integrator = Euler_integrator(time_step, bodies)

        # Create a timer to update the positions of the bodies in the scatter plot
        self.timer = Timer(connect=self.update, interval=time_step * 1000)

        self.timer.start()

    def update(self, *args):
        # Update the positions of the bodies
        for body in self.bodies:
            self.integrator.compute_velocity(body)

        self.pos = np.empty((0, 3))

        for body in self.bodies:
            self.pos = np.vstack((self.pos, [body.location.x, body.location.y, body.location.z]))
        self.scatter.set_data(self.pos, edge_width=0, face_color=self.colors, size=5, symbol=self.symbols)

        # Update the canvas
        self.canvas.update()


bodies = [
    # Sun
    body(point(0, 0, 0), 1.9891e30, point(0, 0, 0), "Sun"),

    # Planets
    body(point(149.6e9, 0, 0), 5.972e24, point(0, 29.8e3, 0), "Earth"),
    body(point(0.723e9, 0, 0), 3.302e23, point(0, 24.1e3, 0), "Mercury"),
    body(point(1.082e11, 0, 0), 4.867e24, point(0, 13.1e3, 0), "Jupiter"),
    body(point(1.496e11, 0, 0), 5.683e26, point(0, 10.0e3, 0), "Saturn"),
    body(point(2.279e11, 0, 0), 8.681e25, point(0, 6.8e3, 0), "Uranus"),
    body(point(2.875e11, 0, 0), 1.024e26, point(0, 5.4e3, 0), "Neptune"),
    body(point(0.387e11, 0, 0), 6.421e23, point(0, 47.4e3, 0), "Venus"),
    body(point(0.524e11, 0, 0), 6.421e23, point(0, 35.0e3, 0), "Mars")
]

# Create a new Euler integrator
if __name__ == '__main__':
    time_step = 1000  # time step in seconds
    # Create the Euler integrator
    euler_integrator = Euler_integrator(time_step, bodies)

    # Create the renderer
    render = Render(bodies, time_step)
    # Start the event loop
    render.timer.start()
    vispy.app.run()
