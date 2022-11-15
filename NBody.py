import math
import random
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class point:
    def __init__(self, x,y,z):
        self.x = x
        self.y = y
        self.z = z

class body:
    def __init__(self, location, mass, velocity, name = ""):
        self.location = location
        self.mass = mass
        self.velocity = velocity
        self.name = name

class Euler_integrator:
    def __init__(self, time_step, bodies):
        self.time_step = time_step
        self.bodies = bodies

    def calculate_single_body_acceleration(self, body_index):
        G_const = 6.67408e-11 #m3 kg-1 s-2
        acceleration = point(0,0,0)
        target_body = self.bodies[body_index]
        for index, external_body in enumerate(bodies):
            if index != body_index:
                r = (target_body.location.x - external_body.location.x)**2 + (target_body.location.y - external_body.location.y)**2 + (target_body.location.z - external_body.location.z)**2
                r = math.sqrt(r)
                tmp = G_const * external_body.mass / r**3
                acceleration.x += tmp * (external_body.location.x - target_body.location.x)
                acceleration.y += tmp * (external_body.location.y - target_body.location.y)
                acceleration.z += tmp * (external_body.location.z - target_body.location.z)
    
        return acceleration
    
    def update_location(self):
        for target_body in self.bodies:
            target_body.location.x += target_body.velocity.x * self.time_step
            target_body.location.y += target_body.velocity.y * self.time_step
            target_body.location.z += target_body.velocity.z * self.time_step
           
    def compute_velocity(self):
        for body_index, target_body in enumerate(self.bodies):
            acceleration = self.calculate_single_body_acceleration(body_index)
            target_body.velocity.x += acceleration.x * self.time_step
            target_body.velocity.y += acceleration.y * self.time_step
            target_body.velocity.z += acceleration.z * self.time_step 
    
    def update_location(self):
        for target_body in self.bodies:
            target_body.location.x += target_body.velocity.x * self.time_step
            target_body.location.y += target_body.velocity.y * self.time_step
            target_body.location.z += target_body.velocity.z * self.time_step

    def compute_gravity_step(self):
        self.compute_velocity()
        self.update_location()

class Render:
    def __init__(self, bodies, fig):
        self.bodies = bodies
        self.fig = fig
        self.ax = self.fig.add_subplot(1,1,1, projection='3d')
        
        max_range = 0
        for current_body in self.bodies:
            max_dim = max(max(current_body["x"]),max(current_body["y"]),max(current_body["z"]))
            if max_dim > max_range:
                max_range = max_dim
        
        self.ax.set_xlim([-max_range,max_range])    
        self.ax.set_ylim([-max_range,max_range])
        self.ax.set_zlim([-max_range,max_range])

    def update(self, *args):
        #print(args)
        fnum = args[0]
        
        colours = ['r','b','g','y','m','c']

        for current_body in self.bodies: 
            scatter = self.ax.scatter(current_body["x"][fnum], current_body["y"][fnum], current_body["z"][fnum], s = 1, c = colours[0], label = current_body["name"]) 
        
        return scatter

    def plot_output(self):
        ani = FuncAnimation(self.fig, func = self.update, frames = 100, interval = 10, repeat = False)
        plot.show()

def run_simulation(integrator, names = None, number_of_steps = 10000, report_freq = 100):

    #create output container for each body
    body_locations_hist = []
    for current_body in bodies:
        body_locations_hist.append({"x":[], "y":[], "z":[], "name":current_body.name})
        
    for i in range(0, int(number_of_steps)):
        if i % report_freq == 0:
            for index, body_location in enumerate(body_locations_hist):
                body_location["x"].append(bodies[index].location.x)
                body_location["y"].append(bodies[index].location.y)           
                body_location["z"].append(bodies[index].location.z)       
        integrator.compute_gravity_step()            

    return body_locations_hist        
            
#planet data (location (m), mass (kg), velocity (m/s)
sun = {"location":point(0,0,0), "mass":2e30, "velocity":point(0,0,0)}
mercury = {"location":point(0,5.7e10,0), "mass":3.285e23, "velocity":point(47000,0,0)}
venus = {"location":point(0,1.1e11,0), "mass":4.8e24, "velocity":point(35000,0,0)}
earth = {"location":point(0,1.5e11,0), "mass":6e28, "velocity":point(30000,0,0)}
mars = {"location":point(0,2.2e11,0), "mass":2.4e24, "velocity":point(24000,0,0)}
jupiter = {"location":point(0,7.7e11,0), "mass":1e28, "velocity":point(13000,0,0)}
saturn = {"location":point(0,1.4e12,0), "mass":5.7e26, "velocity":point(9000,0,0)}
uranus = {"location":point(0,2.8e12,0), "mass":8.7e25, "velocity":point(6835,0,0)}
neptune = {"location":point(0,4.5e12,0), "mass":1e26, "velocity":point(5477,0,0)}
pluto = {"location":point(0,3.7e12,0), "mass":1.3e22, "velocity":point(4748,0,0)}

if __name__ == "__main__":

    #build list of planets in the simulation, or create your own
    bodies = [
        body( location = sun["location"], mass = sun["mass"], velocity = sun["velocity"], name = "sun"),
        body( location = earth["location"], mass = earth["mass"], velocity = earth["velocity"], name = "earth"),
        body( location = mars["location"], mass = mars["mass"], velocity = mars["velocity"], name = "mars"),
        body( location = venus["location"], mass = venus["mass"], velocity = venus["velocity"], name = "venus"),
        ]
    
    integrator = Euler_integrator(time_step = 1000, bodies = bodies)
    motions = run_simulation(integrator, number_of_steps = 80000, report_freq = 1000)
    render = Render(fig = plot.figure(), bodies = motions)

    # render.update()
    render.plot_output()