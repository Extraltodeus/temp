# Make a program using Pygame that simulates the solar system. Follow the following rules precisely:
# 1) Draw the sun and the planets as small balls and also draw the orbit of each planet with a line.
# 2) The balls that represent the planets should move following its actual (scaled) elliptic orbits according to Newtonian gravity and Kepler's laws
# 3) Draw a comet entering the solar system and following an open orbit around the sun, this movement must also simulate the physics of an actual comet while approaching and turning around the sun.
# 4) Do not take into account the gravitational forces of the planets acting on the comet.


import pygame
import math
import numpy as np

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

# Constants
GMSUN = 139 #39.478                  # Gravitational parameter (AU³/year²)
SCALE_FACTOR = 40               # Pixels per astronomical unit (AU)
SUN_RADIUS = 30                 # Sun's radius in pixels

# Simulation time scaling (years/second) to ensure visible motion
SIMULATION_TIME_SCALE = 1

# Screen center for drawing objects
SCREEN_CENTER = (screen.get_width() // 2, screen.get_height() // 2)

# Function to convert simulation coordinates to screen coordinates
def get_screen_coords(x_sim, y_sim):
    screen_x = SCREEN_CENTER[0] + int(x_sim * SCALE_FACTOR)
    screen_y = SCREEN_CENTER[1] - int(y_sim * SCALE_FACTOR)  # Invert Y-axis for Pygame
    return (screen_x, screen_y)

# Function to solve Kepler's equation numerically
def solve_kepler(M, e, tolerance=1e-8, max_iter=100):
    E = M
    for _ in range(max_iter):
        delta = E - e * math.sin(E) - M
        if abs(delta) < tolerance:
            return E
        dE_dM = 1.0 / (1 - e * math.cos(E))
        E -= delta * dE_dM
    raise RuntimeError("Kepler equation did not converge.")

# Function to convert eccentric anomaly to true anomaly
def eccentric_to_true_anomaly(E, e):
    return 2 * math.atan(math.sqrt((1 + e) / (1 - e)) * math.tan(E / 2))

# Define planetary data: a = semi-major axis [AU], e = eccentricity, period [years]
PLANETS_DATA = [
    {'name': 'Mercury', 'a': 0.387, 'e': 0.21, 'period': 0.241},
    {'name': 'Venus', 'a': 0.723, 'e': 0.0067, 'period': 0.615},
    {'name': 'Earth', 'a': 1.0, 'e': 0.0167, 'period': 1.0},
    {'name': 'Mars', 'a': 1.524, 'e': 0.0934, 'period': 1.881},
    {'name': 'Jupiter', 'a': 5.203, 'e': 0.0489, 'period': 11.862}
]

# Precompute orbit points for all planets
planet_orbits = []
for p in PLANETS_DATA:
    a, e = p['a'], p['e']
    num_points = 360
    orbit_points = []
    for theta in np.linspace(0, 2 * math.pi, num_points):
        r_sim = (a * (1 - e**2)) / (1 + e * math.cos(theta))
        x = r_sim * math.cos(theta)
        y = r_sim * math.sin(theta)
        orbit_points.append((x, y))
    planet_orbits.append(orbit_points)

# Comet initial conditions: position and velocity in AU/year
COMET_X, COMET_Y = 10.0, 0.0  # Start at 10 AU from the Sun
COMET_VX, COMET_VY = 0.0, 3.0  # Velocity sufficient for a hyperbolic orbit

# Main simulation loop
running = True
simulation_time = 0.0
previous_simulation_time = 0.0

while running:
    dt_real = clock.tick(60) / 1000.0  # seconds per frame
    simulation_time += SIMULATION_TIME_SCALE * dt_real
    screen.fill((0, 0, 0))

    # Draw the Sun
    pygame.draw.circle(screen, (255, 255, 0), SCREEN_CENTER, SUN_RADIUS)

    # Draw Planets and their orbits
    for i, planet in enumerate(PLANETS_DATA):
        a, e, period = planet['a'], planet['e'], planet['period']
        M = 2 * math.pi * simulation_time / period
        E = solve_kepler(M, e)
        theta = eccentric_to_true_anomaly(E, e)

        r_sim = (a * (1 - e**2)) / (1 + e * math.cos(theta))
        x_sim, y_sim = r_sim * math.cos(theta), r_sim * math.sin(theta)
        screen_x, screen_y = get_screen_coords(x_sim, y_sim)

        # Draw planet as a small white circle
        pygame.draw.circle(screen, (255, 255, 255), (int(screen_x), int(screen_y)), 3)

        # Draw orbit line in light gray
        for j in range(len(planet_orbits[i]) - 1):
            x1_sim, y1_sim = planet_orbits[i][j]
            x2_sim, y2_sim = planet_orbits[i][j + 1]
            sx1, sy1 = get_screen_coords(x1_sim, y1_sim)
            sx2, sy2 = get_screen_coords(x2_sim, y2_sim)
            pygame.draw.line(screen, (200, 200, 200), (int(sx1), int(sy1)), (int(sx2), int(sy2)), 1)

    # Update and draw the comet
    dt_sim = simulation_time - previous_simulation_time

    if dt_sim > 0:
        r_comet = math.hypot(COMET_X, COMET_Y)
        if r_comet != 0.0:
            ax = -GMSUN * COMET_X / (r_comet ** 3)
            ay = -GMSUN * COMET_Y / (r_comet ** 3)
        else:
            ax = ay = 0

        # Update velocity and position
        COMET_VX += ax * dt_sim
        COMET_VY += ay * dt_sim
        COMET_X += COMET_VX * dt_sim
        COMET_Y += COMET_VY * dt_sim

    screen_x, screen_y = get_screen_coords(COMET_X, COMET_Y)
    pygame.draw.circle(screen, (255, 0, 0), (int(screen_x), int(screen_y)), 4)

    previous_simulation_time = simulation_time

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()

pygame.quit()
