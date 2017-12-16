"""
plasmapy.plasma
===============

Defines the core Plasma class used by PlasmaPy to represent plasma properties.
"""

import numpy as np
import astropy.units as u
from astropy.utils.console import ProgressBar
from .simulation import MHDSimulation, dot
from ..constants import mu0


class Plasma():
    """Core class for describing and calculating plasma parameters.

    Attributes
    ----------
    x : `astropy.units.Quantity`
        x-coordinates within the plasma domain. Equal to the
        `domain_x` input parameter.
    y : `astropy.units.Quantity`
        y-coordinates within the plasma domain. Equal to the
        `domain_y` input parameter.
    z : `astropy.units.Quantity`
        z-coordinates within the plasma domain. Equal to the
        `domain_z` input parameter.
    grid : `astropy.units.Quantity`
        (3, x, y, z) array containing the values of each coordinate at
        every point in the domain.
    domain_shape : tuple
        Shape of the plasma domain.
    density : `astropy.units.Quantity`
        (x, y, z) array of mass density at every point in the domain.
    momentum : `astropy.units.Quantity`
        (3, x, y, z) array of the momentum vector at every point in
        the domain.
    pressure : `astropy.units.Quantity`
        (x, y, z) array of pressure at every point in the domain.
    magnetic_field : `astropy.units.Quantity`
        (3, x, y, z) array of the magnetic field vector at every point
        in the domain.

    Parameters
    ----------
    domain_x : `astropy.units.Quantity`
        1D array of x-coordinates for the plasma domain. Must have
        units convertable to length.
    domain_y : `astropy.units.Quantity`
        1D array of y-coordinates for the plasma domain. Must have
        units convertable to length.
    domain_z : `astropy.units.Quantity`
        1D array of z-coordinates for the plasma domain. Must have
        units convertable to length.
    """
    @u.quantity_input(domain_x=u.m, domain_y=u.m, domain_z=u.m)
    def __init__(self, domain_x, domain_y, domain_z, gamma=5/3):
        # Define domain sizes
        self.x = domain_x
        self.y = domain_y
        self.z = domain_z

        self.grid = np.array(np.meshgrid(self.x, self.y, self.z,
                                         indexing='ij'))
        self.domain_shape = (len(self.x), len(self.y), len(self.z))
        self.gamma = gamma
        # Initiate core plasma variables
        self.density = np.zeros(self.domain_shape) * u.kg / u.m**3
        self.momentum = np.zeros((3, *self.domain_shape)) * u.kg / (u.m**2 * u.s)
        self.pressure = np.zeros(self.domain_shape) * u.Pa
        self.magnetic_field = np.zeros((3, *self.domain_shape)) * u.T
        self.electric_field = np.zeros((3, *self.domain_shape)) * u.V / u.m
        self._energy = np.zeros(self.domain_shape) * u.J / u.m**3

        # Collect core variables into a list for usefulness
        self.core_variables = [self.density, self.momentum, self.pressure, self.magnetic_field,
                               self.electric_field]

        # Connect a simulation object for simulating
        self.simulation_physics = MHDSimulation(self)

    @property
    def velocity(self):
        return self.momentum / self.density

    def sound_speed(self):
        return np.sqrt((self.gamma * self.pressure) / self.density)   

    @property
    def energy(self):
        return self._energy

    @energy.setter
    @u.quantity_input
    def energy(self, energy: u.J/u.m**3):
        """Sets the simulation's total energy density profile to the specified array.
        Other arrays which depend on the energy values, such as the kinetic
        pressure, are then redefined automatically.
        Parameters
        ----------
        energy : numpy.ndarray
            Array of energy values. Shape must be (x, y, z), where x, y, and z
            are the grid sizes of the simulation in the x, y, and z dimensions.
            Must have units of energy.
        """

        assert energy.shape == self.domain_shape, """
            Specified density array shape {} does not match simulation grid {}.
            """.format(energy.shape, self.domain_shape)
        self._energy = energy

    @property
    def magnetic_field_strength(self):
        B = self.magnetic_field
        return np.sqrt(np.sum(B * B, axis=0))

    @property
    def electric_field_strength(self):
        E = self.electric_field
        return np.sqrt(np.sum(E * E, axis=0))

    @property
    def alfven_speed(self):
        B = self.magnetic_field
        rho = self.density
        return np.sqrt(np.sum(B * B, axis=0) / (mu0 * rho))

    @u.quantity_input(max_time=u.s)
    def simulate(self, max_its=np.inf, max_time=np.inf * u.s):
        """Simulates the plasma as set up, either for the given number of
        iterations or until the simulation reaches the given time.
        Parameters
        ----------
        max_its : int
            Tells the simulation to run for a set number of iterations.
        max_time : astropy.units.Quantity
            Maximum total (in-simulation) time to allow the simulation to run.
            Must have units of time.
        Examples
        --------
        # >>> # Run a simulation for exactly one thousand iterations.
        # >>> myplasma.simulate(max_time=1000)
        # >>> # Run a simulation for up to half an hour of simulation time.
        # >>> myplasma.simulate(max_time=30*u.minute)
        """
        if np.isinf(max_its) and np.isinf(max_time.value):
            raise ValueError("Either max_time or max_its must be set.")

        physics = self.simulation_physics
        dt = physics.dt

        if np.isinf(max_time):
            pb = ProgressBar(max_its)
        else:
            pb = ProgressBar(int(max_time / dt))

        with pb as bar:
            while (physics.current_iteration < max_its
                   and physics.current_time < max_time):
                physics.time_stepper()
                bar.update()
