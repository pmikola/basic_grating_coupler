import math as m

import numpy as np
import sympy
from sympy.solvers import solve
from sympy import Symbol
from scipy.constants import pi
from numpy import linspace, arange, tan
from numpy.lib.scimath import sqrt
import matplotlib.pyplot as plt


class GCP:
    # Grating coupler params
    def __init__(self, wavelength, n_cladding, n_core, n_substrate, core_height):
        self.center_wavelength = wavelength * 1E-6  # um
        self.k_0 = GCP.lambda2wave_vec(self.center_wavelength)
        self.n_cladding = n_cladding
        self.n_core = n_core
        self.n_substrate = n_substrate
        self.k_n_cladding = GCP.complexWavenumver(self.k_0, self.n_cladding)
        self.k_n_core = GCP.complexWavenumver(self.k_0, self.n_core)
        self.k_n_substrate = GCP.complexWavenumver(self.k_0, self.n_substrate)
        self.core_h = core_height
        GCP.coreSlabTEMmodes(self, 1000, 1)
        # TODO : Add effective index calculation for width and height dimention of waveguide

    def coreSlabTEMmodes(self, no_points, plot_slab):
        # beta**2 + kappa**2 = k**2 => Pythagorean formula for variables
        # rule of tumb => max(n_substrate, n_cladding) <= neff <= n_core
        self.kappamax = sqrt((self.k_n_core * self.n_core) ** 2 - (self.k_n_substrate * self.n_substrate) ** 2)
        self.kappa = linspace(1, self.kappamax, num=no_points)

        self.beta = sqrt((self.k_n_core * self.n_core) ** 2 - self.kappa ** 2)  # propagation constant
        self.gamma_substrate = sqrt(self.beta ** 2 - (self.k_n_substrate * self.n_substrate) ** 2)
        self.gamma_cladding = sqrt(self.beta ** 2 - (self.k_n_cladding * self.n_cladding) ** 2)
        self.y1 = tan(self.core_h * self.kappa)
        self.y2 = (self.gamma_cladding + self.gamma_substrate) / (
                    self.kappa * (1 - (self.gamma_cladding * self.gamma_substrate / self.kappa ** 2)))
        self.y3 = self.kappa * ((self.n_core ** 2 / self.n_substrate ** 2) * self.gamma_substrate + (
                    self.n_core ** 2 / self.n_cladding ** 2) * self.gamma_cladding) / (self.kappa ** 2 - (
                    self.n_core ** 4 / (
                        (self.n_cladding ** 2) * (self.n_substrate ** 2)) * self.gamma_cladding * self.gamma_substrate))
        self.n_eff = self.beta / self.k_n_core
        self.avg_n_eff = np.average(np.array(self.n_eff))
        # TODO Search for crossing functions
        # TODO What is effective index for this waveguide per mode?
        # beta / k = neff (znaleźć index w becie odpowiadający przecięciu kappy dla modu) :)

        # self.y_substracted = np.array(abs(abs(self.y1) - abs(self.y2)))
        # idx = np.argpartition(self.y_substracted, 3)
        # print(idx,self.y_substracted[idx[:3]],self.kappa[idx[:3]])

        if plot_slab == 1:
            plt.figure()
            plt.grid()
            plt.title("TE and TM Modes Func")
            plt.xlabel('kappa')  # x axis label
            plt.ylabel('Transcendental Funcs')
            plt.axis([0, self.kappamax, -10, 10])
            plt.plot(self.kappa, self.y1, label='tanh()')
            plt.plot(self.kappa, self.y2, label='TEmodes')
            plt.plot(self.kappa, self.y3, label='TMmodes')
            plt.plot(self.kappa, self.n_eff, label='n_eff')
            plt.legend()
            plt.show()


        else:
            pass

    @staticmethod
    def complexWavenumver(k_0, n):
        k = k_0 * n
        return k

    @staticmethod
    def lambda2wave_vec(wavelength):
        k_0 = 2 * m.pi / wavelength
        return k_0

    @staticmethod
    def indexFmaterial(er, ur):
        n = m.sqrt(er * ur)
        return n

    @staticmethod
    def BraggPhaseMatch(m, k_0, n_cladding, theta, n_eff):
        # Bragg Phase Match condition
        grating_period = Symbol('grating_period')
        BPM_eqn = ((k_0 * n_cladding * sympy.sin(theta) + m * 2 * sympy.pi / grating_period) / k_0) - n_eff
        solution = solve(BPM_eqn, grating_period)
        return solution


# SiN + SiO2
wl = 1.683  # wavelength
n_clad = 1.44
n_c = 2.45
n_sub = 1.44
c_h = 0.45E-6

G = GCP(wl, n_clad, n_c, n_sub, c_h)


grating_step = G.BraggPhaseMatch(1,G.k_0,G.n_cladding,10,G.avg_n_eff)

print(grating_step)