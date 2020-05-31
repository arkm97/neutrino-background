# neutrino-background

The goal of the project is to calculate the local overdensity of relic neutrinos from the very early universe.  An understanding of this local density provides insight into the mass evolution of our local group; broadly, an understanding of the cosmic neutrino background provides insight into an even earlier stage of the universe than the CMB can provide.  The work preempts empirical calculation of the local overdensity, and thus will help set plausible ranges for relic neutrino detection benchmarks.

I evolve samples from a carefully chosen initial distribution to build a final profile at z=0 (present).  The initial phase space distribution follows Fermi-Dirac statistics for highly relativistic particles.  The final distribution is assembled by adding weighted contributions from each initial bin at their final positions.

`RK4.py` is the class file for my implemenation of a Runge-Kutta 4-5 adaptive integrator.  This does the heavy lifting solving the second order PDE to calculate the evolution of each sample.  

`density profile and potential table.ipynb` is a notebook that calculates the matter evolution and gravitational potential of the milky way.  It outputs a reference tabe used at each step in the numerical integration.

`orbits2.ipynb` is a notebook that runs the time evolution and assembles the final density profile. 
