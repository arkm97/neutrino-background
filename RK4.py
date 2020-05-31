import numpy as np


class RK4:

    def __init__(self, func_array):
        """
        func_array -- array containing RHS of initial value problems to integrate;
                    functions should take vectors as arguments
        """
        self.eq = func_array

    def _take_step(self, func, y, t, step_size):
        """
        take one step of explicit RK4
        """

        h = step_size

        k1 = h*np.array(func(t, y))
        k2 = h*np.array(func(t + h/2, y + k1/2))
        k3 = h*np.array(func(t + h/2, y + k2/2))
        k4 = h*np.array(func(t + h, y + k3))

        return 1/6*(k1 + 2*k2 + 2*k3 + k4)

    def _take_step_adaptive(self, func, y, t, step_size):
        """
        take one adaptive step of RK4
        """

        h = step_size

        k1 = h*np.array(func(t, y))
        k2 = h*np.array(func(t + 1/4 * h, y + 1/4 * k1))
        k3 = h*np.array(func(t + 3/8 * h, y + 3/32 * k1 + 9/32 * k2))
        k4 = h*np.array(func(t + 12/13 * h, y + 1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3))
        k5 = h*np.array(func(t + 1 * h, y + 439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4))
        k6 = h*np.array(func(t + 1/2 * h, y - 8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5))

        step_4 = 25 / 216 * k1 + 1408 / 2565 * k3 + 2197 / 4104 * k4  - 1/5 * k5
        step_5 = 16/135 * k1 + 6656/12825 * k3 + 28561/56430 * k4 - 9/50 * k5 + 2/55 * k6

        R = step_5 - step_4
        scale = (self.tolerance / (2 * np.sqrt(R.dot(R)))) ** .25
        new_step_size = scale * h

        return step_4, new_step_size

    def integrate(self, initial_conditions, t0=0.0, tf=1.0, steps=100):
        """
        initial_conditions -- vector of initial conditions
        t0 -- start time (default: 0)
        tf -- end time (default: 1)
        steps -- number of steps (default: 100)
        """
        if len(initial_conditions) != len(self.eq):
            return "len of init conditions != len eq array"

        step_size = (tf - t0) / steps
        times = np.linspace(t0, tf, steps)

        y = initial_conditions  # vector with len = number of variables

        #  output for each variable
        output = []
        variables = []
        for num_variables in range(len(initial_conditions)):
            output.append([])
            variables.append(y[num_variables])  # initialize each component


        for dt in times:

            # for each function in the set of equations
            for idx, f in enumerate(self.eq):

                step = self._take_step(f, y, dt, step_size)
                output[idx].append(variables[idx] + step[idx])
                variables[idx] += step[idx]

            # update y vector
            y = variables

        return [times, output]

    def integrate_adaptive(self, initial_conditions, t0=0.0, tf=1.0, tolerance=1e-3, min_step_size=None, max_step_size=None):
        """
        initial_conditions -- vector of initial conditions
        t0 -- start time (default: 0)
        tf -- end time (default: 1)
        tolerance -- max tolerable error in RK45; used to set 'optimal' step size
        min_step_size -- minimum step size (default: 0.0, eqivalent to no min step size)
        """
        if len(initial_conditions) != len(self.eq):
            return "len of init conditions != len eq array"

        self.tolerance = tolerance
        step_size = (tf - t0) / 100.

        if min_step_size == None:
            min_step_size = (tf - t0) / 100000

        if max_step_size == None:
            max_step_size = step_size

        times = [t0]

        y = initial_conditions  # vector with len = number of variables

        #  output for each variable
        output = []
        variables = []
        for num_variables in range(len(initial_conditions)):
            output.append([initial_conditions[num_variables]])
            variables.append(y[num_variables])  # initialize each component

        t = t0

        stopping_logic = 't < tf'
        if t0 > tf:
            stopping_logic = 't > tf'

        while eval(stopping_logic):

            smallest_new_step_size = np.inf
            # for each function in the set of equations
            for idx, f in enumerate(self.eq):

                step, new_step_size = self._take_step_adaptive(f, y, t, step_size)

                if np.abs(new_step_size) < np.abs(smallest_new_step_size):
                    smallest_new_step_size = new_step_size

                output[idx].append(variables[idx] + step[idx])
                variables[idx] += step[idx]

            # update y vector, times
            y = variables
            times.append(t + step_size)

            # update parameters for next set of steps
            t += step_size
            if (np.abs(smallest_new_step_size) < np.abs(min_step_size)) | (np.abs(smallest_new_step_size) == np.inf):
                smallest_new_step_size = min_step_size
            elif np.abs(smallest_new_step_size) > np.abs(max_step_size):
                smallest_new_step_size = max_step_size

            step_size = smallest_new_step_size


        return [times, output]
