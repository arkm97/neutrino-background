import numpy as np
import matplotlib.pyplot as plt

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
    
    def integrate(self, initial_conditions, t0=0.0, tf=1.0, steps=100):
        """
        initial_conditions -- vector of initial conditions
        t0 -- start time (default: 0)
        tf -- end time (default: 1)
        steps -- number of steps (default: 100)
        """
        
        if len(init_conditions) != len(self.eq):
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

