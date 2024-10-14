from implementation import Solver, Upwind, SGscheme, Problem, Heat, FokkerPlanck, Results
import numpy as np
np.seterr(invalid='ignore',divide='ignore')

def convergence_heat(left_b,right_b,T_start,T_end):
    N = 7
    dt = 0.5
    dx = 0.5
    result = Results()
    problem = Heat(left_b=left_b,right_b=right_b,T_start=T_start, T_end= T_end)
    for i in range(N):
        solver = SGscheme(dt=dt, dx=dx)
        times, energy, errors = solver.solve(problem)
        result.store_energy_SG(energy-problem.min_energy(solver))
        result.store_L1error_SG(errors)
        result.store_time_SG(times)
        ##
        solver = Upwind(dt=dt, dx=dx)
        times, energy, errors = solver.solve(problem)
        result.store_energy_UP(energy-problem.min_energy(solver))
        result.store_L1error_UP(errors)
        result.store_time_UP(times)
        result.store_dxvalue(dx)
        ##
        dx = dx/2
        dt = dt/2
    result.plot_convergence()
    pass

def convergence_fokkerplanck(left_b,right_b,T_start,T_end):
    N = 8
    dt = 0.5
    dx = 0.5
    result = Results()
    problem = FokkerPlanck(left_b=left_b,right_b=right_b,T_start=T_start, T_end= T_end)
    for i in range(N):
        solver = SGscheme(dt=dt, dx=dx)
        times, energy, errors = solver.solve(problem)
        result.store_energy_SG(energy-problem.min_energy(solver))
        result.store_L1error_SG(errors)
        result.store_time_SG(times)
        ##
        solver = Upwind(dt=dt, dx=dx)
        times, energy, errors = solver.solve(problem)
        result.store_energy_UP(energy-problem.min_energy(solver))
        result.store_L1error_UP(errors)
        result.store_time_UP(times)
        result.store_dxvalue(dx)
        ##
        dx = dx/2
        dt = dt/2
    result.plot_convergence()
    pass

def energy_dissipation(problem, dt, dx, has_minimum = False, do_plot = True):
    """"""
    upsolver = Upwind(dt=dt,dx=dx,plot_solution=do_plot)
    SGsolver = SGscheme(dt=dt,dx=dx, plot_solution=do_plot)
    result = Results()
    times, energy, error = upsolver.solve(problem)
    result.store_dxvalue(dx)
    result.store_time_UP(times)
    if has_minimum:
        result.store_energy_UP(energy-problem.min_energy(upsolver))
    else:
        result.store_energy_UP(energy-energy[-1])
    times, energy, error = SGsolver.solve(problem)
    result.store_time_SG(times)
    if has_minimum:
        result.store_energy_SG(energy-problem.min_energy(SGsolver))
    else:
        result.store_energy_SG(energy-energy[-1])
    result.plot_energy()

def energy_dissipation_rate(left_b,right_b,T_start,T_end):
    N = 4
    dt = 0.5
    dx = 0.5
    result = Results()
    problem = FokkerPlanck(left_b=left_b,right_b=right_b,T_start=T_start, T_end= T_end)
    for i in range(N):
        solver = Upwind(dt=dt, dx=dx)
        times, energy, errors = solver.solve(problem)
        result.store_energy_UP(energy-problem.min_energy(solver))
        result.store_L1error_UP(errors)
        result.store_time_UP(times)
        result.store_dxvalue(dx)
        ##
        # solver = SGscheme(dt=dt, dx=dx)
        # times, energy, errors = solver.solve(problem)
        # result.store_energy_SG(energy-problem.min_energy(solver))
        # result.store_L1error_SG(errors)
        # result.store_time_SG(times)
        # result.store_dxvalue(dx)
        ##
        dx = dx/2
        dt = dt/2

    result.plot_energy_heat()

#energy_dissipation_rate(-5,5,1,5)
#convergence_heat(-10,10,2,3)
#convergence_fokkerplanck(-5,5,2,3)
# W = lambda x: -1.9*np.cos(2*np.pi*x)
# #sigma2 = 0.01
# init_func = lambda x: np.exp(-10*np.cos(2*np.pi*x))#1.0-0.1*np.cos(2*np.pi*x) #np.exp(-10*np.cos(2*np.pi*x))
# problem = Problem('ex1',0,1,0,6,0.1)
# problem.W = W
# problem.init_function = init_func
# # # ##
problem = Heat('heat',-5,5,0.1,5,1)

energy_dissipation(problem, dt=0.05,dx=0.05, has_minimum=False, do_plot=True)