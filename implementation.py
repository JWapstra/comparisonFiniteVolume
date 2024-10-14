import numpy as np
from scipy import signal
from scipy import sparse as sp
import scipy
import matplotlib.pyplot as plt

class Problem:
    def __init__(self, name="heat", left_b = -1, right_b = 1, T_start=2, T_end=4, diff_cof = 1 ):
        self.name = name
        self.left_b = left_b
        self.right_b = right_b
        self.T_start = T_start
        self.T_end=T_end
        self.diff_cof = diff_cof
        self.has_solution = False
        self.W = lambda x: 0*x
        self.init_function = lambda x: 1*x
        plt.rcParams.update({'font.size': 18})

    def min_energy(self, solver):
        x = solver.x
        rho_stat = np.ones(solver.M)/(solver.M*solver.dx)
        def stat_sol(rho):
            sigma = np.exp(-1/self.diff_cof*solver.dx*signal.fftconvolve(rho,solver.vec_w,'valid'))
            sigma = sigma/(solver.dx*np.sum(sigma))
            return sigma - rho
    
        sol = scipy.optimize.root(stat_sol,rho_stat, method='krylov', tol = 1.e-10)
        rho = sol.x
        print(sol.success)
        return self.diff_cof*solver.dx*np.sum(rho*(np.log(np.maximum(rho,solver.eps))-1)) + 0.5*solver.dx**2*np.sum(rho*signal.fftconvolve(rho,solver.vec_w,'valid'))


    def plot_energy(self, times, energy, solver):
        min_energy = self.min_energy(solver)
        print(min_energy)
        plt.figure()
        plt.semilogy(times,energy-min_energy)
        plt.show()
        pass

    def plot_error(self,times,errors):
        plt.figure()
        plt.semilogy(times,errors)
        plt.show()
    
class Heat(Problem):
    """
    
    """
    def __init__(self, name="heat", left_b = -1, right_b = 1, T_start=2, T_end=4, diff_cof = 1 ):
        super().__init__(name,left_b, right_b, T_start, T_end, diff_cof)
        self.W = lambda x: 0*x
        self.init_function = lambda x: self.heat_kernel(self.T_start,x)
        self.has_solution = True
        self.solution_function = lambda t,x: self.heat_kernel(t,x)
        
    def heat_kernel(self, t, x):
        return ((4*np.pi*self.diff_cof*t)**-0.5)*np.exp(-x**2/(4*self.diff_cof*t))
    
    def min_energy(self,solver):
        rho = np.ones(solver.M)/(solver.M*solver.dx)
        eps = 1.e-14
        return self.diff_cof*solver.dx*np.sum(rho*(np.log(np.maximum(rho,eps))-1))
    

    
class FokkerPlanck(Problem):
    def __init__(self, name="FokkerPlanck", left_b = -1, right_b = 1, T_start=2, T_end=4, diff_cof = 1 ):
        super().__init__(name,left_b, right_b, T_start, T_end, diff_cof)
        self.W = lambda x: x**2/2
        self.init_function = lambda x: self.fokker_planck_solution(self.T_start,x)
        self.has_solution = True
        self.solution_function = lambda t,x: self.fokker_planck_solution(t,x)

    def fokker_planck_solution(self, t ,x):
        """
        Returns analytical solution of Fokker-Planck equation at time t and position x 
        """
        return ((2*np.pi*self.diff_cof*(1-np.exp(-2*t)))**-0.5)*np.exp(-x**2/(2*self.diff_cof*(1-np.exp(-2*t))))
    
    def min_energy(self, solver):
        x = solver.x
        rho_stat = ((2*np.pi*self.diff_cof)**-0.5)*np.exp(-x**2/(2*self.diff_cof))
        def stat_sol(rho):
            sigma = np.exp(-1/self.diff_cof*solver.dx*signal.fftconvolve(rho,solver.vec_w,'valid'))
            sigma = sigma/(solver.dx*np.sum(sigma))
            return sigma - rho
    
        sol = scipy.optimize.root(stat_sol,rho_stat, method='krylov', tol = 1.e-10)
        rho = sol.x
        return self.diff_cof*solver.dx*np.sum(rho*(np.log(np.maximum(rho,solver.eps))-1)) + 0.5*solver.dx**2*np.sum(rho*signal.fftconvolve(rho,solver.vec_w,'valid'))

        

class Solver:
    def __init__(self, name, dt, dx, plot_solution = False):
        self.name = name
        self.dx = dx
        self.dt = dt
        self.eps = 1.e-14
        self.c = dt/dx
        self.plot_solution = plot_solution
        if plot_solution:
            fig = plt.figure()
            self.ax = fig.add_subplot(projection='3d')
            plt.rcParams.update({'font.size': 18})

    def energy(self, rho):
        return self.diff_cof*self.dx*np.sum(rho*(np.log(np.maximum(rho,self.eps))-1))+ 0.5*self.dx**2*np.sum(rho*signal.fftconvolve(rho,self.vec_w,'valid'))

    def L1error(self,t,rho,problem):
        # s = np.sum(self.dx*np.abs(0.5*(rho[:self.M-1]+rho[1:self.M])-problem.solution_function(t,self.y[1:self.M])))
        # return s + self.dx*0.5*(np.abs(rho[0]-problem.solution_function(t,self.y[0]))+np.abs(rho[-1]-problem.solution_function(t,self.y[-1])))
        return self.dx*np.sum(np.abs(rho-problem.solution_function(t,self.x)))

    def save_plot(self, rho,t):
        self.ax.plot(self.x, rho, zs=t, zdir='y', color='b', alpha=0.5)
        
    def solution(self):
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('t')
        self.ax.set_zlabel(r'$\rho$')
        self.ax.set_zlim3d(0,2)
        self.ax.set_title('Solution')#+str(self.name))
    
    def solve(self, problem):
        
        self.x = np.arange(problem.left_b,problem.right_b,self.dx) + self.dx/2
        self.y = np.arange(problem.left_b,problem.right_b+self.dx,self.dx)
        self.M = len(self.x)
        self.vec_w = problem.W(np.arange(-self.M+1,self.M)*self.dx)
        self.vec_w_q = -np.diff(self.vec_w)
        self.diff_cof = problem.diff_cof

        #rho0 = 0.5*(problem.init_function(self.y[:-1])+problem.init_function(self.y[1:]))
        rho0 = problem.init_function(self.x)
        #rho0[int(self.M/2)] += 1.e-4 
        s = self.dx*np.sum(rho0) # test if density
        rho = rho0/s

        t = problem.T_start
        times = []
        energy = []
        L1errors = []
        tick = 0
        while t<problem.T_end:
            rho = self.step(rho)
            t += self.dt
            times.append(t)
            energy.append(self.energy(rho))
            mass = self.dx*np.sum(rho)
            if self.plot_solution:
                tick +=1
                if tick %10 ==1:
                    self.save_plot(rho,t)
            if problem.has_solution:
                L1errors.append(self.L1error(t,rho,problem))
        if self.plot_solution:
            self.solution()
        print(1-mass)
        return np.array(times), np.array(energy), np.array(L1errors)
    
class Upwind(Solver):
    def __init__(self, name="Upwind", dt=1.e-1, dx=1.e-1, plot_solution = False):
        super().__init__(name,dt,dx,plot_solution)

    def step(self, rho_n):
        # Do one time step
        def fixed_point(rho):
            E = self.diff_cof*np.log(np.maximum(rho,self.eps)) + self.dx*signal.fftconvolve(0.5*(rho_n+rho),self.vec_w,'valid')# create vector E
            u = -np.diff(E)/self.dx
            umax = np.maximum(u,0)
            umin = np.minimum(u,0)
            # Create matrix from nonlinear system 
            d1 = np.zeros(self.M)
            d1[1:self.M-1] = (1+self.c*umax[1:self.M-1]-self.c*umin[0:self.M-2])
            d1[0] = (1+self.c*umax[0])
            d1[-1] = (1-self.c*umin[-1])
            d2 = -self.c*umax
            d3 = self.c*umin
            A = sp.diags([d2,d1,d3],[-1,0,1],format='csc')
            return A @ rho - rho_n
        
        sol = scipy.optimize.root(fixed_point,rho_n, method='krylov',tol=1.e-6)
        return sol.x
    
    
class SGscheme(Solver):
    def __init__(self, name = "SGscheme", dt=1.e-1, dx=1.e-1, plot_solution = False):
        super().__init__(name, dt, dx, plot_solution)
        self.c = dt/(dx**2)
        self.B = lambda s: np.where(np.abs(s)<10**-10,1, s/(np.exp(s)-1))   

    def step(self, rho_n):
        """"""
        def fixed_point(rho):
            q = signal.fftconvolve(0.5*(rho_n+rho),self.vec_w_q,'valid')
            # Create matrix from nonlinear system 
            d1 = np.zeros(self.M)
            Bq = self.c*self.diff_cof*self.B(self.dx/self.diff_cof*q)
            B_q = self.c*self.diff_cof*self.B(-self.dx/self.diff_cof*q)
            d1[1:self.M-1] = (1+Bq[0:self.M-2]+B_q[1:self.M-1])
            d1[0] = (1+B_q[0])
            d1[-1] = (1+Bq[-1])
            A = sp.diags([-B_q,d1,-Bq],[-1,0,1],format='csc')
            # Solve system
            return A @ rho - rho_n 

        sol = scipy.optimize.root(fixed_point,rho_n, method='krylov',tol=1.e-6)
        return sol.x
    

class Results:
    def __init__(self):
        self.energy_UP = []
        self.energy_SG = []
        self.L1error_UP = []
        self.last_error_up = []
        self.L1error_SG = []
        self.last_error_SG = []
        self.times_UP = []
        self.times_SG = []
        self.dx_values = []
        plt.rcParams.update({'font.size': 18})
    
    def store_time_UP(self,times):
        self.times_UP.append(times)

    def store_time_SG(self,times):
        self.times_SG.append(times)
    
    def store_energy_UP(self,energy):
        self.energy_UP.append(energy)
    
    def store_energy_SG(self,energy):
        self.energy_SG.append(energy)

    def store_dxvalue(self,dx):
        self.dx_values.append(dx)

    def store_L1error_UP(self,error):
        self.L1error_UP.append(error)
        self.last_error_up.append(error[-1])
    
    def store_L1error_SG(self,error):
        self.L1error_SG.append(error)
        self.last_error_SG.append(error[-1])

    def plot_energy(self):
        plt.figure(figsize=(8.5,6))
        for i in range(len(self.times_SG)):
            plt.semilogy(self.times_SG[i], self.energy_SG[i],linestyle = '-', label = r'SG $h = $'+str(self.dx_values[i])) 
            text1 = [r'$SGscheme h = $ ' for i in self.dx_values]
        for i in range(len(self.times_UP)):
            plt.semilogy(self.times_UP[i], self.energy_UP[i],linestyle= '--', label = r'Upwind $h = $'+str(self.dx_values[i]))
            text2 = [r'$Upwind h = $ ' for i in self.dx_values]
        plt.legend()
        plt.xlabel('Time: t')
        plt.ylabel('Relative energy')
        plt.show()

    def plot_energy_heat(self):
        plt.figure(figsize=(8.5,6))
        for i in range(len(self.times_SG)):
            plt.semilogy(self.times_SG[i], self.energy_SG[i], label = r'SG $h = $'+str(self.dx_values[i])) 
            text1 = [r'$SGscheme h = $ ' for i in self.dx_values]
        for i in range(len(self.times_UP)):
            plt.semilogy(self.times_UP[i], self.energy_UP[i], label = r'Upwind $h = $'+str(self.dx_values[i]))
            text2 = [r'$Upwind h = $ ' for i in self.dx_values]
        if len(self.times_SG)>0: 
            plt.semilogy(self.times_SG[-1],self.energy_SG[-1][0]*np.exp(-4*(self.times_SG[-1]-self.times_SG[-1][0])),'k', label = r'$\mathcal{O}(-4t)$')
        if len(self.times_UP)>0:
            plt.semilogy(self.times_UP[-1],self.energy_UP[-1][0]*np.exp(-4*(self.times_UP[-1]-self.times_UP[-1][0])),'k', label = r'$\mathcal{O}(-4t)$')
        plt.legend()
        plt.xlabel('Time: t')
        plt.ylabel('Relative energy')
        plt.title('Energy dissipation')
        plt.show()

    def plot_convergence(self):
        N = len(self.dx_values)
        error_rate_SG = np.log2(np.array(self.last_error_SG)[0:N-1]/np.array(self.last_error_SG)[1:N])
        error_rate_UP = np.log2(np.array(self.last_error_up)[0:N-1]/np.array(self.last_error_up)[1:N])
        print(error_rate_SG)
        print(error_rate_UP)
        plt.figure(figsize=(8.5,6))
        plt.loglog(self.dx_values,self.last_error_SG, 'x-')
        plt.loglog(self.dx_values,self.last_error_up,'x--')
        plt.loglog(self.dx_values,np.array(self.dx_values)*self.last_error_up[0], 'k')
        plt.legend(['SGscheme','Upwind', 'Order 1'], loc='upper left')
        plt.xlabel(r'$h$')
        plt.ylabel(r'$L^1 error$')
        plt.title('Convergence of schemes')
        plt.show()
