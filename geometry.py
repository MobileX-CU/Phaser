
"""
Various utility functions for 3D geometry computations
"""

import numpy as np
from scipy.optimize import curve_fit, fmin
from sympy import nsolve
from sympy.abc import x as x_sym
from sympy.abc import y as y_sym
from sympy.abc import z as z_sym

def func_degree2(data, p00, p10, p01, p20, p11, p02):
    """
    Polynomial function of degree 2, defined by the coefficients p*
    
    Parameters:
    data : 2xN array 
        Array of points, where each column is a point in 2D space
    p00, p10, p01, p20, p11, p02: ints
        Coefficients of the polynomial function

    Returns:
    z : 1xN array 
        Array of z values for each point, computed by the polynomial function

    """
    x = data[0,:]
    y = data[1,:]
    return p00 + p10*x + p01*y + p20*x**2 + p11*x*y + p02*y**2 

def dx_func_degree2(data, coeffs):
    """
    Partial derivative of func_degree2 with respect to x

    Parameters:
    data : 2xN array 
        Array of points, where each column is a point in 2D space
    coeffs : list
        Coefficients of the degree 2 polynomial function
    
    Returns:
    dx : 1xN array 
        Array of partial derivative values for each point
    """
    p00, p10, p01, p20, p11, p02 = coeffs
    x = data[0,:]
    y = data[1,:]
    return p10 + 2*p20*x + p11*y

def dy_func_degree2(data, coeffs):
    """
    Partial derivative of func_degree2 with respect to y

    Parameters:
    data : 2xN array 
        Array of points, where each column is a point in 2D space
    coeffs : list
        Coefficients of the degree 2 polynomial function
    
    Returns:
    dy : 1xN array 
        Array of partial derivative values for each point
    """
    p00, p10, p01, p20, p11, p02 = coeffs
    x = data[0,:]
    y = data[1,:]
    return p01 + 2*p02*y + p11*x

def func_degree3(data, p00, p10, p01, p20, p11, p02, p30, p21, p12, p03):
    """
    Polynomial function of degree 3, defined by the coefficients p*

    Parameters:
    data : 2xN array 
        Array of points, where each column is a point in 2D space
    p00, p10, p01, p20, p11, p02, p30, p21, p12, p03: ints
        Coefficients of the polynomial function
    
    Returns:
    z : 1xN array 
        Array of z values for each point, computed by the polynomial function
    """
    x = data[0,:]
    y = data[1,:]
    return p00 + p10*x + p01*y + p20*x**2 + p11*x*y + p02*y**2 + p30*x**3 + p21*x**2*y + p12*x*y**2 + p03*y**3

def plane_plane_intersection(plane1_eq, plane2_eq):
    """
    Compute the intersection line of two planes
    https://stackoverflow.com/questions/48126838/plane-plane-intersection-in-python

    Parameters:
    plane1_eq, plane2_eq : 4-tuples/lists
        Equations of the planes, in the form of [A, B, C, D], where Ax + By + Cz + D = 0
        is the equation of the plane representation
    
    Returns:
    intersection_line_points : list of two points
        Two 3D points on the line of intersection
    """
    plane1_vec, plane2_vec = plane1_eq[:3], plane2_eq[:3]
    plane1X2_vec = np.cross(plane1_vec, plane2_vec)
    A = np.array([plane1_vec, plane2_vec, plane1X2_vec])
    d = np.array([-plane1_eq[3], -plane2_eq[3], 0.]).reshape(3,1)
    p_inter = np.linalg.solve(A, d).T
    intersection_line_points = [p_inter[0], (p_inter + plane1X2_vec)[0]]
    return intersection_line_points

def line_surface_intersection(line_p0, line_direction, surface_coeffs):
  
    def f(x, y):
        """ 
        Function of the surface"
        [TODO] make this function not redundant with other
        """
        z = surface_coeffs[0] + surface_coeffs[1]*x + surface_coeffs[2]*y + surface_coeffs[3]*x**2 + surface_coeffs[4]*x*y + surface_coeffs[5]*y**2 
        return z

    def line_func(t):
        """Function of the straight line.
        :param t:     curve-parameter of the line

        :returns      xyz-value as array"""
        return line_p0 + t*line_direction

    def target_func(t):
        """Function that will be minimized by fmin
        :param t:      curve parameter of the straight line

        :returns:      (z_line(t) - z_surface(t))**2 – this is zero
                    at intersection points"""
        p_line = line_func(t)
        z_surface = f(*p_line[:2])
        return np.sum((p_line[2] - z_surface)**2)

    t_opt = fmin(target_func, x0=-10)
    intersection_point = line_func(t_opt)
    return intersection_point

def surface_surface_intersection_point(coeffs1, coeffs2, z_val):
    """
    Get point at specified z value on the intersection curve of two polynomial surfaces of degree 2

    Parameters:
    coeffs1, coeffs2 : lists
        Coefficients of the degree 2 polynomial functions (defined by func_degree2) of the two surfaces
    z_val : float
        Z value at which to find the intersection point
    
    Returns:
    point_on_interection : list
        3D point on the intersection curve of the two surfaces
    """
    p00_1, p10_1, p01_1, p20_1, p11_1, p02_1 = coeffs1
    p00_2, p10_2, p01_2, p20_2, p11_2, p02_2 = coeffs2

    eqs = [p00_1 + p10_1*x_sym + p01_1*y_sym + p20_1*x_sym**2 + p11_1*x_sym*y_sym + p02_1*y_sym**2 - z_sym, 
           p00_2 + p10_2*x_sym + p01_2*y_sym + p20_2*x_sym**2 + p11_2*x_sym*y_sym + p02_2*y_sym**2 - z_sym, 
           z_sym - z_val]

    point_on_interection = nsolve(eqs, [x_sym, y_sym, z_sym], [0, 0, 0])
    return [point_on_interection[0], point_on_interection[1], point_on_interection[2]]

def fit_3d_surface(data, func):
    """
    Fit a 3D polynomial surface 

    Parameters:
    data : N x 3 array 
        Array of 3D points, where each row is one 3D point
    func : function
        General form of polynomial function to fit the data to
    
    Returns:
    popt : list
        Optimal values for the coefficients so that the sum of the squared residuals of func(xdata, *popt) - ydata is minimized
    X, Y, Z : 2D arrays
        Meshgrid arrays for plotting the fitted surface
    rms : float
        Root mean squared error of the fit w.r.t to the input data
    """
    
    # compute fit
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    xdata = np.vstack((x.ravel(), y.ravel()))
    guess_prms = [.5, .5, .5, .5, .5, .5]
    popt, pcov = curve_fit(func, xdata, z.ravel(), guess_prms) #not sure if ulab scipy has optimize

    # compute RMS of fit
    fit = np.zeros(z.shape)
    for i in range(z.shape[0]):
        points = xdata[:,i].reshape((2,1))
        fit[i] = func(points, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])[0] #popt[6], popt[7], popt[8], popt[9])[0]
    rms = np.sqrt(np.mean((y - fit)**2))

    # below for plotting
    X,Y = np.meshgrid(np.linspace(min(x), max(x), 10), np.linspace(min(y), max(y), 10))
    XX = X.flatten()
    YY = Y.flatten()
    grid = np.vstack((XX, YY))
    Z = func(grid,  popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]).reshape(X.shape) #popt[6], popt[7], popt[8], popt[9]).reshape(X.shape)
    return popt, X, Y, Z, rms

def compute_tangent_plane(p0, func, dx, dy, coeffs):
    """
    Compute the tangent plane of the surface at a point
    https://math.libretexts.org/Courses/Mount_Royal_University/MATH_3200%3A_Mathematical_Methods/6%3A__Differentiation_of_Functions_of_Several_Variables/6.4%3A__Tangent_Planes_and_Linear_Approximations

    Parameters:
    p0 : 1D array, shape (3,)
        3D point on the surface
    func: function 
        General form for polynomial function of the surface
    coeffs: list
        Coefficients of the polynomial function func
    dx : function
        Partial derivative function of the surface (defined by func) with respect to x
    dy : function
        Partial derivative function of the surface (defined by func) with respect to y

    Returns:
    plane : list 
        Equation of the tangent plane, in the form of [A, B, C, D], where Ax + By + Cz + D = 0
        is the equation of the plane representation
    """
    A = dx(p0[:2], coeffs)
    B = dy(p0[:2], coeffs)
    C = -1
    D = func(p0[:2], *coeffs) - A*p0[0] - B*p0[1]
    plane_eq = np.array([A[0], B[0], C, D[0]]).astype(np.float64) # casting necessary for later operations
    return plane_eq
    
def get_angle(vec1, vec2):
    """
    Compute the angle between two vectors

    Parameters:
    vec1, vec2 : 1D arrays
        Two vectors to compute the angle between
    
    Returns:
    angle : float
        Angle between the two vectors, in degrees
    """
    c = np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2) 
    angle = np.rad2deg(np.arccos(np.clip(c, -1, 1))) 
    return angle
