import numpy as np
import pdb
import geomdl
from geomdl import helpers
import copy
from geomdl.visualization import VisPlotly
from geomdl.visualization import VisMPL as vis
import matplotlib.pyplot as plt

    
def assign_ub(Q):
    ''' Sets parameter uk that corresponds to control point Qk
    
    From eq 9.5 p 365 BON
    :param Q: Array containing the sampled points
    :type Q: nparray
    :return: parameters associated with each controlpoint
    :rtype: nparray
    '''
    d = 0
    for i in range(1, len(Q)):
        d += np.linalg.norm(Q[i] - Q[i - 1])
        
    ub = np.array([0])
    for j in range(1, len(Q)):
        ub = np.append(ub, np.array(ub[-1] + np.linalg.norm(Q[j] - Q[j - 1]) / d))
    
    np.append(ub, np.array([1]))
    return ub.tolist()


def set_knots(n, p, m, ub):
    ''' Creates an internal knotvector
    
    Uses eq. 9.68 and 9.69 from BON p 412
    :param n: n+1 control points
    :type n: int
    :param p: degree of curve
    :type p: int
    :param m: m+1 number of sample points
    :type m: int
    :param ub: parameter associated with every controlpoint
    :type ub: np.array
    :return: U, list of knots
    :rtype: nparray
    '''
    k = n - p + 1
    d = (m + 1) / k
    
    # Multiplicity of first and last knot is p+1!
    U = np.zeros((n + p + 2,))
    # last p+1 knots equal 1 !
    U[-(p + 1):] = 1
    
    # Internal knots
    for j in range(1, k):
        i = int(np.floor(j * d))
        alpha = j * d - i
        U[p + j] = (1 - alpha) * ub[i - 1] + alpha * ub[i]
    return U


def curve_fit2D(Q, Wq, D, I, Wd, n, p, s=-1):
    # Paramter originally in function call
    r = len(Q) - 1
    
    ''' Weighted & constrained least square fit
    
    
    : param Q: An array containing the points to be fit( constrained and unconstrain
    ed)
    : type Q: nparray, Q[r+1]
    : param Wq: Wq[i] > 0 is weight of unconst Q[i]. Wq[i] < 0, Q[i] constrained
    : type Wq: list of floats, Wq[r+1]
    : param D: An array containing the derivatives; s=-1 means no derivatives spec.
    : type D: List, D[s+1]
    : param s: s+1 derivatives in D[]. s=-1 means no derivatives specified
    : type s: int
    : param I: I[j] gives the index into Q[] of the point corresponding to derivative D[j]
    : type I: list of int
    : param Wd: Wd[j]>0 is weight of unconstrained D[j]. Wd[j]<0 then D[j] is const.
    : type Wd: list of floats
    : param n: Specifies a fit with n+1 control points
    : type n: int
    : param p: Specifies a fit with degree p curve
    : type p: int
    :return: An array P with control points values
    :rtype: numpy array

    Algorithm A9. 6, p 417, BON
    '''
    
    '''
    Local arrays:
    ub[r+1]: The parameters
    N[mu+1][n+1], M[mc+1][n+1], S[mu+1], T[mc+1], A[mc+1]: The arrays at p.416 in BON
    W[mu+1]: The weights (diagonal matrix or vector...)
    funs [2] [p+1]: basis functions and derivatives where specified
    '''

    ru = -1
    rc = -1
    for i in range(0, r + 1):
        if Wq[i] > 0:
            ru = ru + 1
        else:
            rc = rc + 1
    su = -1
    sc = -1
    for j in range(0, s + 1):
        if Wd[j] > 0:
            su = su + 1
        else:
            sc = sc + 1
    mu = ru + su + 1
    mc = rc + sc + 1
    
    if mc >= n or mc + n >= mu + 1:
        raise ValueError("Too many constrained points, or too many control \
                         points in relation to number of samples done!")
    
    # Allocate memory
    dim = len(Q[0])  # Number of dimensions
    M = np.ones((mc + 1, n + 1)) * np.nan
    N = np.ones((mu + 1, n + 1)) * np.nan
    Wv = np.ones((mu + 1,)) * np.nan
    S = np.ones((mu + 1, 1)) * np.nan
    T = np.ones((mc + 1, 1)) * np.nan
    P = np.ones((n + 1, dim)) * np.nan
    # Compute and load parameters uk into ub[], (eq 9.5)
    ub = assign_ub(Q)
    ub = np.linspace(0, 1, len(Q))
    
    # Compute and load the knots into U[], (eq 9.68 and 9.69)
    U = set_knots(n, p, r, ub)
        
    # Set up arrays N, W, S, T, M
    j = 0
    mu2 = 0
    mc2 = 0
    for i in range(0, r + 1):
        span = helpers.find_span_binsearch(p, U, n + 1, ub[i])  # "n+1" is only n in BON
        dflag = 0
        if j <= s:
            if i == I[j]:
                dflag = 1
        if dflag == 0:
            funs_nz = helpers.basis_function(p, U, span, ub[i])
            funs = np.zeros((n + 1,))
            funs[(span - p): (span + 1)] = funs_nz
        else:
            funs = helpers.basis_function_ders(p, U, span, ub[i], 1)  # Derivative function
        if Wq[i] > 0:  # Unconstrained point
            Wv[mu2] = Wq[i]
            # Load the m2th row of N[][] from funs [0][];
            N[mu2, :] = funs
            S[mu2] = Wv[mu2] * Q[i, 1]
            mu2 = mu2 + 1
        else:  # Constrained point
            # Load the mc2th row of M[][] from funs [0][];
            M[mu2, :] = funs
            T[mc2] = Q[i, 1]
            mc2 = mc2 + 1
        
        if dflag == 1:
            # Derivative at this point
            if Wd[j] > 0:
                # Unconstrianed derivative
                Wv[mu2] = Wd[j]
                # Load the mu2th row of N[][] from funs [1][]
                N[mu2, :] = funs[1, :]
                S[mu2] = Wv[mu2] * D[j, 1]
                mu2 = mu2 + 1
            else:
                # Constrained derivative
                # Load the mc2h row of M[][} from funs[1][];
                M[mu2, :] = funs[1, :]
                T[mc2] = D[j, 1]
                mc2 = mc2 + 1
        j += 1  # Correct?! Or should it be outside this loop?
        
    # Make W a diagonal matrix!
    W = np.diag(Wv)
    
    # # Compute the matrices N^T W N and N^T W S
    # NWN = np.matmul(N.transpose(), np.matmul(W, N))
    # NWS = np.matmul(N.transpose(), np.dot(W, S))
    # LUDecomposition(NTWN,n+1,p);
    
    # Fix S into matrix
    S = np.column_stack((Q[:, 0], S))
    
    if mc < 0:
        for i in range(0, dim):
            # No constraints
            
            # Compute the matrices N^T W N and N^T W S
            NWN = np.matmul(N.transpose(), np.matmul(W, N))
            NWS = np.matmul(N.transpose(), np.dot(W, S[:, i]))
            P[:, i] = np.linalg.solve(NWN, NWS)
              
            # Use ForwardBackward() to solve for the control points P[]
            # (N^T*W*N)P = N^TW*S
        # Override!
        
        # min_x = min(Q[:, 0])
        # max_x = max(Q[:, 0])
        # P[:, 0] = np.linspace(min_x, max_x, n+1)
        
        # Set type to same as geomdl package
        P = tuple([tuple(l) for l in P])
        U = U.tolist()
        
        return P, U
        
    else:
        # We have constraints this code is not yet implemented.
        raise ValueError('Constraints in Fitting.py is not yet implemented')
        # Compute the inverse (N^T*W*N)^(-1), using ForwardBackward().
        # Do matrix operations to get: M((N^T*W*N)^(-1))M^T and
        # M(N^T*W*N)^(-1)N^TW*S - T
        # Solve Eq.(9.75) for the Lagrange multipliers, load into A[]
        # Then P = ... (eq 9.74)


def Bp(curve, ul, j):
    ''' Computes derivatives of nurbe function at position ul wrt change of control point pl
    See Nils Carlson master thesis, p. 28
    
    NOT TESTED!
    
    param: ul, parameter of interest, np.array, scalar, float
    param: j, index of the control point that is to be differentiated, int
    retrun: lst object
    '''
    
    # Do a loop for N-dimensional
    
    # Function value at nominal point
    fun_val = curve.evaluate_single(ul)
    
    # Create two new curves, one with change of x, another with change of y
    curve_dx = copy.deepcopy(curve)
    curve_dy = copy.deepcopy(curve)
    
    # if curve_dx.ctrlptsw[j][0] < 0:
    #     curve_dx.ctrlptsw[j][0] = 0
    #     print('WARNING: nurb weights are smaller than zero!')
    # elif curve_dy.ctrlptsw[j][1] < 0:
    #     curve_dy.ctrlptsw[j][1] = 0
    #     print('WARNING: nurb weights are smaller than zero!')
    
    # Set forward finite disturbance to 1%,  Log value (transoformed weights e^w)
    dx = curve_dx.ctrlptsw[j][0] * (1 + 1e-4) - curve_dx.ctrlptsw[j][0]
    dy = curve_dy.ctrlptsw[j][1] * (1 + 1e-4) - curve_dy.ctrlptsw[j][1]
    
    if dx == 0:
        dx = 1e-4
    if dy == 0:
        dy = 1e-4
    
    # Change the values
    curve_dx.ctrlptsw[j][0] += dx
    curve_dy.ctrlptsw[j][1] += dy
    
    # Function value with change of dx, dy
    fun_dx = curve_dx.evaluate_single(ul)
    fun_dy = curve_dy.evaluate_single(ul)
    
    der_x = (np.array(fun_dx) - np.array(fun_val)) / dx
    der_y = (np.array(fun_dy) - np.array(fun_val)) / dy
    
    return np.vstack((der_x, der_y)).tolist()
    
    
def weight_der(curve, wi, u):
    ''' derivative wrt to a weight
    
    NOT YET TESTED
    
    param: curve, nurbs curve, NURBS curve object
    param: wi, weight (number), int
    param: parameter along curve
    return: lst
    '''
    
    # Nominal value
    fun_val = curve.evaluate_single(u)
    W = []
    for elem in curve.ctrlptsw:
        W.append(elem[-1])
    
    W = np.log(W)
    W_old = copy.deepcopy(W)
    dw = W[wi] * 1e-4  # one percent increase
    if dw <= 0:
        dw = 1e-4
    W[wi] += W[wi] + dw
    W = np.exp(W).tolist()
    
    # Function value with change of dw
    curve_dw = copy.deepcopy(curve)  # Have to make a new curve!
    for i in range(0, len(curve.ctrlptsw)):
        curve_dw.ctrlptsw[i][-1] = W[i]
    
    fun_dw = curve_dw.evaluate_single(u)
    
    return ((np.array(fun_dw) - np.array(fun_val)) / dw).tolist()
    
    
def gn_jacobi(curve, ub, Q, alpha):
    ''' Function that outputs jacobi for line search with regularisation step
    NOT TESTED!
    
    param: curve, Nurbs curve from geomdl package,
    param: ub, np.array of parameterization, [r+1,]
    param: alpha, weight related to weight matrix, see Nils Carlson master thesis.
    param: Q, test points
    
    local parameters
    param: U, np.array of knots, [m+1,]
    param: W, np.array of weights, [n+1,]
    param: CP, np.array of control points [n+1, 2]
    return: J, list (2*(r+1) + n ) x (r + 3 * (n+1))
    '''
    
    CP = list()
    for elem in curve.ctrlptsw:
        CP.append(elem[:-1])
    
    # Indexing
    n = len(curve.ctrlptsw)
    r = len(ub)
    dim = 2  # Dimension, for future update
    
    # Setting up Ju
    J = list()
    Ju_lst = list()
    Jp_lst = list()
    Jw_lst = list()
    Jreg = alpha * np.diag(np.ones((n,)))
    
    for i in range(0, r):
        
        # Assemble values into Ju-mat
        for h in range(0, r):
            if h == i:
                Ju_lst.append(curve.derivatives(ub[i], 1)[1])
            else:
                Ju_lst.append([0] * dim)
        # Setting up Jp
        for j in range(0, n):
            [Bp_unp1, Bp_unp2] = Bp(curve, ub[i], j)
            Jp_lst.extend((Bp_unp1, Bp_unp2))  # Note, possible error in Carlson
        
        # Setting up Jw
        for k in range(0, n):
            Jw_lst.append(weight_der(curve, k, ub[i]))
        
        # check for NAN and replacing with 0
        Jp_np = np.array(Jp_lst)
        logic_nan = np.isnan(Jp_lst)
        Jp_np[logic_nan] = 0
        
        # Assemble the rows
        Ju_row = np.reshape(Ju_lst, (r, 2)).T.tolist()
        Jp_row = np.reshape(Jp_np.tolist(), (2 * n, 2)).T.tolist()
        Jw_row = np.reshape(Jw_lst, (n, 2)).T.tolist()
        
        for i in range(0, dim):
            J.append(Ju_row[i] + Jp_row[i] + Jw_row[i])
                    
        Ju_lst = []
        Jp_lst = []
        Jw_lst = []
    
    for i in range(0, n):
        J.append([0] * (2 * n + r) + Jreg[i].tolist())
        
    return J
    
    
def gn_f(curve, ub, Q, alpha):
    ''' computes the function vector according to Nils Carlson's master thesis
    p. 15
    
    NOT YET TESTED
    
    return: list of function values (least square function!)
    f = [ g1(u1) - g11, g2(u1) - g21, g1(u2) - g12, g2(g2) -g22 ... ]
    And possible regularisation terms!
    '''
    r = len(Q)
    n = len(curve.ctrlptsw)
    f = list()
    W = []
    for elem in curve.ctrlptsw:
        W.append(elem[-1])
        
    for i in range(0, r):
        val = curve.evaluate_single(ub[i]) - Q[i]
        for elem in val:
            f.append(elem)
        
    if isinstance(alpha, float) == True and np.isnan(alpha) != True:
        # isinstance(alpha, np.float64)
        val = (alpha * np.array(np.log(W))).tolist()
        for elem in val:
            f.append(elem)
    else:
        return np.linalg.norm(f) / (r * n * 10)
        
    return f
    
    
def simplebounds(x):
    ''' Adjust the bounds for parameterization, from Nils Carlson algorithm 6.1
    param x: input paramaterization vector [u1, u2, u3 , ..., ur]
    return: bounded parameterization vector 0 <= ui <= 1, list
    '''
    out = list()
    for elem in x:
        if elem < 0:
            out.append(0)
        elif elem > 1:
            out.append(1)
        else:
            out.append(elem)
    return out


def update_curve(curve, x, r):
    ''' Update curve with new values on u, p and w
    
    Implements transformation of weights so that weights always are positive!
    
    param: curve, nurbs curve
    param: x updated x values, list
    param: r, number of test points, int
    '''
    
    dim = len(curve.ctrlptsw[0]) - 1
    n = len(curve.ctrlptsw)
    
    W = x[(r + dim * n):]  # Note dim*n ctrlpts values
    ew = np.exp(W).tolist()
    CP = x[r:(r + dim * n)]
    
    # Make new curve object
    new_curve = copy.deepcopy(curve)
    
    i_old = 0
    CP_list = list()
    for i in range(dim, len(CP) + 1, dim):
        CP_list.append(CP[i_old:i])
        i_old = i
    
    for j in range(0, len(ew)):
        new_curve.ctrlptsw[j] = CP_list[j] + [ew[j]]
    
    return new_curve
    
    
def gauss_newton2D(N_cur, Q, tol=1e-3, mintol=1e-3):
    ''' Forward gaus newton fit of weights, knots, and CP
    
    Inspired by "Surface Fitting with NURBS - a Gauss Newton With Trust Region Approach"
    
    param: CP, Controlpoints value, nd.array[n+1]
    param: W, Weights of the contorl points (=one, if B-spline), nd.array[n+1]
    param: U, knot vector, nd.array[m+1]
    param: ub, parameterization, nd.array[r+1]
    param: Q, Sample data, nparray, Q[r+1]
    param: tol, tolerance wrt jacobian value, float
    param: mintol, tolerance wrt step length, float
    '''
    
    # numbers
    r = len(Q)  # Experiments
    n = len(N_cur.ctrlptsw)  # Ctrlpts, weights
    
    # Dummy values
    J = 1000
    p = 1
    step_length = 1000
    
    # Initialize x
    CP_list = list()
    for CP in N_cur.ctrlptsw:
        for each in CP[:-1]:
            CP_list.append(each)
    W = []
    for elem in N_cur.ctrlptsw:
        W.append(elem[-1])
    ub = assign_ub(Q)  # Should be same as for the B-spline!
    
    x = ub + CP_list + np.log(W).tolist()
    
    # Hard Coded
    x[r] = 0.0
    x[r + 1] = 1.0
    x[r + 2 * n - 2] = 1.0
    x[r + 2 * n - 1] = 0.0
    
    N_cur = update_curve(N_cur, x, r)
    while np.linalg.norm(J * p) > tol and step_length > mintol:
        
        alpha = gn_f(N_cur, ub, Q, np.nan)
        
        #  Calculate J(x) and f(x)
        J = np.array(gn_jacobi(N_cur, ub, Q, alpha))
        f = np.array(gn_f(N_cur, ub, Q, alpha))
        p = (np.linalg.solve(np.matmul(J.T, J), -np.dot(J.T, f))).tolist()
        
        g = 1.0  # Dummy
        f_p = f  # Dummy
        iter = 0
        
        while np.linalg.norm(f_p) >= np.linalg.norm(f) and iter < 100:
            p_g = (np.array(p) * g).tolist()
            x_p = (np.array(x) + np.array(p_g)).tolist()
            
            # Set location of first and last controlpoints as they where in x!
            # x_p[r] = x[r]  # first
            # x_p[r + 2 * n - 2] = x[r + 2 * n - 2]  # last
            
            # Hard Coded
            x_p[r] = 0.0
            x_p[r + 1] = 1.0
            x_p[r + 2 * n - 2] = 1.0
            x_p[r + 2 * n - 1] = 0.0
            
            ub = simplebounds(x_p[0:r])
            New_cur = update_curve(N_cur, x_p, r)
            f_p = gn_f(New_cur, ub, Q, alpha)
            g = g * 0.5
            
            if iter == 99:
                print('100 iterations!')
                # raise ValueError("more than 1000 iteration in gauss_newton2D algorithm")
            iter += 1
            
        New_cur.delta = 0.01
        New_cur.evaluate()
        # vis_comp = vis.VisCurve2D()
        # New_cur.vis = vis_comp
        fig = plt.figure(num=1)
        
        x_plot = []
        y_plot = []
        for elem in New_cur.evalpts:
            x_plot.append(elem[0])
            y_plot.append(elem[1])
            
        plt.plot(x_plot, y_plot)  # Nicer plot
        
        pdb.set_trace()
        # New_cur.render()
        # plt.gcf().clear()
        N_cur = copy.deepcopy(New_cur)
        x = copy.deepcopy(x_p)
            
    return New_cur
    
