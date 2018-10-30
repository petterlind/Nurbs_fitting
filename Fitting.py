import numpy as np
import pdb
import geomdl
from geomdl import helpers
import copy



def LUDecomposition(A, Q, sbw):
    ''' Utility function, p 369
    
    Not neccecary
    
    To decompose the q*q coefficient matrix with semibandwidth sbw into lower
    and upper triangular components; for simplicity we assume A is an q*q square
    array, but a utility should be used which only stores the nonzero band.
'''
    pass
    
    
def ForwardBackward(A, q, sbw, rhs, sol):
    ''' Utility function p.369
    
    Not neccecary
    
    To perform the forward/backward substitution (see [pres88]); rhs[] is the
    right hand side of the system (the coordinates of the Q_k), and sol[] is the
    solution vectir (coordinates of the P_i)
    '''
    pass
    
    
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
    return ub


def set_knots(n, p, m, ub):
    ''' Creates an internal knot knotvector
    
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
    : param r: upper index of Q[]
    : type r: int
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
    
    # Create two new curves, one with change of x, another with change of y
    curve_dx = copy.deepcopy(curve)
    curve_dy = copy.deepcopy(curve)
    
    # Set forward finite disturbance to 1%
    dx = curve_dx.ctrlptsw[j][0] * 1.01 - curve_dx.ctrlptsw[j][0]
    dy = curve_dy.ctrlptsw[j][1] * 1.01 - curve_dy.ctrlptsw[j][1]
    
    # Change the values
    curve_dx.ctrlptsw[j][0] += dx
    curve_dy.ctrlptsw[j][1] += dy
    
    # Function value at nominal point
    fun_val = curve.evaluate_single(ul)
    
    # Function value with change of px
    fun_dx = curve_dx.evaluate_single(ul)
    
    # Function value with change of py
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
        
    dw = W[wi] * 0.02
    W[wi] += dw
    
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
    
    # Transformed weights
    W = []
    for elem in curve.ctrlptsw:
        W.append(elem[-1])
    ew = np.exp(W)  # Transformed weights (not really necceccary here or?)
    
    U = curve.knotvector
    
    # Indexing
    n = len(W)
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
        
        for j in range(0, n):
            # Setting up Jp
            [Bp_unp1, Bp_unp2] = Bp(curve, ub[i], j)
            Jp_lst.extend((Bp_unp1, Bp_unp2))  # Note, possible error in Carlson
            
        for k in range(0, n):
            # Setting up Jw
            Jw_lst.append(weight_der(curve, k, ub[i]))
        
        # Assemble the rows
        Ju_row = np.reshape(Ju_lst, (r, 2)).T.tolist()
        Jp_row = np.reshape(Jp_lst, (2 * n, 2)).T.tolist()
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
        val = (alpha * np.array(W)).tolist()
        for elem in val:
            f.append(elem)
    else:
        return np.linalg.norm(f) / n
        
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


def update_curve(curve, x_p):
    ''' Update curve with new values on u, p and w
    
    Implements transformation of weights so that weights always are positive!
    '''
    New_cur = copy.deepcopy(curve)
    
    ew = np.exp(W)  # Transformed weights (not really necceccary here or?)
    raise NotImplementedError("This functionality is not implemented at the moment")
    
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
    
    # Numbers
    n = len(CP)
    m = len(W)  # length of knot vector
    r = len(Q)  # Number of experiments
    
    # Order of the base functions
    p = m - n - 1
    
    ub = assign_ub(Q)  # Should be same as for the B-spline!
    
    # Dummy values
    J = 1000
    p = 1
    step_length = 1000
    
    while np.linalg.norm(J * p) > tol or step_length > mintol:
        
        
        # U, CP and W
        CP_list = list()
        for CP in N_cur.ctrlptsw:
            for each in CP[:-1]:
                CP_list.append(each)
        
        # Transformed weights
        W = []
        for elem in N_cur.ctrlptsw:
            W.append(elem[-1])
        
        x = ub + W + CP_list
        
        alpha = gn_f(N_cur, ub, Q, np.nan)
        
        #  Calculate J(x) and f(x)
        J = np.ndarray(gn_jacobi(N_cur, ub, Q, alpha))
        f = np.ndarray(gn_f(N_cur, ub, Q, alpha))
        
        p = np.linalg.solve(np.dot(J.T, J), -np.dot(J, f))
        g = 1
        f_p = f  # Dummy
        
        while np.linalg.norm(f_p) >= np.linalg.norm(f) and iter < 1000:
            p_g = p * g
            x_p = x + p_g
            x_p[0:r] = simplebounds(x_p[0:r])
            
            New_cur = update_curve(N_cur, x_p)
            f_p = gn_f(New_cur, ub, Q, alpha)
            g = g * 0.5
            
            if iter == 1000:
                raise ValueError("more than 1000 iteration in gauss_newton2D algorithm")
        
        # Set the new curve as the best fit!
        N_cur = copy.deepcopy(New_cur)
        
    return N_cur
    
