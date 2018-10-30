import unittest
import numpy as np
from Nurbs import Nurbs
import pdb
import Fitting as F
from geomdl import NURBS
from geomdl import exchange
from geomdl import convert
from geomdl import BSpline
from geomdl.visualization import VisPlotly

'''
Test functions
'''


def Branin_1d(x):
    
    # y = x
    y = 0.5
    
    X1 = 15 * x - 5
    X2 = 15 * y
    a = 1
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    d = 6
    e = 10
    ff = 1 / (8 * np.pi)
    return (a * (X2 - b * X1**2 + c * X1 - d)**2 + e * (1 - ff) * np.cos(X1) + e) + 5 * x


class Test_Nurbs(unittest.TestCase):
    '''
    Some test data
    '''
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self,*args,**kwargs)
        self.U_cube = [0, 0, 0, 0, 1, 5, 6, 8, 8, 8, 8]
        self.p_cube = 3
        self.m_cube = 10
        self.n_cube = 6
        self.Q = np.array([[0, 0], [1, 1], [2, 2], [3, 0], [4, 0], [5, -3], [6, 0], [7, 5]])  # Sample points
        
        self.N_cur, self.B_cur = self.set_curves()
    
    def set_curves(self):
        degree = 3
        num_cp = 4
        CP, U = F.curve_fit2D(self.Q, np.ones((len(self.Q), 1)), None, None, None, num_cp, degree)
        
        B_cur = BSpline.Curve()
        B_cur.degree = degree
        B_cur.ctrlpts = CP
        B_cur.knotvector = U
        B_cur.evaluate()
        B_cur.delta = 0.01
        
        N_cur = convert.bspline_to_nurbs(B_cur)
        
        # Set evaluation delta
        N_cur.delta = 0.01
        N_cur.evaluate()
        
        return N_cur, B_cur
    
    def assertListAlmostEqual(self, list1, list2, tol):
        self.assertEqual(len(list1), len(list2))
        for i in range(0, len(list1)):
            for a, b in zip(list1[i], list2[i]):
                self.assertAlmostEqual(a, b, tol)
    
    def test_FindSpan(self):
        self.assertTrue(Nurbs.FindSpan(6, 3, 0, self.U_cube) == 3)
        self.assertTrue(Nurbs.FindSpan(6, 3, 8, self.U_cube) == 6)
        self.assertTrue(Nurbs.FindSpan(6, 3, 3, self.U_cube) == 4)
        self.assertTrue(Nurbs.FindSpan(6, 3, 5.5, self.U_cube) == 5)
        self.assertTrue(Nurbs.FindSpan(6, 3, 7.9, self.U_cube) == 6)
        
    def test_assign_ub(self):
        ub = F.assign_ub(self.Q)
        # Just check if the output is correct (and that the functions actually works).
        self.assertTrue(isinstance(ub, np.ndarray))
        self.assertTrue(isinstance(F.set_knots(4, 2, len(self.Q), ub), np.ndarray))

    def test_fitting(self):
        CP, U = F.curve_fit2D(self.Q, np.ones((len(self.Q), 1)), None, None, None, 4, 2)
        self.assertTrue(isinstance(CP, tuple))
        self.assertTrue(isinstance(U, list))
    
    def test_gn_jacobi(self):
        J = F.gn_jacobi(self.N_cur, F.assign_ub(self.Q), self.Q, 1.0)
    
        r = len(self.Q)
        n = len(self.N_cur.ctrlptsw)
        
        (rows, cols) = np.shape(np.array(J))
        pdb.set_trace()
        self.assertTrue(rows == (2 * r + n))
        self.assertTrue(cols == (r + 3 * n))
        self.assertTrue(isinstance(J, list))
        
        # Write some test to test for sanity check of the contents?!
    
    def test_gn_f(self):
        alpha = F.gn_f(self.N_cur, F.assign_ub(self.Q), self.Q, np.nan)
        fun_val = F.gn_f(self.N_cur, F.assign_ub(self.Q), self.Q, alpha)
        
        r = len(self.Q)
        n = len(self.N_cur.ctrlptsw)
        (rows,) = np.shape(np.array(fun_val))
        self.assertTrue(rows == (2 * r + n))
        self.assertTrue(isinstance(alpha, float))
        self.assertTrue(isinstance(fun_val, list))
    
    def test_NURBS_BSPLINE_EQUAL(self):
        # Save results
        B_res = self.B_cur.evalpts
        N_res = self.N_cur.evalpts
        self.assertListAlmostEqual(B_res, N_res, 3)
    
    # def test_gauss_newton2D(self):
    #     ''' Performs gauss newton search of optimum weights
    #     '''
    # 
    #     # Fit the NURBS weights and control points
    #     B_fit = F.gauss_newton2D(self.N_cur, self.Q)
    # 
    #     # Plot (debug)
    #     vis_fit = VisPlotly.VisCurve2D()
    #     B_fit.vis = vis_fit
    #     B_fit.render()
