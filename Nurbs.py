import numpy as np
import math
import pdb

class Nurbs:
    def __init__(self, x, y,):
        self.x = x
        self.y = y
        
        if type(x) == np.ndarray:
            self.Nd = x.size  # Number of data point
        elif type(x) == int:
            self.Nd = 1
            
        # def CurvePoint(n, p, U, Pw, u, C):
            # '''
            # Compute point on rational B-spline curve
            # Input: n,p,U,Pw,u
            # Output: C
            # '''
            # 
            # span = FindSpan(n,p,u,U);
            # BasisFuns(span,u,p,U,N);
            # Cw = 0.0;
            # for (j=O; j<=p; j++)
            # Cw = Cw + N[j]*Pw[span-p+j];
            # C = Cw/w; /* Divide by weight */
            # }
            
    def FindSpan(n, p, u, U):
        ''' p 68 in Book of nurbs
        Determine the knot span index
        Input: n,p,u,U
        Return: the knot span index (Int)
        
        The NURBS Book states that the knot span index always starts from zero, i.e. for a knot vector [0, 0, 1, 1];
        if FindSpan returns 1, then the knot is between the interval [0, 1). i.e. 0<= u < 1
        '''
        if (u == U[n + 1]):
            return(n)  # Special case
        
        low = p
        high = n + 1  # Do binary search
        mid = int(np.floor(low + high) / 2)
        while u < U[mid] or u >= U[mid + 1]:
            if (u < U[mid]):  # Move upper bound down
                high = mid
            else:
                low = mid  # Move lower bound down
            mid = int(np.floor(low + high) / 2)
        
        return mid
        
            
            # BasisFuns(i,u,p,U,N)
            # ''' ALGORITHM A2.2, p 70 in Book of nurbs 
            # {/* Compute the nonvanishing basis functions */
            # /* Input: i,u,p,U */
            # /* Output: N */
            # '''
            # N[0]=1.0;
            # for (j=1; j<=p; j++)
            # }
            # {
            # left[j] = u-U[i+1-j];
            # right[j] = U[i+j]-u;
            # saved = 0.0;
            # for (r=O; r<j; r++)
            # {
            # temp = N[r]/(right[r+1]+left[j-r]);
            # N[r] = saved+right[r+1]*temp;
            # saved = left[j-r]*temp;
            # }
            # N[j] = saved;
            # }    
            
