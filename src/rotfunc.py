"""
Set of commonly used rotation functions
Made on Sat 23 Dec 2017 16:38:49 PM IST for use with 1E_G3 pointing 
"""
import numpy as np

def iner2dc(alf,delt) :
    roll = np.matrix([[np.cos(alf)*np.cos(delt)], \
            [np.sin(alf)*np.cos(delt)],\
           [np.sin(delt)]])
    return roll

def unit2iner(x,y,z) :
    ra = np.arctan2(y,x)
    if (ra < 0) :
        ra = ra + 2*np.pi
    dec = np.arcsin(z)
    return ra, dec

#def unit2iner(vec) :
#   return unit2iner(vec[0],vec[1],vec[2])

def angdist(alph1,delt1,alph2,delt2) :
    fac = np.sin(delt1)*np.sin(delt2) + np.cos(delt1)*np.cos(delt2)*np.cos(alph1 - alph2)
    thet = np.arccos(fac)
    return thet

def quat2dc(q) :
    dq1 = q[0]**2
    dq2 = q[1]**2
    dq3 = q[2]**2
    dq4 = q[3]**2
    dq12 = 2*q[0]*q[1]
    dq13 = 2*q[0]*q[2]
    dq14 = 2*q[0]*q[3]
    dq23 = 2*q[1]*q[2]
    dq24 = 2*q[1]*q[3]
    dq34 = 2*q[2]*q[3]

    mat = np.empty((3,3))
    mat[0][0] = dq1 - dq2 - dq3 + dq4
    mat[0][1] = dq12 + dq34
    mat[0][2] = dq13 - dq24

    mat[1][0] = dq12 - dq34
    mat[1][1] = -dq1 + dq2 -dq3 + dq4
    mat[1][2] = dq23 + dq14

    mat[2][0] = dq13 + dq24
    mat[2][1] = dq23 - dq14
    mat[2][2] = -dq1 -dq2 + dq3 + dq4

    return mat

def angvec2quat(vec,ang) :
  """
  Function to return quaternion (last element cos(alf)) given the vector and
  angle
  """
  quat = np.empty(4)
  quat[0] = np.sin(ang/2)*vec[0]
  quat[1] = np.sin(ang/2)*vec[1]
  quat[2] = np.sin(ang/2)*vec[2]
  quat[3] = np.cos(ang/2)
  return quat


# Also add quaternion multiplication and rotation functions !
# Note the notation - quaternions have angle as last element, RotMat are row
# wise d.c.s of F2 (x,y,z axes) in F1 coord and RotMat*Vec gives transformation from F1 to
# F2. Note that since RotMat is transpose inverse, column elements of RotMat are
# d.cs of F1 (x,y,z) in F2 coord.
