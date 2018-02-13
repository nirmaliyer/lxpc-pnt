from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

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

def unit2iner(x,y,z) :
    ra = np.arctan2(y,x)
    if (ra < 0) :
        ra = ra + 2*np.pi
    dec = np.arcsin(z)
    return ra, dec




hdl = fits.open('AS1A03_116T01_9000001460lxp_level1-aux1.att')
idx =234
quart = hdl[1].data['Q_SAT'][idx]
alf = np.deg2rad(hdl[1].data['Roll_RA'][idx])
delt = np.deg2rad(hdl[1].data['Roll_DEC'][idx])
rotn = np.deg2rad(hdl[1].data['Roll_ROT'][idx])

p = np.empty(4)
p[0] = np.cos(rotn/2.)
p[1] = np.cos(alf)*np.cos(delt)*np.sin(rotn/2.)
p[2] = np.sin(alf)*np.cos(delt)*np.sin(rotn/2.)
p[3] = np.sin(delt)*np.sin(rotn/2.)

dc = quat2dc(quart)
rollra, rolldec = unit2iner(dc[1][0],dc[1][1], dc[1][2])







