import  numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
#from mpl_toolkits import basemap


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

def getcirc(nvec) :
    """
    Solving n.r - n.r0 = 0 and the perpendicular vector to this
    where r0 = 0,0,0
    n is passed to the function
    """
    numpts = 100
    x0 = 1.0
    y0 = 1.0
    z0 = (-x0*nvec[0] - y0*nvec[1])/nvec[2]
    v0 = np.matrix([[x0],[y0],[z0]])
    thetas = np.linspace(0,2*np.pi,numpts,endpoint=False)
    vec0 = np.empty((numpts,3))
    vec1 = np.empty((numpts,3))
    for ii in np.arange(numpts) :
        #In FDG notation quaternion is sin(thet/2)*vec + cos(thet/2) 
        #Scalar is last qty. Rotating v0 about nvec axis in thetas steps
        qt0 = np.sin(thetas[ii]/2.0)*nvec[0]
        qt1 = np.sin(thetas[ii]/2.0)*nvec[1]
        qt2 = np.sin(thetas[ii]/2.0)*nvec[2]
        qt3 = np.cos(thetas[ii]/2.0)
        rmat = quat2dc(np.array((qt0,qt1,qt2,qt3)))
        vt0 = rmat*v0
        vec0[ii,:] = vt0.A1/np.linalg.norm(vt0) #making it a unit vector
    vec1 = np.cross(nvec,vec0) #automatically all will be unit vectors !!
    # Note order of return is important as Roll (nvec) x Pitch (vec0) = Yaw (vec1)
    # and I return yaw, pitch
    return vec1,vec0 


def unit2iner(x,y,z) :
    ra = np.arctan2(y,x)
    if (ra < 0) :
        ra = ra + 2*np.pi
    dec = np.arcsin(z)
    return ra, dec

def iner2dc(alf,delt) :
    roll = np.matrix([[np.cos(alf)*np.cos(delt)], \
            [np.sin(alf)*np.cos(delt)],\
           [np.sin(delt)]])
    return roll

e1alph = 266.58787
e1del = -28.72842
saxalph = 266.76083
saxdel = -28.88303
grsalph = 266.25971
grsdel = -28.91381
ksalph = 266.21275
ksdel = -29.35467
e2alph = 266.52167
e2del = -29.51481
igralph = 266.82525
igrdel = -27.34414

#plt.subplot(111,projection="aitoff")
ax = plt.subplot(111)
#mm1 = basemap.Basemap(projection='stere',llcrnrlat=-27,urcrnrlat=-30,\
#        llcrnrlon=265.8,urcrnrlon=267.2,ax=ax,lat_0=e1del,lon_0=e1alph)
plt.ylim(-30,-27)
plt.xlim(265.8,267.2)
plt.axes().set_aspect('equal', 'datalim')
plt.xlabel("$ lpha $ (RA)")
plt.ylabel("$ \delta $  (dec)")
plt.grid()
plt.plot(e1alph,e1del,'bo')
plt.plot([saxalph,grsalph,ksalph,e2alph,igralph],\
        [saxdel,grsdel,ksdel,e2del,igrdel],'kx')
ylo,yhi = plt.ylim()
offs = (yhi - ylo)*0.05 # 10% of the range
plt.text(saxalph-2*offs,saxdel-offs,'SAX J1747.0-2853')
plt.text(grsalph-2*offs,grsdel-offs,'GRS J1741.9-2853')
plt.text(ksalph-2*offs,ksdel-offs,'KS J1741-293')
plt.text(e2alph-2*offs,e2del-offs,'2E 1742.9-2929')
plt.text(igralph-2*offs,igrdel-offs,'IGR J17473-272')


alfcr = np.deg2rad(83.63)
deltcr = np.deg2rad(22.01)
alf1 = np.deg2rad(83.78)
delt1 = np.deg2rad(22.01)
alf2 = np.deg2rad(83.63)
delt2 = np.deg2rad(22.08)
alf3 = np.deg2rad(83.74)
delt3 = np.deg2rad(22.03)
rollcr = iner2dc(alfcr,deltcr)
vv1,vv2 = getcirc(rollcr.A1)


vec1 = iner2dc(alf1,delt1)
vec2 = iner2dc(alf2,delt2)
vec3 = iner2dc(alf3,delt3)

hdllc = fits.open('./Burst1-lxp2.lc')
tstart1 = hdllc[1].header['TSTART']
tend1 = hdllc[1].header['TSTOP']
hdllc.close()

hdl = fits.open('AS1A03_116T01_9000001460lxp_level1-aux1.att')
idsel = np.where((hdl[1].data['TIME'] >= tstart1) & \
        (hdl[1].data['TIME'] <= tend1))[0]
quart = hdl[1].data['Q_SAT'][idsel]
plt.plot(hdl[1].data['Roll_RA'][idsel][0],hdl[1].data['Roll_DEC'][idsel][0],'k^')
ptra = hdl[1].data['Roll_RA'][idsel][0]
ptdec = hdl[1].data['Roll_DEC'][idsel][0]
hdl.close()


nsm = 2 #quart.shape[0]
ncirc = vv1.shape[0]
alph1 = np.empty(nsm)
alph2 = np.empty(nsm)
alph3 = np.empty(nsm)
del1 = np.empty(nsm)
del2 = np.empty(nsm)
del3 = np.empty(nsm)
d1max = d2max = d3max = 0.0
d1min = d2min = d3min = 1.0
dn = np.empty(ncirc)
for jj in np.arange(ncirc) :
    Ri2b_cr = np.matrix((vv1[jj],rollcr.A1,vv2[jj]))
    vec1_b = Ri2b_cr*vec1
    vec2_b = Ri2b_cr*vec2
    vec3_b = Ri2b_cr*vec3

    for ii in np.arange(nsm) :
        Rb2i = quat2dc(quart[ii]).T # taking the inverse via transpose
        cv1 = Rb2i*vec1_b
        cv2 = Rb2i*vec2_b
        cv3 = Rb2i*vec3_b
        alph1[ii],del1[ii] = unit2iner(cv1[0,0],cv1[1,0],cv1[2,0])
        alph2[ii],del2[ii] = unit2iner(cv2[0,0],cv2[1,0],cv2[2,0])
        alph3[ii],del3[ii] = unit2iner(cv3[0,0],cv3[1,0],cv3[2,0])
        plt.plot(np.rad2deg(alph1[ii]),np.rad2deg(del1[ii]),'r+',markersize=2.0)
        plt.plot(np.rad2deg(alph2[ii]),np.rad2deg(del2[ii]),'g+',markersize=2.0)
        plt.plot(np.rad2deg(alph3[ii]),np.rad2deg(del3[ii]),'m+',markersize=2.0)
    d1 = angdist(alph1[ii-1],del1[ii-1],np.deg2rad(e1alph),np.deg2rad(e1del))
    d2 = angdist(alph2[ii-1],del2[ii-1],np.deg2rad(e1alph),np.deg2rad(e1del))
    d3 = angdist(alph3[ii-1],del3[ii-1],np.deg2rad(e1alph),np.deg2rad(e1del))
    dn[jj] = angdist(alph1[ii-1],del1[ii-1],np.deg2rad(ptra),np.deg2rad(ptdec))
    if (d1 > d1max): d1max = d1
    if (d1 < d1min): d1min = d1
    if (d2 > d2max): d2max = d2
    if (d2 < d2min): d2min = d2
    if (d3 > d3max): d3max = d3
    if (d3 < d3min): d3min = d3




