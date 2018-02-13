import  numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
#import 


#V.V. Interesting to note bug in numpy 
# causing (Rb2i*Ri2b)*vec1 not equal to Rb2i*(Rieb*vec1) - 
# Same calculation works fine in root but not in python 2.7 and python 3.5

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



alfcr = np.deg2rad(83.63)
deltcr = np.deg2rad(22.01)
alf1 = np.deg2rad(83.78)
delt1 = np.deg2rad(22.01)
alf2 = np.deg2rad(83.63)
delt2 = np.deg2rad(22.08)
alf3 = np.deg2rad(83.74)
delt3 = np.deg2rad(22.03)

rollcr = iner2dc(alfcr,deltcr)
vec1 = iner2dc(alf1,delt1)
vec2 = iner2dc(alf2,delt2)
vec3 = iner2dc(alf3,delt3)
#placing the spacecraft yaw and pitch - taking 316 for now ! 
# Is wrong must do the 360 !

hdlp = fits.open('./Archive/AS1T01_052T01_9000000316lxp_level1.mkf')
qty = np.abs(hdlp[1].data['Roll_RA'] - 83.63)
idx = np.where(qty == qty.min())[0][0]
quartcr = hdlp[1].data['Q_SAT'][idx]
ptcra = hdlp[1].data['Roll_RA'][idx]
ptcdec = hdlp[1].data['Roll_DEC'][idx]
Ri2b_cr = quat2dc(quartcr)
vec1_b = Ri2b_cr*vec1
vec2_b = Ri2b_cr*vec2
vec3_b = Ri2b_cr*vec3

hdlp.close()

hdllc = fits.open('./Burst1-lxp2.lc')
tstart1 = hdllc[1].header['TSTART']
tend1 = hdllc[1].header['TSTOP']
hdllc.close()

hdl = fits.open('AS1A03_116T01_9000001460lxp_level1-aux1.att')
idsel = np.where((hdl[1].data['TIME'] >= tstart1) & \
        (hdl[1].data['TIME'] <= tend1))[0]
ptra = hdl[1].data['Roll_RA'][idsel][0]
ptdec = hdl[1].data['Roll_DEC'][idsel][0]
quart = hdl[1].data['Q_SAT'][idsel]
hdl.close()

#nsm = quart.shape[0]
nsm = 1 # trying now for 1st 3000 rows only
alph1 = np.empty(nsm)
alph2 = np.empty(nsm)
alph3 = np.empty(nsm)
del1 = np.empty(nsm)
del2 = np.empty(nsm)
del3 = np.empty(nsm)

#plt.axis('equal')
plt.ylim(-30,-27)
plt.xlim(265.8,267.2)
plt.axes().set_aspect('equal', 'datalim')
plt.xlabel("$ lpha $ (RA)")
plt.ylabel("$ \delta $  (dec)")
plt.grid()
plt.plot(ptra,ptdec,'k^')
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


for ii in np.arange(nsm) :
    Rb2i = quat2dc(quart[ii]).T # taking the inverse via transpose
    cv1 = Rb2i*vec1_b
    cv2 = Rb2i*vec2_b
    cv3 = Rb2i*vec3_b
    alph1[ii],del1[ii] = unit2iner(cv1[0,0],cv1[1,0],cv1[2,0])
    alph2[ii],del2[ii] = unit2iner(cv2[0,0],cv2[1,0],cv2[2,0])
    alph3[ii],del3[ii] = unit2iner(cv3[0,0],cv3[1,0],cv3[2,0])
    plt.plot(np.rad2deg(alph1[ii]),np.rad2deg(del1[ii]),'r+')
    plt.plot(np.rad2deg(alph2[ii]),np.rad2deg(del2[ii]),'g+')
    plt.plot(np.rad2deg(alph3[ii]),np.rad2deg(del3[ii]),'m+')


d1pre = angdist(alph1[0],del1[0],np.deg2rad(ptra),np.deg2rad(ptdec))
d2pre = angdist(alph2[0],del2[0],np.deg2rad(ptra),np.deg2rad(ptdec))
d3pre = angdist(alph3[0],del3[0],np.deg2rad(ptra),np.deg2rad(ptdec))
d1pos = angdist(alf1,delt1,np.deg2rad(ptcra),np.deg2rad(ptcdec))
d2pos = angdist(alf2,delt2,np.deg2rad(ptcra),np.deg2rad(ptcdec))
d3pos = angdist(alf3,delt3,np.deg2rad(ptcra),np.deg2rad(ptcdec))
