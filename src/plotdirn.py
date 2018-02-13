import  numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from rotfunc import *

folfits = "../../"

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

fth = fits.open(folfits + '4thorbit-AS1T01_052T01_9000000316lxp_level1.mkf') # RA Scan
sth = fits.open(folfits + '6thorbit-AS1T01_052T01_9000000316lxp_level1.mkf') # Dec Scan
fiid = np.arange(5120,10120) #indices for RA scan 4th orbit
qtyf = np.abs(1.0*fth[1].data['Roll_RA'][fiid] - 83.63)
idxf = np.where(qtyf == qtyf.min())[0][0] + fiid[0]

siid = np.arange(7555,12940) #indices for RA scan 4th orbit
qtys = np.abs(1.0*sth[1].data['Roll_DEC'][siid] - 22.01)
idxs = np.where(qtys == qtys.min())[0][0] + siid[0]

rrotn = 0.5*fth[1].data['Roll_ROT'][idxf] + 0.5*sth[1].data['Roll_ROT'][idxs]



qrrotn = angvec2quat(rollcr.A1,np.deg2rad(rrotn))
rrotnmat = quat2dc(qrrotn)
yawvect = np.matrix([[1.0],[-rollcr[0,0]/rollcr[1,0]],[0.0]])
yrotn = rrotnmat.T*yawvect
yawvec = yrotn/np.linalg.norm(yrotn)
pitvec = np.cross(yawvec.A1,rollcr.A1)
Ri2b_cr = np.matrix((yawvec.A1,rollcr.A1,pitvec))

vec1_b = Ri2b_cr*vec1
vec2_b = Ri2b_cr*vec2
vec3_b = Ri2b_cr*vec3

hdllc = fits.open(folfits + 'Burst1-lxp2.lc')
tstart1 = hdllc[1].header['TSTART']
tend1 = hdllc[1].header['TSTOP']
hdllc.close()

hdllc2 = fits.open(folfits + 'Burst2-lxp2.lc')
tstart2 = hdllc2[1].header['TSTART']
tend2 = hdllc2[1].header['TSTOP']
hdllc2.close()


hdl = fits.open(folfits + 'AS1A03_116T01_9000001460lxp_level1-aux1.att')
idsel = np.where((1.0*hdl[1].data['TIME'] >= tstart1) & \
        (1.0*hdl[1].data['TIME'] <= tend1))[0]
idsel2 = np.where((1.0*hdl[1].data['TIME'] >= tstart2) & \
        (1.0*hdl[1].data['TIME'] <= tend2))[0]
plt.plot(1.0*hdl[1].data['Roll_RA'][idsel][0],1.0*hdl[1].data['Roll_DEC'][idsel][0],'k^',markersize=8)
plt.plot(1.0*hdl[1].data['Roll_RA'][idsel2][0],1.0*hdl[1].data['Roll_DEC'][idsel2][0],'^',color='gray',markersize=8)
quart = 1.0*hdl[1].data['Q_SAT'][idsel]
quart2 = 1.0*hdl[1].data['Q_SAT'][idsel2]
nsm = quart2.shape[0] if (quart2.shape[0] > quart.shape[0]) else quart.shape[0]
# take the bigger size

alph1 = np.empty(nsm)
alph2 = np.empty(nsm)
alph3 = np.empty(nsm)
del1 = np.empty(nsm)
del2 = np.empty(nsm)
del3 = np.empty(nsm)

plt.axes().set_aspect('equal', 'datalim')
plt.ylim(-30,-27)
plt.xlim(265.8,267.2)
plt.xlabel(r" $\alpha$ (RA)",fontsize=15)
plt.ylabel(r" $\delta$ (dec)",fontsize=15)
plt.tick_params(labelsize=15)
plt.grid()
plt.plot(e1alph,e1del,'bo',markersize=7)
plt.plot([saxalph,grsalph,ksalph,e2alph,igralph],\
        [saxdel,grsdel,ksdel,e2del,igrdel],'ko',markersize=7)
ylo,yhi = plt.ylim()
offs = (yhi - ylo)*0.05 # 10% of the range
plt.text(266.1, e1del,'1E 1743.1-2843')
plt.text(saxalph + 0.8*offs,saxdel,'SAX J1747.0-2853')
plt.text(grsalph-2*offs,grsdel-offs,'GRS J1741.9-2853')
plt.text(ksalph-2*offs,ksdel-offs,'KS J1741-293')
plt.text(e2alph-2*offs,e2del-offs,'2E 1742.9-2929')
plt.text(igralph-2*offs,igrdel-offs,'IGR J17473-272')

for ii in np.arange(nsm) :
    if ( ii < quart.shape[0]) :
       Rb2i = quat2dc(quart[ii]).T # taking the inverse via transpose
       cv1 = Rb2i*vec1_b
       cv2 = Rb2i*vec2_b
       cv3 = Rb2i*vec3_b
       alph1[ii],del1[ii] = unit2iner(cv1[0,0],cv1[1,0],cv1[2,0])
       alph2[ii],del2[ii] = unit2iner(cv2[0,0],cv2[1,0],cv2[2,0])
       alph3[ii],del3[ii] = unit2iner(cv3[0,0],cv3[1,0],cv3[2,0])
       plt.plot(np.rad2deg(alph1[ii]),np.rad2deg(del1[ii]),'r+',markersize=10)
       plt.plot(np.rad2deg(alph2[ii]),np.rad2deg(del2[ii]),'g+',markersize=10)
       plt.plot(np.rad2deg(alph3[ii]),np.rad2deg(del3[ii]),'m+',markersize=10)
    if ( ii < quart2.shape[0]) :
       Rb2i = quat2dc(quart2[ii]).T # taking the inverse via transpose
       cv1 = Rb2i*vec1_b
       cv2 = Rb2i*vec2_b
       cv3 = Rb2i*vec3_b
       alph1[ii],del1[ii] = unit2iner(cv1[0,0],cv1[1,0],cv1[2,0])
       alph2[ii],del2[ii] = unit2iner(cv2[0,0],cv2[1,0],cv2[2,0])
       alph3[ii],del3[ii] = unit2iner(cv3[0,0],cv3[1,0],cv3[2,0])
       plt.plot(np.rad2deg(alph1[ii]),np.rad2deg(del1[ii]),'+',color='gray',markersize=10)
       plt.plot(np.rad2deg(alph2[ii]),np.rad2deg(del2[ii]),'+',color='gray',markersize=10)
       plt.plot(np.rad2deg(alph3[ii]),np.rad2deg(del3[ii]),'+',color='gray',markersize=10)
       


