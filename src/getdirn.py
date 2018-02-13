import  numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit,fsolve 
from scipy.interpolate import InterpolatedUnivariateSpline as ip
from scipy.integrate import quad
from rotfunc import *

folfits="../../"

def effarea(lxpc,elo=3.0,ehi=15.0) :
  """
  Gets integrated eff. area between 3 to 80 keV
  """
  fname=folfits + "effarea_lxpc.txt"
  ener,area = np.genfromtxt(fname,unpack=True)
  idnan = np.where(np.isnan(ener))[0]
  if (lxpc == 1) :
    idpc = np.arange(0,idnan[0])
  if (lxpc == 2) :
    idpc = np.arange(idnan[0]+1,idnan[1])
  if (lxpc == 3) :
    idpc = np.arange(idnan[1]+1,idnan[2])
  fn = ip(ener[idpc],area[idpc],k=1)
  #intar = quad(fn,3.0,31.90)[0] + quad(fn,32.5,80.0)[0] # breaking integral at 32 keV discontinuity
  intar = quad(fn,elo,ehi) # no use breaking the func - see what prblm
  geomar = 3600 # cm^2 100cm x 36 cm (and 15 cm depth - see http://www.tifr.res.in/~astrosat_laxpc/specification.html)
  return intar[0]/(geomar*(ehi - elo))


def collimfunc(delthetra,delthetdec,lxpc=1,ra0=83.63,dec0=22.01,full=False) :
    """
    Read LAXPC collimator response from files and give response (max unity) for
    given offset angles theta and phi
    """
    #print "delthetra = " 
    #print delthetra
    #print "delthetdec = "
    #print delthetdec

    fnamera = folfits + "LX" + str(lxpc) + "0_RA.txt"
    fnamedec = folfits + "LX" + str(lxpc) + "0_Dec.txt"

    tlpcr,raval = np.genfromtxt(fnamera,skip_header=1,unpack=True)
    ipr = ip(tlpcr,raval,k=2)
    drr = lambda z : ipr.derivatives(z)[1]
    tlrmax = fsolve(drr,16879)[0] # Note that I am using my maximas instead of
    #the ones given in Table 2 of Antia's paper. Cursory examination shows
    #difference to be minimal - must shift to Table 2 vals soon
    tlpcd,decval = np.genfromtxt(fnamedec,skip_header=1,unpack=True)
    ipd = ip(tlpcd,decval,k=2)
    ddr = lambda z : ipd.derivatives(z)[1]
    tldmax = fsolve(ddr,29226)[0]

    tra,ra = np.genfromtxt(folfits + 'rascan.txt',skip_header=1,unpack=True)
    tdec,dec = np.genfromtxt(folfits + 'decscan.txt',skip_header=1,unpack=True)
    poltra = np.polyfit(ra-ra0,tra,1)
    poltdec = np.polyfit(dec-dec0,tdec,1)#Inverting the variables for linfit
    fra = np.poly1d(poltra)
    fdec = np.poly1d(poltdec)

    timra = fra(np.rad2deg(delthetra))
    timdec = fdec(np.rad2deg(delthetdec))
    if ( np.any(timra > tra.max()) | np.any(timra < tra.min()) | \
        np.any(timdec > tdec.max()) | np.any(timdec < tdec.min()) ) :
      raise ValueError("Angle out of FOV")
    if (full) :
        dramax = (tlrmax - poltra[1])/poltra[0] # using inverse polynom funcn
        ddecmax = (tldmax - poltdec[1])/poltdec[0]
        return ipr(timra)/ipr(tlrmax)*ipd(timdec)/ipd(tldmax), dramax, ddecmax

    return ipr(timra)/ipr(tlrmax)*ipd(timdec)/ipd(tldmax)


def getangle(pntvec,offsvec) :
    p1 = 1.0*pntvec
    o1 = 1.0*offsvec
    p2 = 1.0*pntvec
    o2 = 1.0*offsvec
    p1[0] = 0.0 # projn on yz plane to get angle perpendicular to x axis
    p1 = p1/np.linalg.norm(p1)
    o1[0] = 0.0
    o1 = o1/np.linalg.norm(o1)
    p2[1] = 0.0 # projn on xz plane to get angle perpendicular to y axis
    p2 = p2/np.linalg.norm(p2)
    o2[1] = 0.0
    o2 = o2/np.linalg.norm(o2)
    ang1 = np.arccos(np.clip(np.dot(p1,o1),-1.0,1.0))
      # picked up the clip trick from stackexcahnge for preventing bit precision err from causing nan return
    if (np.cross(p1,o1)[0] < 0) :
      ang1 = -ang1
    ang2 = np.arccos(np.clip(np.dot(p2,o2),-1.0,1.0))
    if (np.cross(p2,o2)[1] < 0) :
      ang2 = -ang2
    return ang1,ang2


def fitlicu(time,A,tau,t0,ra,dec,Ri2carr,pntvec) :
    """
    Function to fit the lightcurve and get ra and dec of source
    """
    nsingle = int(time.size/3) # For each LAXPC
    orig = A*np.exp(-(time[:nsingle] - t0)/tau) #Assuming time for all LAXPCs are same
    offvec = iner2dc(ra,dec) # undecided on whether to keep ra/dec in deg or rad.
    # does keeping in deg give more area to search for algo and make it easier to
    # converge ?
    thet1 = np.empty(nsingle) #In YZ plane perp. to x axis
    thet2 = np.empty(nsingle) #In XZ plane perp. to y axis of ccsc
    for ii in np.arange(nsingle) :
      offvec_c = np.matrix(Ri2carr[ii])*offvec
      ptpnt_c = np.matrix(Ri2carr[ii])*np.matrix(pntvec[ii]).T
      thet1[ii],thet2[ii] = getangle(ptpnt_c.A1,offvec_c.A1)
    
    lx1 = orig*effarea(1)*collimfunc(thet1,thet2,1)
    lx2 = orig*effarea(2)*collimfunc(thet1,thet2,2)
    lx3 = orig*effarea(3)*collimfunc(thet1,thet2,3)
    retarr = np.reshape((lx1,lx2,lx3),-1) # append to 1d array
    return retarr,thet1,thet2



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

#getting Ri2b_cr other way of using Roll Rotation avg of 4th and 6th orbit
fth = fits.open(folfits + '4thorbit-AS1T01_052T01_9000000316lxp_level1.mkf') # RA Scan
sth = fits.open(folfits + '6thorbit-AS1T01_052T01_9000000316lxp_level1.mkf') # Dec Scan
fiid = np.arange(5120,10120) #indices for RA scan 4th orbit
qtyf = np.abs(1.0*fth[1].data['Roll_RA'][fiid] - 83.63)
idxf = np.where(qtyf == qtyf.min())[0][0] + fiid[0]
qf = 1.0*fth[1].data['Q_SAT'][idxf] # multiplying 1.0 to get 64 bit precision.
Rf = quat2dc(qf)
vrs = np.array([0.0,0.0,1.0]) #x axis of collimator camera scan coordinate
# (ccsc) = axis perpendicular to ra scan
#vrs = Rf*np.matrix(vrsi).T

siid = np.arange(7555,12940) #indices for DEC scan 6th orbit
qtys = np.abs(1.0*sth[1].data['Roll_DEC'][siid] - 22.01)
idxs = np.where(qtys == qtys.min())[0][0] + siid[0]
rdid = idxs + 1069 # some random id in DEC scan
qs = 1.0*fth[1].data['Q_SAT'][idxs]
Rs = quat2dc(qs)
vd1 = iner2dc(np.deg2rad(1.0*sth[1].data['Roll_RA'][idxs]),np.deg2rad(1.0*sth[1].data['Roll_DEC'][idxs]))
vd2 = iner2dc(np.deg2rad(1.0*sth[1].data['Roll_RA'][idxs]),np.deg2rad(1.0*sth[1].data['Roll_DEC'][rdid]))
#Manipulating the scan to be along orthogonal RA and DEC axes !
vdsi = np.cross(vd1.A1,vd2.A1) # y axis of ccsc
vdsi[2] = 0.0 #Manipulating the scan to be along orthogonal RA and DEC axes !
vds = vdsi/np.linalg.norm(vdsi)
#vds = Rs*np.matrix(vdsi).T Not shifting to body as scans are not perpendicular
#in body coordinates !


rrotn = 0.5*fth[1].data['Roll_ROT'][idxf] + 0.5*sth[1].data['Roll_ROT'][idxs]
# Roll rotation is angle of rotation about the roll axis between xaxis (yaw)
# of satellite/body coordinates and the vector which lies both in xy plane of inertial
# and yaw-pitch plane of body coordinates. Thus it is perpendicular to roll
# vector (of body) and z axis (of inertial system)



vzs = np.cross(vrs,vds) # z axis of ccsc
Ri2c_cr = np.matrix((vrs,vds,vzs)) 


qrrotn = angvec2quat(rollcr.A1,np.deg2rad(rrotn))
rrotnmat = quat2dc(qrrotn)
yawvect = np.matrix([[1.0],[-rollcr[0,0]/rollcr[1,0]],[0.0]])
yrotn = rrotnmat.T*yawvect
yawvec = yrotn/np.linalg.norm(yrotn)
pitvec = np.cross(yawvec.A1,rollcr.A1)
Ri2b_cr = np.matrix((yawvec.A1,rollcr.A1,pitvec))
#Ri2b_cr = quat2dc(fth[1].data['Q_SAT'][idxf])
Rb2c = Ri2b_cr.T*Ri2c_cr #camera to body conversion matrix - important and
#fixed matrix irrespective of pointing

vec1_b = Ri2b_cr*vec1
vec2_b = Ri2b_cr*vec2
vec3_b = Ri2b_cr*vec3


hdllc = fits.open(folfits + 'Burst1-lxp2.lc')
tstart1 = hdllc[1].header['TSTART']
tend1 = hdllc[1].header['TSTOP']
hdllc.close()

hdl = fits.open(folfits + 'AS1A03_116T01_9000001460lxp_level1-aux1.att')
idsel = np.where((1.0*hdl[1].data['TIME'] >= tstart1) & \
        (1.0*hdl[1].data['TIME'] <= tend1))[0]
quart = 1.0*hdl[1].data['Q_SAT'][idsel]

l1bg = 440.0 # cts/s approx from looking at preburst licu
l2bg = 350.0 
l3bg = 225.0

hdllc = fits.open(folfits + "test1.lc")
tdel = hdllc[1].header['TIMEDEL']*86400.0 # in seconds
time = 1.0*hdllc[1].data['TIME'] + (1.0*hdllc[1].header['TIMEZERI'] + \
    hdllc[1].header['TIMEZERF'] - hdl[0].header['MJDREFI'] + 40000)*86400. # in sec since launch
l1 = hdllc[1].data['RATE1']*tdel - l1bg*tdel
e1 = hdllc[1].data['ERROR1']*tdel
l2 = hdllc[1].data['RATE2']*tdel - l2bg*tdel
e2 = hdllc[1].data['ERROR2']*tdel
l3 = hdllc[1].data['RATE3']*tdel - l3bg*tdel
e3 = hdllc[1].data['ERROR3']*tdel
tmatt = hdl[1].data['TIME'][idsel]

t0 = time[0]
tau = 4.6 # from fplot fit 
#A = 660.0 - 1eval
A = 960.0 


tmarr = np.reshape((time,time,time),-1)
nn = time.size
Ri2carr = np.empty((time.size,3,3))
pntvec = np.empty((time.size,3))
for ii in np.arange(time.size) :
    idd = np.where(tmatt <= (time[ii] + tdel))[0][-1] # taking last att value in that time window
    Ri2b = quat2dc(quart[idd])
    Ri2b[0] = -Ri2b[0] # inverting yaw and pitch axes by 180 deg.
    Ri2b[2] = -Ri2b[2]
    pntvec[ii] = Ri2b[1] # roll vector
    Ri2carr[ii] = Ri2b*Rb2c

ltarr,t1,t2 = fitlicu(tmarr,A,tau,t0,np.deg2rad(e1alph),np.deg2rad(e1del),Ri2carr,pntvec)
plt.figure('1E Licu')
plt.errorbar(time,l1,yerr=e1,fmt='ro-',drawstyle='steps-mid')
plt.errorbar(time,l2,yerr=e2,fmt='go-',drawstyle='steps-mid')
plt.errorbar(time,l3,yerr=e3,fmt='bo-',drawstyle='steps-mid')
plt.plot(time,ltarr[:nn],'r--',linewidth=1.9)
plt.plot(time,ltarr[nn:2*nn],'g--',linewidth=1.9)
plt.plot(time,ltarr[2*nn:],'b--',linewidth=1.9)

plt.figure('Theta offsets')
plt.axes().set_aspect('equal', 'datalim')
plt.ylim(-29.6,-28.4)
plt.xlim(265.8,267.2)
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
cv1 = Ri2b.T*vec1_b
cv2 = Ri2b.T*vec2_b
cv3 = Ri2b.T*vec3_b
alph1,del1 = unit2iner(cv1[0,0],cv1[1,0],cv1[2,0])
alph2,del2 = unit2iner(cv2[0,0],cv2[1,0],cv2[2,0])
alph3,del3 = unit2iner(cv3[0,0],cv3[1,0],cv3[2,0])
plt.plot(np.rad2deg(alph1),np.rad2deg(del1),'r+')
plt.plot(np.rad2deg(alph2),np.rad2deg(del2),'g+')
plt.plot(np.rad2deg(alph3),np.rad2deg(del3),'b+')
plt.plot(1.0*hdl[1].data['Roll_RA'][idsel][idd],1.0*hdl[1].data['Roll_DEC'][idsel][idd],'k^',markersize=8)

Rc2i_pl = Ri2carr[-1].T
#RA Scan - xaxis, Dec Scan - yaxis
xax = Rc2i_pl*np.matrix([1.0,0.0,0.0]).T
yax = Rc2i_pl*np.matrix([0.0,1.0,0.0]).T
pvec_c = Ri2carr[-1]*np.matrix(pntvec[-1]).T
cf,dra1,ddec1 = collimfunc(1e-2,1e-2,1,full=True) # dummy values to ra and dec
# done mainly for getting max vals for getting dra and ddec 
cf,dra2,ddec2 = collimfunc(1e-2,1e-2,2,full=True)
cf,dra3,ddec3 = collimfunc(1e-2,1e-2,3,full=True)
thetvra1 = np.arctan2(pvec_c[2],pvec_c[1]) + dra1*np.pi/180.
thetvra2 = np.arctan2(pvec_c[2],pvec_c[1]) + dra2*np.pi/180.
thetvra3 = np.arctan2(pvec_c[2],pvec_c[1]) + dra3*np.pi/180.

thetvdec1 = np.arctan2(pvec_c[0],pvec_c[2]) + ddec1*np.pi/180.
thetvdec2 = np.arctan2(pvec_c[0],pvec_c[2]) + ddec2*np.pi/180.
thetvdec3 = np.arctan2(pvec_c[0],pvec_c[2]) + ddec3*np.pi/180.

plxpc1 = np.array([np.tan(thetvdec1),1./np.tan(thetvra1),1.0])
plxpc1 = plxpc1/np.linalg.norm(plxpc1)
lxpc1 = Ri2carr[-1].T*np.matrix(plxpc1).T
alpc1,delpc1 = unit2iner(lxpc1[0,0],lxpc1[1,0],lxpc1[2,0])
plt.plot(np.rad2deg(alpc1),np.rad2deg(delpc1),'rs',alpha=0.3)

plxpc2 = np.array([np.tan(thetvdec2),1./np.tan(thetvra2),1.0])
plxpc2 = plxpc2/np.linalg.norm(plxpc2)
lxpc2 = Ri2carr[-1].T*np.matrix(plxpc2).T
alpc2,delpc2 = unit2iner(lxpc2[0,0],lxpc2[1,0],lxpc2[2,0])
plt.plot(np.rad2deg(alpc2),np.rad2deg(delpc2),'gs',alpha=0.3)

plxpc3 = np.array([np.tan(thetvdec3),1./np.tan(thetvra3),1.0])
plxpc3 = plxpc3/np.linalg.norm(plxpc3)
lxpc3 = Ri2carr[-1].T*np.matrix(plxpc3).T
alpc3,delpc3 = unit2iner(lxpc3[0,0],lxpc3[1,0],lxpc3[2,0])
plt.plot(np.rad2deg(alpc3),np.rad2deg(delpc3),'bs',alpha=0.3)


#camera axis drawing
dtetas = np.linspace(-0.01*np.pi/2,0.01*np.pi/2,101)
tetras = np.arctan2(pvec_c[2],pvec_c[1])[0,0] + dtetas
tetdecs = np.arctan2(pvec_c[0],pvec_c[2])[0,0] + dtetas
for ik in np.arange(dtetas.size) :
    pax = np.array([np.tan(tetdecs[ik]),1.0/np.tan(tetras[int(dtetas.size/2)]),1.0])
    pax = pax/np.linalg.norm(pax)
    ipax = Ri2carr[-1].T*np.matrix(pax).T
    a,d = unit2iner(ipax[0,0],ipax[1,0],ipax[2,0])
    plt.plot(np.rad2deg(a),np.rad2deg(d),'k+')
    pay = np.array([np.tan(tetdecs[int(dtetas.size/2)]),1.0/np.tan(tetras[ik]),1.0])
    pay = pay/np.linalg.norm(pay)
    ipay = Ri2carr[-1].T*np.matrix(pay).T
    a,d = unit2iner(ipay[0,0],ipay[1,0],ipay[2,0])
    plt.plot(np.rad2deg(a),np.rad2deg(d),'m+')

plt.show()
fth.close()
sth.close()
hdllc.close()
hdl.close()
