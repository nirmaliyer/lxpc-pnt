import  numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit,fsolve 
from scipy.interpolate import InterpolatedUnivariateSpline as ip
from scipy.integrate import quad
from rotfunc import *

def effarea(lxpc) :
  """
  Gets integrated eff. area between 3 to 80 keV
  """
  fname="effarea_lxpc.txt"
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
  intar = quad(fn,3.0,80.0) # no use breaking the func - see what prblm
  return intar[0]


def collimfunc(delthetra,delthetdec,lxpc=1,ra0=83.63,dec0=22.01) :
	"""
	Read LAXPC collimator response from files and give response (max unity) for
	given offset angles theta and phi
	"""
	fnamera = "LX" + str(lxpc) + "0_RA.txt"
	fnamedec = "LX" + str(lxpc) + "0_Dec.txt"

	tlpcr,raval = np.genfromtxt(fnamera,skip_header=1,unpack=True)
	ipr = ip(tlpcr,raval,k=2)
	drr = lambda z : ipr.derivatives(z)[1]
	tlrmax = fsolve(drr,16879)[0]
	tlpcd,decval = np.genfromtxt(fnamedec,skip_header=1,unpack=True)
	ipd = ip(tlpcd,decval,k=2)
	ddr = lambda z : ipd.derivatives(z)[1]
	tldmax = fsolve(ddr,29226)[0]

	tra,ra = np.genfromtxt('rascan.txt',skip_header=1,unpack=True)
	tdec,dec = np.genfromtxt('./decscan.txt',skip_header=1,unpack=True)
	poltra = np.polyfit(ra-ra0,tra,1)
	poltdec = np.polyfit(dec-dec0,tdec,1)#Inverting the variables for linfit
	fra = np.poly1d(poltra)
	fdec = np.poly1d(poltdec)

	timra = fra(np.rad2deg(delthetra))
	timdec = fdec(np.rad2deg(delthetdec))
	if (timra > tra.max() | timra < tra.min() | timdec > tdec.max() | timdec < \
		tdec.min()) :
	  raise ValueError("Angle out of FOV")
	return ipr(timra)/ipr(tlrmax)*ipd(timdec)/ipd(tldmax)


def fitlicu(time,A,tau,t0,ra,dec,Ri2carr) :
	"""
	Function to fit the lightcurve and get ra and dec of source
	"""
	nsingle = time.size/3.0 # For each LAXPC
	orig = A*np.exp(-(time[:nsingle] - t0)/tau) #Assuming time for all LAXPCs are same
	ptvec = iner2dc(ra,dec) # undecided on whether to keep ra/dec in deg or rad.
	# does keeping in deg give more area to search for algo and make it easier to
	# converge ?
	thet1 = np.empty(nsingle)
	thet2 = np.empty(nsingle)
	for ii in np.arange(nsingle) :
	  ptvec_c = np.matrix(Ri2carr[ii])*ptvec
	  thet1[ii] = np.arctan2(ptvec_c[1],ptvec_c[2]) # y/z
	  thet2[ii] = np.arctan2(ptvec_c[0],ptvec_c[2]) # x/z
	lx1 = orig*effarea(1)/collimfunc(0,0,1)*collimfunc(thet1,thet2,1)
	lx2 = orig*effarea(2)/collimfunc(0,0,2)*collimfunc(thet1,thet2,2)
	lx3 = orig*effarea(3)/collimfunc(0,0,3)*collimfunc(thet1,thet2,3)
	retarr = np.reshape((lx1,lx2,lx3),-1) # append to 1d array
	return retarr



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

#hdlp = fits.open('./Archive/AS1T01_052T01_9000000316lxp_level1.mkf')
#getting Ri2b_cr other way of using Roll Rotation avg of 4th and 6th orbit
fth = fits.open('4thorbit-AS1T01_052T01_9000000316lxp_level1.mkf') # RA Scan
sth = fits.open('6thorbit-AS1T01_052T01_9000000316lxp_level1.mkf') # Dec Scan
fiid = np.arange(5120,10120) #indices for RA scan 4th orbit
qtyf = np.abs(1.0*fth[1].data['Roll_RA'][fiid] - 83.63)
idxf = np.where(qtyf == qtyf.min())[0][0] + fiid[0]
qf = 1.0*fth[1].data['Q_SAT'][idxf] # multiplying 1.0 to get 64 bit precision.
Rf = quat2dc(qf)
vrsi = np.array([0.0,0.0,1.0]) #x axis of collimator camera scan coordinate (ccsc)
vrs = Rf*np.matrix(vrsi).T

siid = np.arange(7555,12940) #indices for RA scan 4th orbit
qtys = np.abs(1.0*sth[1].data['Roll_DEC'][siid] - 22.01)
idxs = np.where(qtys == qtys.min())[0][0] + siid[0]
rdid = idxs + 169
qs = 1.0*fth[1].data['Q_SAT'][idxs]
Rs = quat2dc(qs)
vd1 = iner2dc(np.deg2rad(1.0*sth[1].data['Roll_RA'][idxs]),np.deg2rad(1.0*sth[1].data['Roll_DEC'][idxs]))
vd2 = iner2dc(np.deg2rad(1.0*sth[1].data['Roll_RA'][idxs]),np.deg2rad(1.0*sth[1].data['Roll_DEC'][rdid]))
#Manipulating the scan to be along orthogonal RA and DEC axes !
vdsi = np.cross(vd1.A1,vd2.A1) # y axis of ccsc
vdsi[2] = 0.0 #Manipulating the scan to be along orthogonal RA and DEC axes !
vdsi = vdsi/np.linalg.norm(vdsi)
vds = Rs*np.matrix(vdsi).T


rrotn = 0.5*fth[1].data['Roll_ROT'][idxf] + 0.5*sth[1].data['Roll_ROT'][idxs]

vzs = np.cross(vrs,vds) # z axis of ccsc
Rb2c = np.matrix((vrs,vds,vzs)) #camera to body conversion matrix - important and
#fixed matrix irrespective of pointing


qrrotn = angvec2quat(rollcr.A1,np.deg2rad(rrotn))
rrotnmat = quat2dc(qrrotn)
yawvect = np.matrix([[1.0],[-rollcr[0,0]/rollcr[1,0]],[0.0]])
yrotn = rrotnmat.T*yawvect
yawvec = yrotn/np.linalg.norm(yrotn)
pitvec = np.cross(yawvec.A1,rollcr.A1)
Ri2b_cr = np.matrix((yawvec.A1,rollcr.A1,pitvec))
#Ri2b_cr = quat2dc(quartcr)

#Rb2c = Ri2b_cr.T*Ri2c_cr 

vec1_b = Ri2b_cr*vec1
vec2_b = Ri2b_cr*vec2
vec3_b = Ri2b_cr*vec3


hdllc = fits.open('./Burst1-lxp2.lc')
tstart1 = hdllc[1].header['TSTART']
tend1 = hdllc[1].header['TSTOP']
hdllc.close()

hdl = fits.open('AS1A03_116T01_9000001460lxp_level1-aux1.att')
idsel = np.where((1.0*hdl[1].data['TIME'] >= tstart1) & \
        (1.0*hdl[1].data['TIME'] <= tend1))[0]
plt.plot(1.0*hdl[1].data['Roll_RA'][idsel][0],1.0*hdl[1].data['Roll_DEC'][idsel][0],'k^')
quart = 1.0*hdl[1].data['Q_SAT'][idsel]
nsm = quart.shape[0]
#nsm = 3000 # trying now for 1st 3000 rows only
alph1 = np.empty(nsm)
alph2 = np.empty(nsm)
alph3 = np.empty(nsm)
del1 = np.empty(nsm)
del2 = np.empty(nsm)
del3 = np.empty(nsm)

plt.axes().set_aspect('equal', 'datalim')
plt.ylim(-30,-27)
plt.xlim(265.8,267.2)
plt.xlabel(r" $\alpha$ (RA)")
plt.ylabel(r" $\delta$ (dec)")
plt.grid()
plt.plot(e1alph,e1del,'bo')
plt.plot([saxalph,grsalph,ksalph,e2alph,igralph],\
        [saxdel,grsdel,ksdel,e2del,igrdel],'kx')
ylo,yhi = plt.ylim()
offs = (yhi - ylo)*0.05 # 10% of the range
plt.text(saxalph + offs,saxdel,'SAX J1747.0-2853')
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

hdllc = fits.open("test1.lc")
tdel = hdllc[1].header['TIMEDEL']*86400.0 # in seconds
time = 1.0*hdllc[1].data['TIME'] + (1.0*hdllc[1].header['TIMEZERI'] + \
	hdllc[1].header['TIMEZERF'] - hdl[0].header['MJDREFI'] + 40000)*86400. # in sec since launch
l1 = hdllc[1].data['RATE1']*tdel
e1 = hdllc[1].data['ERROR1']*tdel
l2 = hdllc[1].data['RATE2']*tdel
e2 = hdllc[1].data['ERROR2']*tdel
l3 = hdllc[1].data['RATE3']*tdel
e3 = hdllc[1].data['ERROR3']*tdel
tmatt = hdl[1].data['TIME'][idsel]

A = 1000
t0 = time[0]
tau = 13.0 # from fplot fit

tmarr = np.reshape((time,time,time),-1)
nn = time.size
Ri2carr = np.empty((time.size,3,3))
for ii in np.arange(time.size) :
    idd = np.where(tmatt <= (time[ii] + tdel))[0][-1] # taking last att value in that time window
    Rb2i = quat2dc(quart[idd]).T
    Ri2carr[ii] = Rb2i.T*Rb2c

ltarr = fitlicu(tmarr,A,tau,t0,e1alph,e1del,Ri2carr)
plt.figure('1E Licu')
plt.errorbar(time,l1,yerr=e1,fmt='bo-',drawstyle='steps')
plt.errorbar(time,l2,yerr=e2,fmt='ro-',drawstyle='steps')
plt.errorbar(time,l3,yerr=e3,fmt='go-',drawstyle='steps')
plt.plot(time,ltarr[:nn],'b--')
plt.plot(time,ltarr[nn:2*nn],'r--')
plt.plot(time,ltarr[2*nn:],'g--')

fth.close()
sth.close()
hdllc.close()
hdl.close()
