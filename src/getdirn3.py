import  numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit,fsolve 
from scipy.interpolate import InterpolatedUnivariateSpline as ip
from scipy.integrate import quad
from rotfunc import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from os.path import isfile

folfits="../../"
threedplot = False
#Nifty trick picked from https://stackoverflow.com/questions/13531247/python-catching-specific-exception
class MyValueError(ValueError) : pass


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
  #intar = romb(area[idpc],ener[idpc],elo,ehi) # why not use discrete integration ?
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
    rapc = (tlpcr - poltra[1])/poltra[0]
    decpc = (tlpcd - poltdec[1])/poltdec[0] 
    if ( np.any(np.rad2deg(delthetra) > rapc.max()) | \
         np.any(np.rad2deg(delthetra) < rapc.min()) | \
         np.any(np.rad2deg(delthetdec) > decpc.max()) | \
         np.any(np.rad2deg(delthetdec) < decpc.min()) ) :
      raise MyValueError("Angle out of FOV")

    timra = fra(np.rad2deg(delthetra))
    timdec = fdec(np.rad2deg(delthetdec))
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


def getvect1t2(thetx,thety,primary='x') :
    """
    Function to generate unit vector given angles t1 and t2
    Note in this function thetx is vector in xz plane = t2 which will give x
    axis of vector and thety is vector in yz plane = t1 which will give y
    Generally, primary should not matter as both thetx and thety should have
    same sign for z. But for some applications (like draw camera axis below)
    this may not be true. Sign for z is then taken from the primary
    """
    thetx = np.mod(thetx,2*np.pi)
    thety = np.mod(thety,2*np.pi)
    if ((thetx >= 0) & (thetx <= np.pi)) : 
        sgnx = 1
    else :
        sgnx = -1
    if ((thety >= 0) & (thety <= np.pi)) : 
        sgny = 1
    else :
        sgny = -1
    tchk = thetx if (primary == 'x') else thety
    if ((tchk >= 0) & (tchk <= np.pi/2) | (tchk >= 3*np.pi/2) & (tchk <= 2*np.pi)) : 
        sgnz = 1
    else :
        sgnz = -1
    x = sgnx*np.abs(np.tan(thetx))
    y = sgny*np.abs(np.tan(thety))
    z = sgnz
    vec = np.array([x,y,z])
    vec = vec/np.linalg.norm(vec)
    return vec

def getratio(srcra,srcdec,pntvec,Ri2c) :
    offvec = iner2dc(srcra,srcdec)
    ofc = Ri2c*offvec
    pnc = Ri2c*np.matrix(pntvec).T
    t1,t2 = getangle(pnc.A1,ofc.A1) 
    lx1 = effarea(1)*np.abs(collimfunc(t1,t2,1))
    lx2 = effarea(2)*np.abs(collimfunc(t1,t2,2))
    return lx2/lx1


listra,listdec = np.genfromtxt('srclist',usecols=(2,3),delimiter=',',unpack=True)
srcnam = np.genfromtxt('srclist',usecols=(1),delimiter=',',dtype=str)

offs = 0.017
e1alph = 266.58787
e1del = -28.72842

alfcr = np.deg2rad(83.63)
deltcr = np.deg2rad(22.01)

rollcr = iner2dc(alfcr,deltcr)

fth = fits.open(folfits + '4thorbit-AS1T01_052T01_9000000316lxp_level1.mkf') # RA Scan
sth = fits.open(folfits + '6thorbit-AS1T01_052T01_9000000316lxp_level1.mkf') # Dec Scan
fiid = np.arange(5120,10120) #indices for RA scan 4th orbit
qtyf = np.abs(1.0*fth[1].data['Roll_RA'][fiid] - 83.63)
idxf = np.where(qtyf == qtyf.min())[0][0] + fiid[0]
qf = 1.0*fth[1].data['Q_SAT'][idxf] # multiplying 1.0 to get 64 bit precision.
vrs = np.array([0.0,0.0,1.0]) #x axis of collimator camera scan coordinate

siid = np.arange(7555,12940) #indices for DEC scan 6th orbit
qtys = np.abs(1.0*sth[1].data['Roll_DEC'][siid] - 22.01)
idxs = np.where(qtys == qtys.min())[0][0] + siid[0]
rdid = idxs + 1069 # some random id in DEC scan
qs = 1.0*fth[1].data['Q_SAT'][idxs]
vd1 = iner2dc(np.deg2rad(1.0*sth[1].data['Roll_RA'][idxs]),np.deg2rad(1.0*sth[1].data['Roll_DEC'][idxs]))
vd2 = iner2dc(np.deg2rad(1.0*sth[1].data['Roll_RA'][idxs]),np.deg2rad(1.0*sth[1].data['Roll_DEC'][rdid]))
#Manipulating the scan to be along orthogonal RA and DEC axes !
vdsi = np.cross(vd1.A1,vd2.A1) # y axis of ccsc
vdsi[2] = 0.0 #Manipulating the scan to be along orthogonal RA and DEC axes !
vds = vdsi/np.linalg.norm(vdsi)


rrotn = 0.5*fth[1].data['Roll_ROT'][idxf] + 0.5*sth[1].data['Roll_ROT'][idxs]
# Roll rotation is angle of rotation about the roll axis between xaxis (yaw)
# of satellite/body coordinates and the vector which lies both in xy plane of inertial
# and yaw-pitch plane of body coordinates. Thus it is perpendicular to roll
# vector (of body) and z axis (of inertial system)

vzs = np.cross(vrs,vds) # z axis of ccsc
Ri2c_cr = np.matrix((vrs,vds,vzs)) 
pvec_cr_c = Ri2c_cr*rollcr 

qrrotn = angvec2quat(rollcr.A1,np.deg2rad(rrotn))
rrotnmat = quat2dc(qrrotn)
yawvect = np.matrix([[1.0],[-rollcr[0,0]/rollcr[1,0]],[0.0]])
yrotn = rrotnmat.T*yawvect
yawvec = yrotn/np.linalg.norm(yrotn)
pitvec = np.cross(yawvec.A1,rollcr.A1)
Ri2b_cr = np.matrix((yawvec.A1,rollcr.A1,pitvec))
#Ri2b_cr = quat2dc(fth[1].data['Q_SAT'][idxf])
Rb2c = Ri2c_cr*Ri2b_cr.T #camera to body conversion matrix - important and
#fixed matrix irrespective of pointing


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
quart = 1.0*hdl[1].data['Q_SAT'][idsel]
quart2 = 1.0*hdl[1].data['Q_SAT'][idsel2]
tmatt = hdl[1].data['TIME'][idsel]
tmatt2 = hdl[1].data['TIME'][idsel]

Ri2b_b1 = quat2dc(quart[-1])
Ri2b_b2 = quat2dc(quart2[-1])
pntvec_b1 = Ri2b_b1[1] # roll vector
pntvec_b2 = Ri2b_b2[1] # roll vector
Ri2c_b1 = Rb2c*Ri2b_b1
Ri2c_b2 = Rb2c*Ri2b_b2


#Brute force scan of field of view

#Scan Params
nptg = 70
gridlim = 0.95
centra = 1.0*hdl[1].data['Roll_RA'][idsel][-1]
centdec = 1.0*hdl[1].data['Roll_DEC'][idsel][-1]
#Digitzed rectangular grid of nptg*nptg of the FOV

#ratiogrid = np.full(nptg,nptg),np.nan)
doloop=True
fldat = folfits + "ratios.npz"
if (isfile(fldat)) :
    rtd = np.load(fldat)
    nptgfl = rtd["nptg"]
    limfl = rtd["grid"]
    rafl = rtd["centra"]
    decfl = rtd["centdec"]
    if ( (nptg == nptgfl) & (limfl == gridlim) & \
            (rafl == 1.0*hdl[1].data['Roll_RA'][idsel][-1]) & \
            (decfl == 1.0*hdl[1].data['Roll_DEC'][idsel][-1]) ) :
        doloop = False
        ratiogrid = rtd["ratiogrid"]
        ratiogrid2 = rtd["ratiogrid2"]
        ragrid = rtd["ragrid"]
        decgrid = rtd["decgrid"]

if (doloop) :
    ragrid = np.linspace(-gridlim,gridlim,nptg) + centra
    decgrid = np.linspace(-gridlim,gridlim,nptg) + centdec
    ratiogrid = np.zeros((nptg,nptg))
    ratiogrid2 = np.zeros((nptg,nptg))
    for ira in np.arange(nptg) :
        for idc in np.arange(nptg) :
            try :
                ratiogrid[idc,ira] = getratio(ragrid[ira]*np.pi/180, \
                        decgrid[idc]*np.pi/180,pntvec_b1,Ri2c_b1)
                ratiogrid2[idc,ira] = getratio(ragrid[ira]*np.pi/180, \
                        decgrid[idc]*np.pi/180,pntvec_b2,Ri2c_b2)
            except MyValueError :
                continue
    np.savez(folfits+"ratios.npz",nptg=nptg,grid=gridlim,centra=centra,centdec=centdec,\
            ratiogrid=ratiogrid,ratiogrid2=ratiogrid2,ragrid=ragrid,decgrid=decgrid)

ratb1 = 1.66
ratb2 = 3.06

rg,dg = np.meshgrid(ragrid,decgrid)

if threedplot :
    fg = plt.figure('3d plot')
    ax = fg.add_subplot(111,projection='3d')
    ax.scatter(1.0*hdl[1].data['Roll_RA'][idsel][-1],1.0*hdl[1].data['Roll_DEC'][idsel][-1],marker='^',color='k')

    b1pvec_c = Ri2c_b1*np.matrix(pntvec_b1).T
    cf,dra1,ddec1 = collimfunc(1e-2,1e-2,1,full=True) # dummy values to ra and dec
    # done mainly for getting max vals for getting dra and ddec 
    cf,dra2,ddec2 = collimfunc(1e-2,1e-2,2,full=True)
    cf,dra3,ddec3 = collimfunc(1e-2,1e-2,3,full=True)
    b1thetvra1 = np.arctan2(b1pvec_c[1],b1pvec_c[2])[0,0] - dra1*np.pi/180. 
    b1thetvra2 = np.arctan2(b1pvec_c[1],b1pvec_c[2])[0,0] - dra2*np.pi/180.
    b1thetvra3 = np.arctan2(b1pvec_c[1],b1pvec_c[2])[0,0] - dra3*np.pi/180.

    b1thetvdec1 = np.arctan2(b1pvec_c[0],b1pvec_c[2])[0,0] + ddec1*np.pi/180.
    b1thetvdec2 = np.arctan2(b1pvec_c[0],b1pvec_c[2])[0,0] + ddec2*np.pi/180.
    b1thetvdec3 = np.arctan2(b1pvec_c[0],b1pvec_c[2])[0,0] + ddec3*np.pi/180.

    b1plxpc1 = getvect1t2(b1thetvdec1,b1thetvra1)
    b1lxpc1 = Ri2c_b1.T*np.matrix(b1plxpc1).T
    b1alpc1,b1delpc1 = unit2iner(b1lxpc1[0,0],b1lxpc1[1,0],b1lxpc1[2,0])
    ax.text(np.rad2deg(b1alpc1),np.rad2deg(b1delpc1),0,'L1',fontdict={'color':'r'})

    b1plxpc2 = getvect1t2(b1thetvdec2,b1thetvra2)
    b1lxpc2 = Ri2c_b1.T*np.matrix(b1plxpc2).T
    b1alpc2,b1delpc2 = unit2iner(b1lxpc2[0,0],b1lxpc2[1,0],b1lxpc2[2,0])
    ax.text(np.rad2deg(b1alpc2),np.rad2deg(b1delpc2),0,'L2',fontdict={'color':'g'})

    b1plxpc3 = getvect1t2(b1thetvdec3,b1thetvra3)
    b1lxpc3 = Ri2c_b1.T*np.matrix(b1plxpc3).T
    b1alpc3,b1delpc3 = unit2iner(b1lxpc3[0,0],b1lxpc3[1,0],b1lxpc3[2,0])
    ax.text(np.rad2deg(b1alpc3),np.rad2deg(b1delpc3),0,'L3',fontdict={'color':'m'})

    ax.scatter(e1alph,e1del,0,color='b',marker='o')

    sf = ax.plot_surface(rg,dg,ratiogrid,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0)
    fg.colorbar(sf,shrink=0.7)

    sfr11 = ax.plot_surface(rg,dg,ratb1*1.1,rstride=1,cstride=1,alpha=0.2,color='k')
    sfr11 = ax.plot_surface(rg,dg,ratb1*0.9,alpha=0.2,color='k')


hdliu = fits.open(folfits + "liu_cat.fits")
hdrit = fits.open(folfits + "ritter_cat.fits")

width = 6
ht = width
plt.rc('font', family='serif', serif='Times',size=10)
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('axes', labelsize=12)
plt.rc('legend',fontsize=10)
fig = plt.figure('Ratios')
fig.set_size_inches(width,ht)
plt.axes().set_aspect('equal', 'datalim')
plt.ylim(-30.05,-27.65)
plt.xlim(265.4,267.9)
plt.gca().invert_xaxis();
plt.xlabel(r" $\alpha$ (RA)")
plt.ylabel(r" $\delta$ (dec)")
plt.grid()

plt.plot(listra,listdec,'ko',markersize=3.7)
plt.plot(listra[-1],listdec[-1],'r*',markersize=4.7)
plt.text(266.42,-29.07,"Gal. Cent.")
plt.text(266.08,-29.42,"GC X-4")
plt.plot(hdliu[1].data['RA'],hdliu[1].data['DEC'],'ro',markersize=3.6)
plt.plot(hdrit[1].data['RA'],hdrit[1].data['DEC'],'go',markersize=3.5)
plt.plot(1.0*hdl[1].data['Roll_RA'][idsel][-1],1.0*hdl[1].data['Roll_DEC'][idsel][-1],'k^',markersize=5)
plt.plot(1.0*hdl[1].data['Roll_RA'][idsel2][-1],1.0*hdl[1].data['Roll_DEC'][idsel2][-1],color='gray',marker='^',markersize=5)
selpts = np.where((ratiogrid > 0.9*ratb1) & (ratiogrid < 1.1*ratb1))
selpts2 = np.where((ratiogrid2 > 0.9*ratb2) & (ratiogrid2 < 1.1*ratb2))
#plt.fill(rg[selpts],dg[selpts],'k',alpha=0.6) - Not useful - fills out a larger polygon !!
#h,xe,ye = np.histogram2d(rg[selpts],dg[selpts],bins=50)
#h2,xe2,ye2 = np.histogram2d(rg[selpts2],dg[selpts2],bins=50)
#plt.imshow(h==0,origin='lower',cmap=plt.gray(),extent=[xe[-1],xe[0],ye[-1],ye[0]])
plt.plot(rg[selpts],dg[selpts],'k.',markersize=1.2,alpha=0.6)
plt.plot(rg[selpts2],dg[selpts2],'.',color='gray',markersize=1.2,alpha=0.8)
plt.hlines([centdec-gridlim,centdec+gridlim],centra-gridlim,centra+gridlim,color='k')
plt.vlines([centra-gridlim,centra+gridlim],centdec-gridlim,centdec+gridlim,color='k')
plt.text(266.9,-28.08,"Ratio=1.66")
plt.text(266.1,-27.9,"Ratio=3.06")
plt.show()
fth.close()
sth.close()
fig.savefig('../report/Area.pdf',format='pdf',dpi=500)
#hdllc.close()
hdl.close()
