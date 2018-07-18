"""
Code just to get ratios given source location.
Copied relevant bits from getdirn2.py on Wed 18 Jul 2018 01:57:29 PM CEST 
"""

import  numpy as np
from astropy.io import fits
from rotfunc import *
from scipy.optimize import fsolve 
from scipy.interpolate import InterpolatedUnivariateSpline as ip
from scipy.integrate import quad

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

def getsrcratio(srcra,srcdec,pntvec,Ri2c) :
    offvec = iner2dc(srcra,srcdec)
    ofc = Ri2c*offvec
    pnc = Ri2c*np.matrix(pntvec).T
    t1,t2 = getangle(pnc.A1,ofc.A1) 
    lx1 = effarea(1)*np.abs(collimfunc(t1,t2,1))
    lx2 = effarea(2)*np.abs(collimfunc(t1,t2,2))
    return lx2/lx1

def getratio(lra,ldec,sburst='B1') :
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
    vdsi = np.cross(vd1.A1,vd2.A1) # y axis of ccsc
    vdsi[2] = 0.0 #Manipulating the scan to be along orthogonal RA and DEC axes !
    vds = vdsi/np.linalg.norm(vdsi)
    rrotn = 0.5*fth[1].data['Roll_ROT'][idxf] + 0.5*sth[1].data['Roll_ROT'][idxs]
    
    vzs = np.cross(vrs,vds) # z axis of ccsc
    Ri2c_cr = np.matrix((vrs,vds,vzs)) 
    qrrotn = angvec2quat(rollcr.A1,np.deg2rad(rrotn))
    rrotnmat = quat2dc(qrrotn)
    yawvect = np.matrix([[1.0],[-rollcr[0,0]/rollcr[1,0]],[0.0]])
    yrotn = rrotnmat.T*yawvect
    yawvec = yrotn/np.linalg.norm(yrotn)
    pitvec = np.cross(yawvec.A1,rollcr.A1)
    Ri2b_cr = np.matrix((yawvec.A1,rollcr.A1,pitvec))
    Rb2c = Ri2c_cr*Ri2b_cr.T

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

    if (sburst == 'B1') :
        return getsrcratio(np.deg2rad(lra),np.deg2rad(ldec),pntvec_b1,Ri2c_b1)
    else :
        return getsrcratio(np.deg2rad(lra),np.deg2rad(ldec),pntvec_b2,Ri2c_b2)

