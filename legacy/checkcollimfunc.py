import  numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import InterpolatedUnivariateSpline as ip
from scipy.optimize import fsolve
import matplotlib.cm as cm

folfits="../../"
lxpc=3
ra0 = 83.63
dec0 = 22.01
full = False
sz= 100
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


rapc = (tlpcr - poltra[1])/poltra[0] + ra0
decpc = (tlpcd - poltdec[1])/poltdec[0] + dec0
agra = np.argsort(rapc)
agdec = np.argsort(decpc)
ipr2 = ip(rapc[agra],raval[agra],k=2)
ipd2 = ip(decpc[agdec],decval[agdec],k=2)
drr2 = lambda z : ipr2.derivatives(z)[1]
ddr2 = lambda z : ipd2.derivatives(z)[1]
ramax2 = fsolve(drr2,84.01)[0]
decmax2 = fsolve(ddr2,22)[0]
#ramax2 = (tlrmax - poltra[1])/poltra[0] + ra0
#decmax2 = (tldmax - poltdec[1])/poltdec[0] + dec0

rasims = np.linspace(rapc.min(),rapc.max(),sz) - ra0
timra = fra(rasims)
tra0 = fra(0)
decsims = np.linspace(decpc.min(),decpc.max(),sz) - dec0
tdec0 = fdec(0)
timdec = fdec(decsims)
#if ( np.any(timra > tra.max()) | np.any(timra < tra.min()) | \
#    np.any(timdec > tdec.max()) | np.any(timdec < tdec.min()) ) :
#  raise ValueError("Angle out of FOV")

raresp = ipr(timra)/ipr(tlrmax)#*ipd(tdec0)/ipd(tldmax)
decresp = ipd(timdec)/ipd(tldmax)
raresp2 = ipr2(rasims + ra0)/ipr2(ramax2)#*ipd2(dec0)/ipd2(decmax2)
decresp2 = ipd2(decsims + dec0)/ipd2(decmax2)

r2d = np.empty((sz,sz))
r2d2 = np.empty((sz,sz))
for ir in np.arange(sz) :
    r2d[ir,:] = ipr(fra(rasims[ir]))/ipr(tlrmax)*ipd(timdec)/ipd(tldmax)
    r2d2[ir,:] = ipr2(rasims[ir]+ra0)/ipr2(ramax2)*ipd2(decsims + dec0)/ipd2(decmax2)

plt.figure('RA at dec = 83.63')
plt.plot(ra,ipr(tra)/ipr(tlrmax),'r+')
plt.plot(rapc,raval/ipr(tlrmax),'ks')
plt.plot(rasims+ra0,raresp,'b')
plt.plot(ra0 + (tlrmax - poltra[1])/poltra[0],1.0,'bo')
plt.plot(rasims+ra0,raresp2,'r')
plt.plot(ramax2,1.0,'ro')
plt.axhline(0,color='k')

plt.figure('DEC at ra = 22.01')
plt.plot(dec,ipd(tdec)/ipd(tldmax),'r+')
plt.plot(dec0 + (tlpcd - poltdec[1])/poltdec[0],decval/ipd(tldmax),'ks')
plt.plot(decsims+dec0,decresp,'b')
plt.plot(dec0 + (tldmax - poltdec[1])/poltdec[0],1.0,'bo')
plt.plot(decsims+dec0,decresp2,'r')
plt.plot(decmax2,1.0,'ro')
plt.axhline(0,color='k')

fg = plt.figure('2D resp')
ax = fg.add_subplot(111,projection='3d')
r2,d2 = np.meshgrid(rasims+ra0,decsims+dec0)
sf = ax.plot_surface(r2,d2,r2d,rstride=1,cstride=1,cmap=cm.coolwarm)
#m = cm.ScalarMappable(cmap=sf.cmap,norm=sf.norm)
#m.set_array(r2d)
fg.colorbar(sf,shrink=0.7)


fg2 = plt.figure('2D resp2')
ax2 = fg2.add_subplot(111,projection='3d')
r2,d2 = np.meshgrid(rasims+ra0,decsims+dec0)
sf2 = ax2.plot_surface(r2,d2,r2d2,rstride=1,cstride=1,cmap=cm.coolwarm)
fg2.colorbar(sf2,shrink=0.7)

plt.show()

# Tue 17 Jul 2018 02:11:33 PM CEST 
# Based on this - conclusion is to stick with current collimfunc code with abs
# values for edge points !!
# Only diff. is using rapc and decpc for FOV edges :)
