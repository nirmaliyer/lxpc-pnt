import numpy as np
from rotfunc import *
from astropy.io import fits

idx = 1942
folfits = "../../"
#hdl = fits.open(folfits + "4thorbit-AS1T01_052T01_9000000316lxp_level1.mkf")
#hdl = fits.open(folfits + "AS1A03_116T01_9000001460lxp_level1-aux1.att")
hdl = fits.open(folfits + "6thorbit-AS1T01_052T01_9000000316lxp_level1.mkf")
hdl = fits.open(folfits + "AS1A03_116T01_9000001460lxp_level1.mkf")
rollra = np.deg2rad(hdl[1].data['Roll_RA'][idx])
rolldec = np.deg2rad(hdl[1].data['Roll_DEC'][idx])
rrotn = hdl[1].data['Roll_ROT'][idx]
quat = hdl[1].data['Q_SAT'][idx]

rvec = iner2dc(rollra,rolldec)
rmat = quat2dc(quat)
qrrotn = angvec2quat(rvec.A1,np.deg2rad(rrotn))
rrotnmat = quat2dc(qrrotn)
pvec = rmat[2,:]
yvec = rmat[0,:]

avc11 = rrotnmat.T*np.matrix(pvec).T
avc1 = rrotnmat*np.matrix(pvec).T
avc2 = rrotnmat*np.matrix(yvec).T
avc22 = rrotnmat.T*np.matrix(yvec).T

a,b = unit2iner(avc2[0],avc2[1],avc2[2]) #gives 0 dec for different idx
# Thus Roll_ROT is angle of yaw wrt 0 dec vector in plane perpendicular to roll vector
# i.e its z = 0 in iner


#Checking inverse 
idx2 = 8824
rollra = np.deg2rad(1.0*hdl[1].data['Roll_RA'][idx2]) # 1.0 multiply as this
# makes dtype float64 instead of float32
rolldec = np.deg2rad(1.0*hdl[1].data['Roll_DEC'][idx2])
rrotn = 1.0*hdl[1].data['Roll_ROT'][idx2]

rvec = iner2dc(rollra,rolldec)
qrrotn = angvec2quat(rvec.A1,np.deg2rad(rrotn))
rrotnmat = quat2dc(qrrotn)

yawvect = np.matrix([[1.0],[-rvec[0,0]/rvec[1,0]],[0.0]])
yrotn = rrotnmat.T*yawvect
yawvec = yrotn/np.linalg.norm(yrotn)

pitvec = np.cross(yawvec.A1,rvec.A1)
rmat2 = np.matrix((yawvec.A1,rvec.A1,pitvec))
qtr = 1.0*hdl[1].data['Q_SAT'][idx2]
rmattest = quat2dc(qtr)

print (rmattest - rmat2)
#rvec err of 1e9 but pvec and yvec err of 1e-4 - why ??


