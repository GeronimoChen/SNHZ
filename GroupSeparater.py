import os
import sys
import glob
import numpy as np
import pandas as pd
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.ndimage import gaussian_filter
from scipy.stats import wasserstein_distance
from astropy.wcs.utils import pixel_to_skycoord,skycoord_to_pixel
from astropy.visualization import wcsaxes,ZScaleInterval,ImageNormalize

i=int(sys.argv[1])
groupSize=int(sys.argv[2])

uvals,vvals=np.arange(50),np.arange(50)
FinalSelMuse=pd.read_csv('FinalSelMuseMask3.csv',index_col=0)

def emdKnn(sfhCube,massMap,stepper=100,coreNum=10):
    groupMap=np.random.randint(coreNum,size=(sfhCube.shape[1],sfhCube.shape[2]))+1
    groupMap[np.isnan(sfhCube.sum(axis=0))]=0
    groupMap[sfhCube.sum(axis=0)<0.8]=0
    groupMap[np.isnan(massMap)]=0
    groupMap=groupMap*1.0
    groupMap[groupMap==0]=np.nan
    groupMapOld=groupMap.copy()
    for indexer in range(stepper):
        meanSFH=[]
        massList=[]
        for group in range(1,coreNum+1):
            sfhCubeSel=sfhCube[:,groupMap==group]
            massSel=massMap[groupMap==group]
            if sfhCubeSel.shape[1]<=1:continue
            sfhCubeAvg=np.sum(sfhCubeSel*massSel,axis=1)/np.sum(massSel)
            meanSFH.append(sfhCubeAvg)
            massList.append(np.sum(massSel))
        coreNum=len(meanSFH)
        for x in range(groupMap.shape[0]):
            for y in range(groupMap.shape[1]):
                if np.isnan(groupMap[x,y]):continue
                emdList=[wasserstein_distance(uvals,vvals,meanSFH[j],sfhCube[:,x,y]) for j in range(0,coreNum)]
                groupMap[x,y]=np.argmin(emdList)+1
        print(indexer)
        if np.average(groupMap[np.isnan(groupMap)==False]==groupMapOld[np.isnan(groupMapOld)==False])==1:break
        else:groupMapOld=groupMap.copy()
    return groupMap,meanSFH,massList

arcName=FinalSelMuse['ArcName'][i]
crcName=arcName.replace('ADP','CDP')
fitsFile=fits.open('../MuseObject/MuseResample/'+crcName+'.fits')
ppxfOut=fits.open('../MuseSFH/dataOut/'+str(i)+'_'+FinalSelMuse['SN Name'][i]+'.fits')

wcsselect=WCS(fitsFile[1])
snCoord=SkyCoord(FinalSelMuse['SN RA'][i]*u.deg,FinalSelMuse['SN DEC'][i]*u.deg)
minpos=skycoord_to_pixel(snCoord,wcsselect)
minpos=(int(minpos[0]),int(minpos[1]))
psfSig=fitsFile[0].header['SKY_RES']/0.6/2.355#0.6 here is the pixel size in arcsec for MUSE, 2.355 is the conversion between sigma and FWHM

fitFlux=ppxfOut[0].data
massMap=np.nanmedian(fitsFile[1].data,axis=0)/np.nanmedian(fitFlux,axis=0)
massMap[massMap<0]=np.nan

sfhCube=ppxfOut[1].data
sfhCube=sfhCube.sum(axis=1)
sfhCube[sfhCube<0]=0
sfhCube[np.isnan(sfhCube)]=0
sfhCube[:,np.isnan(massMap)]=0

for indexer in range(50):sfhCube[indexer]=gaussian_filter(sfhCube[indexer],sigma=psfSig,truncate=3)

groupMap,meanSFH,massList=emdKnn(sfhCube,massMap,coreNum=groupSize)

meanSFH=np.array(meanSFH)
massList=np.array(massList)

mapHdu=fits.PrimaryHDU(data=groupMap)
mapHdu.header['Type']='Group Map'
sfhHdu=fits.ImageHDU(data=meanSFH)
sfhHdu.header['Type']='Mean SFH'
masHdu=fits.ImageHDU(data=massList)
masHdu.header['Type']='Mass List'
mapFits=fits.HDUList([mapHdu,sfhHdu,masHdu])
mapFits.writeto('GroupMap/'+str(i)+'_'+FinalSelMuse['SN Name'][i]+'_'+str(groupSize)+'.fits',overwrite=True)





