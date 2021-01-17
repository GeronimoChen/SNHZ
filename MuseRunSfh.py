import os
import sys
import math
import glob
import numpy as np
from os import path
import pandas as pd
import multiprocessing
from ppxf.ppxf import ppxf
from astropy.wcs import WCS
from astropy.io import fits
import ppxf as ppxf_package
import ppxf.ppxf_util as util
import ppxf.miles_util as lib
from astropy import units as u
import matplotlib.pyplot as plt
from multiprocessing import Pool
from specutils import Spectrum1D
from astropy import constants as con
from time import perf_counter as clock
from specutils.manipulation import FluxConservingResampler

i=sys.argv[1]
i=int(i)

#FinalSelMuse=pd.read_csv('../MuseObject/FinalSelMuse.csv',index_col=0)
FinalSelMuse=pd.read_csv('../MuseRegulFind/FinalSelMuseRegul10Check.csv',index_col=0)
SPmask=np.load('../MuseMasker/FinalMask2/'+str(i)+'_'+FinalSelMuse['SN Name'][i]+'.npy')
regulSelHere=FinalSelMuse['RegulSelect'][i]
if np.isnan(regulSelHere):
    regulSelHere=FinalSelMuse['Regul2Select'][i]
    if np.isnan(regulSelHere):regulSelHere=0.03125
print(regulSelHere)

def DER_SNR(flux):
    from numpy import array,where,median,abs 
    flux=array(flux)
    flux=array(flux[where(flux!=0.0)])
    n=len(flux)
    if (n>4):
        signal=median(flux)
        noise=0.6052697*median(abs(2.0*flux[2:n-2]-flux[0:n-4]-flux[4:n]))
        return signal,noise
    else:return 0,100

def logSpecConverter(spec):
    waveOld=spec[:,0]*u.AA
    flux=spec[:,1]*u.erg/u.s/u.cm**2/u.AA
    noise=spec[:,2]*u.erg/u.s/u.cm**2/u.AA
    befSpec=Spectrum1D(spectral_axis=waveOld,flux=flux)
    fluxCon=FluxConservingResampler()
    waveNew=10**np.linspace(np.log10(waveOld.min().value),np.log10(waveOld.max().value),num=len(waveOld))*u.AA
    aftSpec=fluxCon(befSpec,waveNew)
    wave=aftSpec.wavelength.value
    flux=aftSpec.flux.value
    flux[np.isnan(flux)]=0
    
    befSpecN=Spectrum1D(spectral_axis=waveOld,flux=noise)
    fluxCon=FluxConservingResampler()
    aftSpecN=fluxCon(befSpecN,waveNew)
    noise=aftSpecN.flux.value
    noise[np.isnan(noise)]=9*10**9
    
    return np.array([wave,flux,noise]).T
def logSpecConvBack(wave,flux,waveOld):
    wave=wave*u.AA
    befSpec=Spectrum1D(spectral_axis=wave,flux=flux*u.erg/u.s/u.cm**2/u.AA)
    fluxCon=FluxConservingResampler()
    aftSpec=fluxCon(befSpec,waveOld*u.AA)
    wave=aftSpec.wavelength.value
    flux=aftSpec.flux.value
    flux[np.isnan(flux)]=0
    return flux

def MP(taskid_lst=None,func=None,Nprocs=20):
    def worker(taskid_lst,out_q):
        outdict={}
        for tid in taskid_lst:
            outdict[tid]=func(tid)
        out_q.put(outdict)
    out_q=multiprocessing.Queue()
    chunksize=int(math.ceil(len(taskid_lst)/float(Nprocs)))
    procs=[]
    for i in range(Nprocs):
        p=multiprocessing.Process(target=worker,\
        args=(taskid_lst[chunksize*i:chunksize*(i+1)],out_q)) 
        procs.append(p)
        p.start()
    resultdict={}
    for i in range(Nprocs):
        resultdict.update(out_q.get())
    for p in procs:
        p.join()
    return resultdict

def ppxfRunner(spec):
    t=spec
    tie_balmer=False
    limit_doublets=False
    ppxf_dir=path.dirname(path.realpath(ppxf_package.__file__))
    z=0.0
    flux=t[:,1]
    galaxy=flux/np.median(flux)
    wave=t[:,0]
    wave*=np.median(util.vac_to_air(wave)/wave)
    noise=t[:,2]
    c=299792.458  # speed of light in km/s
    velscale=c*np.log(wave[1]/wave[0])  # eq.(8) of Cappellari (2017)
    FWHM_gal=2.26  # SDSS has an approximate instrumental resolution FWHM of 2.76A.#Califa is 6. #Muse is 2.26. 

    pathname='/scratch/user/chenxingzhuo/SNHZ/ppxfSFH/ppxfGrid/EMILES_PADOVA00_BASE_KB_FITS_Sigma228/*'
    miles=lib.miles(pathname,velscale,FWHM_gal)
    reg_dim = miles.templates.shape[1:]
    stars_templates = miles.templates.reshape(miles.templates.shape[0], -1)
    lam_range_gal = np.array([np.min(wave),np.max(wave)])/(1 + z)
    gas_templates,gas_names,line_wave=util.emission_lines(
        miles.log_lam_temp,lam_range_gal,FWHM_gal,
        tie_balmer=tie_balmer,limit_doublets=limit_doublets)
    templates=np.column_stack([stars_templates,gas_templates])

    dv=c*(miles.log_lam_temp[0]-np.log(wave[0]))  # eq.(8) of Cappellari (2017)
    vel=c*np.log(1+z)   # eq.(8) of Cappellari (2017)
    start=[vel,280.]  # (km/s), starting guess for [V, sigma]

    n_temps = stars_templates.shape[1]
    n_forbidden = np.sum(["[" in a for a in gas_names])  # forbidden lines contain "[*]"
    n_balmer = len(gas_names) - n_forbidden

    component = [0]*n_temps + [1]*n_balmer + [2]*n_forbidden
    gas_component = np.array(component) > 0  # gas_component=True for gas templates
    
    bounds_star=[[-500,500],[-300,300],[-0.1,0.1],[-0.1,0.1]]
    bounds_fbd=[[-500,500],[-300,300]]
    bounds_gas=[[-500,500],[-300,300]]
    bounds=[bounds_star,bounds_fbd,bounds_gas]
    
    moments = [4,2,2]
    start = [start, start, start]
    gas_reddening = 0 if tie_balmer else None

    pp = ppxf(templates, galaxy, noise, velscale, start,
              plot=True, moments=moments, degree=-1, mdegree=10, vsyst=dv,
              lam=wave, clean=False, regul=regulSelHere, reg_dim=reg_dim,
              component=component, gas_component=gas_component,
              gas_names=gas_names, gas_reddening=gas_reddening,bounds=bounds)
    weights=pp.weights[~gas_component]  # Exclude weights of the gas templates
    weights=weights.reshape(reg_dim)/weights.sum()  # Normalized
    return pp,weights

def MPcover(coordName):
    try:
        xpix=coordName.split('_')[0]
        ypix=coordName.split('_')[1]
        xpix=int(xpix)
        ypix=int(ypix)
        mask=np.isnan(mucu[1].data[:,xpix,ypix])
        assert np.sum(mask==0)>len(mask)/2
        flux=mucu[1].data[:,xpix,ypix]
        ferr=mucu[2].data[:,xpix,ypix]
        flux=flux[mask==0]
        wave=waveBase[mask==0]
        ferr=ferr[mask==0]**0.5
        spec=np.array([wave,flux,ferr]).T
        waveOld=spec[:,0]
        spec=logSpecConverter(spec)
        waveNew=spec[:,0]
        pp,weights=ppxfRunner(spec)
        chi2=pp.chi2
        fitFluxOnePix=pp.bestfit
        gasFluxOnePix=pp.gas_bestfit
        fitFluxOnePix=logSpecConvBack(waveNew,fitFluxOnePix,waveOld)
        gasFluxOnePix=logSpecConvBack(waveNew,gasFluxOnePix,waveOld)
    except:fitFluxOnePix,gasFluxOnePix,chi2,weights,mask=np.nan,np.nan,np.nan,np.nan,np.nan
    return fitFluxOnePix,gasFluxOnePix,chi2,weights,mask


oldName=FinalSelMuse['ArcName'][i]
newName=oldName.replace('ADP','CDP')
mucu=fits.open('../MuseObject/MuseResample/'+newName+'.fits')
wavestep=float(mucu[1].header['CD3_3'])
wavestart=float(mucu[1].header['CRVAL3'])
waveBase=np.arange(wavestart,mucu[1].shape[0]*wavestep+wavestart,wavestep)
waveBase=waveBase/(1+FinalSelMuse['Redshift'][i])
fitFlux=mucu[1].data*np.nan
gasFlux=mucu[1].data*np.nan
agePopu=np.zeros([50,7,mucu[1].data.shape[1],mucu[1].data.shape[2]])*np.nan
chi2Map=np.zeros([mucu[1].data.shape[1],mucu[1].data.shape[2]])

coordList=[]
for x in range(mucu[1].data.shape[1]):
    for y in range(mucu[1].data.shape[2]):
        if SPmask[x,y]==False:continue
        coordList.append(str(x)+'_'+str(y))
print(coordList)
mpOut=MP(coordList,MPcover)

for coordName in coordList:
    fitFluxRaw,gasFluxRaw,chi2,weights,mask=mpOut[coordName]
    if np.isnan(mask).sum()==1:continue
    xpix=coordName.split('_')[0]
    ypix=coordName.split('_')[1]
    xpix=int(xpix)
    ypix=int(ypix)
    fitFluxOnePix=np.zeros(len(waveBase))*np.nan
    fitFluxOnePix[mask==0]=fitFluxRaw
    fitFlux[:,xpix,ypix]=fitFluxOnePix
    gasFluxOnePix=np.zeros(len(waveBase))*np.nan
    gasFluxOnePix[mask==0]=gasFluxRaw
    gasFlux[:,xpix,ypix]=gasFluxOnePix
    agePopu[:,:,xpix,ypix]=weights
    chi2Map[xpix,ypix]=chi2

fluxHdu=fits.PrimaryHDU(data=fitFlux)
fluxHdu.header['FIT_TYPE']='Fitting Spectra'
agesHdu=fits.ImageHDU(data=agePopu)
agesHdu.header['FIT_TYPE']='Age Metal Datacube'
hduList=fits.HDUList([fluxHdu,agesHdu])

hduList.writeto('dataOut/'+str(i)+'_'+FinalSelMuse['SN Name'][i]+'.fits',overwrite=True)

















