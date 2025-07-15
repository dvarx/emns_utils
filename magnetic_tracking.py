"""
this module contains common functios and classes used for electromagnetic tracking
"""

import numpy as np
from scipy.fftpack import fft
import pickle as pkl
from numpy.linalg import pinv,norm,svd,det
import matplotlib.pyplot as plt

#apply a window function to the sampled data and compute its power DFT samples[frameno,sampleno,channelno] or samples[sampleno,channelno]
def extract_windowed_pdft(samples,window):
    if(len(samples.shape)==2 or len(samples.shape)==1):
        no_samples=samples.shape[0]
        dftax=0
    elif(len(samples.shape)==3):
        no_samples=samples.shape[1]
        dftax=1
    ns=np.arange(0,no_samples)
    if window=="blackman":
        alpha=0.16
        a0=(1-alpha)/2
        a1=1/2
        a2=alpha/2
        blackman_window=a0*np.ones(len(ns))-a1*np.cos(2*np.pi*ns/no_samples)+a2*np.cos(4*np.pi*ns/no_samples)
        if len(samples.shape)==1:
            samples=blackman_window*samples
        elif len(samples.shape)>1:
            samples=blackman_window[:,None]*samples
        else:
            samples=blackman_window[None,:,None]*samples
    return fft(samples,axis=dftax)/no_samples

def getvoltphasormat(data,ks,compute_ref_sensor=False):
    #data[measno,sampleno,channelno] or data[sampleno,channelno]
    pdft=extract_windowed_pdft(data,"blackman")
    voltmat=np.zeros((3,3),dtype="complex64")
    if len(pdft.shape)>2:
        voltmat[:,0]=pdft[:,ks[0],0:3].mean(axis=0)
        voltmat[:,1]=pdft[:,ks[1],0:3].mean(axis=0)
        voltmat[:,2]=pdft[:,ks[2],0:3].mean(axis=0)
    else:
        voltmat[:,0]=pdft[ks[0],0:3]
        voltmat[:,1]=pdft[ks[1],0:3]
        voltmat[:,2]=pdft[ks[2],0:3]
    refsensor=np.zeros(3,dtype="complex64")
    if compute_ref_sensor:
        if len(pdft.shape)>2:
            refsensor[0]=pdft[:,ks[0],3].mean(axis=0)
            refsensor[1]=pdft[:,ks[1],3].mean(axis=0)
            refsensor[2]=pdft[:,ks[2],3].mean(axis=0)
        else:
            refsensor[0]=pdft[ks[0],3]
            refsensor[1]=pdft[ks[1],3]
            refsensor[2]=pdft[ks[2],3]
        return (voltmat,refsensor)
    else:
        return voltmat

#solved the optimization problem R=argmin_{R' in SO(3)} [R'B1-B2]_{F2}
def findrotmat(phasormat1,phasormat2,method="Inverse"):
    if method=="Inverse":
        rotmat=phasormat2.dot(pinv(phasormat1))
    if method=="Procrustes":
        U,S,Vt=svd(np.real(phasormat1.dot(phasormat2.transpose().conj())))
        d=det(U.dot(Vt))
        rotmat=Vt.transpose().dot(np.diag([1,1,d]).dot(U.transpose()))
    return rotmat

def plot_dataframe_dft(dataframe,N=5000):
    fig1=plt.figure()
    plt.subplot(211)
    plt.plot(dataframe[:,0],color="red",alpha=0.5)
    plt.plot(dataframe[:,1],color="green",alpha=0.5)
    plt.plot(dataframe[:,2],color="blue",alpha=0.5)
    plt.plot(dataframe[:,3],color="gray",alpha=0.5)
    #plt.plot(data[:,3],color="gray")
    plt.xlabel("Sample No")
    plt.ylabel("Voltage [V]")
    plt.ylim([-5,5])
    plt.subplot(212)
    plt.plot(dataframe[:,0],color="red",alpha=0.5)
    plt.plot(dataframe[:,1],color="green",alpha=0.5)
    plt.plot(dataframe[:,2],color="blue",alpha=0.5)
    plt.plot(dataframe[:,3],color="gray",alpha=0.5)
    #plt.plot(data[:,3],color="gray")
    plt.xlabel("Sample No")
    plt.ylabel("Voltage [V]")
    plt.xlim([500,1000])
    plt.ylim([-5,5])

    yticks=[10**(-n) for n in range(0,8)]
    data_dft=extract_windowed_pdft(dataframe,"blackman")

    fig2=plt.figure()
    df=50
    plt.subplot(411)
    plt.semilogy(1e-3*df*np.arange(0,N),np.abs(data_dft[:,0]),color="red")
    plt.grid()
    plt.xlim([0,50])
    plt.yticks(yticks)
    plt.ylim([1e-7,1e-1])
    plt.ylabel("$V_x$   [V/$\sqrt{Hz}$]")
    plt.subplot(412)
    plt.semilogy(1e-3*df*np.arange(0,N),np.abs(data_dft[:,1]),color="green")
    plt.grid()
    plt.xlim([0,50])
    plt.yticks(yticks)
    plt.ylim([1e-7,1e-1])
    plt.ylabel("$V_y$   [V/$\sqrt{Hz}$]")
    plt.subplot(413)
    plt.semilogy(1e-3*df*np.arange(0,N),np.abs(data_dft[:,2]),color="blue")
    plt.grid()
    plt.xlim([0,50])
    plt.yticks(yticks)
    plt.ylim([1e-7,1e-1])
    plt.ylabel("$V_z$   [V/$\sqrt{Hz}$]")
    plt.subplot(414)
    plt.semilogy(1e-3*df*np.arange(0,N),np.abs(data_dft[:,3]),color="gray")
    plt.grid()
    plt.xlim([0,50])
    plt.yticks(yticks)
    plt.ylim([1e-7,1e-1])
    plt.ylabel("$V_{ref}$   [V/$\sqrt{Hz}$]")
    plt.xlabel("Frequency [kHz]")
    # plt.subplot(414)
    # plt.semilogy(np.abs(data_dft[:,3]),color="gray")
    # plt.yticks(yticks)
    
    return (fig1,fig2)

def plot_phasors(data_dft,ks):
    #phasor scatter plot
    for k in ks:
        plt.figure()
        plt.scatter(np.real(data_dft[:,k,0]),np.imag(data_dft[:,k,0]))
        plt.scatter(np.real(data_dft[:,k,1]),np.imag(data_dft[:,k,1]))
        plt.scatter(np.real(data_dft[:,k,2]),np.imag(data_dft[:,k,2]))
        plt.axis('equal')
        plt.xlabel("Real")
        plt.ylabel("Imaginary")
        plt.title("Phasor at k=%d"%(k))
        plt.show()


class pickup_coils:
    def __init__(self,sigpowercalfilename,phasorcalfilename,ks):
        self.ks=ks
        #load the signal power sensitivity matrices which are used to estimate the position
        with open(sigpowercalfilename,"rb") as fptr:
            #powsensmats[freqno,channelno,dirno]
            powsensmats=pkl.load(fptr)
        invpowsensmats=np.zeros((3,3,3),dtype="complex128")
        invpowsensmats[0,:,:]=pinv(powsensmats[0,:,:])
        invpowsensmats[1,:,:]=pinv(powsensmats[1,:,:])
        invpowsensmats[2,:,:]=pinv(powsensmats[2,:,:])
        self.invpowsensmats=invpowsensmats
        self.powsensmats=powsensmats
        #load the phasor sensitivity matrices which are used to estimate the orientation
        with open(phasorcalfilename,"rb") as fptr:
            #phasorsensmats[freqno,channelno,dirno]
            phasorsensmats=pkl.load(fptr)
        invphasorsensmats=np.zeros((3,3,3),dtype="complex128")
        invphasorsensmats[0,:,:]=pinv(phasorsensmats[0,:,:])
        invphasorsensmats[1,:,:]=pinv(phasorsensmats[1,:,:])
        invphasorsensmats[2,:,:]=pinv(phasorsensmats[2,:,:])
        self.invphasorsensmats=invphasorsensmats
        self.phasorsensmats=phasorsensmats
        
    #get voltage phasor matrix from a file
    def getvoltmatfromfile(self,filename):
        with open(filename,"rb") as fptr:
            data=pkl.load(fptr)
        return getvoltphasormat(data,self.ks)
        
    def getphasormat(self,voltphasormat):
        return np.array([self.invphasorsensmats[freqno,:,:].dot(voltphasormat[:,freqno]) for freqno in range(0,3)]).transpose()
    
    def getphasorsensmats(self):
        return self.phasorsensmats
    
    def getsignalpowers(self,voltphasormat,use_phasormat=True):
        #fieldphasormat[channelno,freqno]
        if use_phasormat:
            phasormat=self.getphasormat(voltphasormat)
            return norm(phasormat,axis=0)
        fieldphasormat=np.zeros((3,3),dtype="complex128")
        fieldphasormat[:,0]=self.invpowsensmats[0,:,:].dot(voltphasormat[:,0])
        fieldphasormat[:,1]=self.invpowsensmats[1,:,:].dot(voltphasormat[:,1])
        fieldphasormat[:,2]=self.invpowsensmats[2,:,:].dot(voltphasormat[:,2])
        return norm(fieldphasormat,axis=0)