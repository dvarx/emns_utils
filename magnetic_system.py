# -*- coding: utf-8 -*-
"""
This file contains the `magnetic_system` class which is used for performing computations related to electromagnetic actuation
"""
import numpy as np
import matplotlib.pyplot as plt
from math import floor,pi,sin,cos
from numpy.linalg import pinv,norm
import pickle
from numpy.linalg import norm,svd
from scipy.interpolate import RegularGridInterpolator

class magnetic_system:
    """
    This class can be used to obtain magnetic field data from Comsol simulation files.
    """
    
    """
    Args:
        calibration_data_files:     list containing path of three csv files generated by comsol containing the Bx, By and Bz field values
        calibration_type:           "comsol_csv" , "pickle" , 
        N:                          number of samples in the grid, should be the same for all directions
    """    
    def __init__(self,calibration_data_files,calibration_type,Nx,Ny,Nz,Ncoils,posunit="m"):
        if(calibration_type=="comsol_csv"):
            self.Bs,self.coordinates=self.get_magfield_data_comsol(calibration_data_files,Nx,Ny,Nz,Ncoils)
        else:
            self.Bs,self.coordinates=self.get_magfield_data_sampled(calibration_data_files,Ncoils)
        if posunit=="mm":
            self.coordinates*=1e-3
        self.Ncoils=Ncoils
        #extract the coordinate values of the grid
        self.xcoords=self.extract_unique_vals_tol(self.coordinates[:,0])
        self.xcoords.sort()
        self.deltax=self.xcoords[1]-self.xcoords[0]
        self.xmin=min(self.xcoords)
        self.xmax=max(self.xcoords)
        self.ycoords=self.extract_unique_vals_tol(self.coordinates[:,1])
        self.ycoords.sort()
        self.deltay=self.ycoords[1]-self.ycoords[0]
        self.ymin=min(self.ycoords)
        self.ymax=max(self.ycoords)
        self.zcoords=self.extract_unique_vals_tol(self.coordinates[:,2])
        self.zcoords.sort()
        self.deltaz=self.zcoords[1]-self.zcoords[0]
        self.zmin=min(self.zcoords)
        self.zmax=max(self.zcoords)
        self.BactInterpolator=RegularGridInterpolator((self.xcoords,self.ycoords,self.zcoords),self.Bs)
        #compute the gradient tensor
        self._compute_gradient_tensor()
        self.BxInterpolator=RegularGridInterpolator((self.xcoords,self.ycoords,self.zcoords),self.Bx)
        self.ByInterpolator=RegularGridInterpolator((self.xcoords,self.ycoords,self.zcoords),self.By)
        self.BzInterpolator=RegularGridInterpolator((self.xcoords,self.ycoords,self.zcoords),self.Bz)
        print("initialized")  

    def getBact(self,pos):
        """returns the actuation matrix at a certain position

        Args:
            pos (list,ndarray): position at which the actuation matrix is queried in [m]

        Returns:
            ndarray: actuation matrix in units [T/A]
        """
        return self.BactInterpolator(pos)

    def getCoords(self):
        """returns the coordinate values of the regular grid on which the magnetic field is sampled

        Returns:
            tuple of list: (xcoords,ycoords,zcoords)
        """
        return (self.xcoords,self.ycoords,self.zcoords)

    def extract_unique_vals_tol(self,arr):
        return list(set(list([1e-6*round(1e6*c) for c in arr])))

    def getBArray(self):
        return self.Bs

    def _linidx2volidx(self,n):
        return (n%self.Nx,floor((n%(self.Nx*self.Ny))/self.Nx),floor(n/(self.Nx*self.Ny)))

    def _volidx2linidx(nx,ny,nz,N):
        return nz*N**2+ny*N+nx

    def permutate_coils(self,coilvec):
        """
        Permutates the coils to the new order given by coilvec (e.g. coilvec=[2,1,0,3,4,5] will permutate coil 0 and 2)
        """
        #create a copy of the field array and permutate it
        Bs_temp=np.copy(self.Bs)
        for idx in range(0,6):
            Bs_temp[:,:,:,:,idx]=self.Bs[:,:,:,:,coilvec[idx]]
        #update internal variables
        self.Bs=Bs_temp
        self.BactInterpolator=RegularGridInterpolator((self.xcoords,self.ycoords,self.zcoords),self.Bs)
        self._compute_gradient_tensor()

    def getDerMatrices(self,pos):
        """returns the spatial derivatives of the actuation matrix in x-, y- and z-direction at a certain position

        Args:
            pos (ndarray): position at which the derivatives are queried in [m]

        Returns:
            ndarray: `(dB/dx,dB/dy,dB/dz)`, a tuple containing the actuation matrix spatial derivatives of dimension `(3,Ncoils)` in units `[T/(m*A)]`
        """
        return (self.BxInterpolator(pos),self.ByInterpolator(pos),self.BzInterpolator(pos))

    def getA(self,pos,m):
        """returns the extended actuation matrix [Bact; m^T Bx; m^T By; m^t Bz] at a certain position pos

        Args:
            pos (ndarray): position where the extended actuation matrix is computed in [m]
            m (ndarray): magnetic dipole moment of the magnetic agent upon which the magnetic fields act

        Returns:
            ndarray: extended actuation matrix in units of [T], [A] and [Am^2] (for dipole moment m)
        """
        (Bx,By,Bz)=self.getDerMatrices(pos)
        Bact=self.getBact(pos)
        return np.vstack((Bact,m.dot(Bx),m.dot(By),m.dot(Bz)))
        
    def _compute_gradient_tensor(self):
        """computes the internal field spatial gradients `(dB/dx,dB/dy,dB/dz)`
        """
        Nx=self.Bs.shape[0]
        Ny=self.Bs.shape[1]
        Nz=self.Bs.shape[2]
        #compute x derivatives
        self.Bx=np.zeros(self.Bs.shape)
        for nx in range(0,Nx):
            for ny in range(0,Ny):
                for nz in range(0,Nz):
                    #check if x-index corner point, use single sided difference quotient if yes
                    if nx==0:
                        self.Bx[nx,ny,nz,:,:]=(self.Bs[nx+1,ny,nz,:,:]-self.Bs[nx,ny,nz,:,:])/self.deltax
                    elif nx==Nx-1:
                        self.Bx[nx,ny,nz,:,:]=(self.Bs[nx,ny,nz,:,:]-self.Bs[nx-1,ny,nz,:,:])/self.deltax
                    else:
                        self.Bx[nx,ny,nz,:,:]=(self.Bs[nx+1,ny,nz,:,:]-self.Bs[nx-1,ny,nz,:,:])/self.deltax
        #compute y derivatives
        self.By=np.zeros(self.Bs.shape)
        for nx in range(0,Nx):
            for ny in range(0,Ny):
                for nz in range(0,Nz):
                    #check if y-index corner point, use single sided difference quotient if yes
                    if ny==0:
                        self.By[nx,ny,nz,:,:]=(self.Bs[nx,ny+1,nz,:,:]-self.Bs[nx,ny,nz,:,:])/self.deltay
                    elif ny==Ny-1:
                        self.By[nx,ny,nz,:,:]=(self.Bs[nx,ny,nz,:,:]-self.Bs[nx,ny-1,nz,:,:])/self.deltay
                    else:
                        self.By[nx,ny,nz,:,:]=(self.Bs[nx,ny+1,nz,:,:]-self.Bs[nx,ny-1,nz,:,:])/self.deltay
        #compute z derivatives
        self.Bz=np.zeros(self.Bs.shape)
        for nx in range(0,Nx):
            for ny in range(0,Ny):
                for nz in range(0,Nz):
                    #check if z-index corner point, use single sided difference quotient if yes
                    if nz==0:
                        self.Bz[nx,ny,nz,:,:]=(self.Bs[nx,ny,nz+1,:,:]-self.Bs[nx,ny,nz,:,:])/self.deltaz
                    elif nz==Nz-1:
                        self.Bz[nx,ny,nz,:,:]=(self.Bs[nx,ny,nz,:,:]-self.Bs[nx,ny,nz-1,:,:])/self.deltaz
                    else:
                        self.Bz[nx,ny,nz,:,:]=(self.Bs[nx,ny,nz+1,:,:]-self.Bs[nx,ny,nz-1,:,:])/self.deltaz

    def get_magfield_data_sampled(self,filename,Ncoils):
        """read magnetic field data from a calibration pickle file

        Args:
            filename (string): pickle file containting the calibration data in a dictionary of the form
            `{"field_measurements":<fields>,"sampled_points":<points>,"calibration_current":current}`
            Ncoils (int): Number of coil in the system

        Returns:
            tuple: (Bs,coords) where Bs is a `(Nx,Ny,Nz,Ncoils,3)` array containting the field data and coords is a `(Nx*Ny*Nz,3)` array containing the coordinates of the sampled position
        """
        fptr=open(filename,"rb")
        calibration_data=pickle.load(fptr)
        coords=calibration_data["sampled_points"]
        fields=calibration_data["field_measurements"]
        calibration_current=calibration_data["calibration_current"]
        #determine the sampling step widths and sample numbers for each direction
        deltas=[0,0,0]
        ranges=[]
        for coordidx in range(0,3):
            #only consider first 6 significant figures here
            cs=self.extract_unique_vals_tol(coords[coordidx,:])
            cs.sort()
            deltas[coordidx]=cs[1]-cs[0]
            ranges.append((min (cs),max(cs)))
        self.Nx=int(1+round((ranges[0][1]-ranges[0][0])/deltas[0]))
        self.Ny=int(1+round((ranges[1][1]-ranges[1][0])/deltas[1]))
        self.Nz=int(1+round((ranges[2][1]-ranges[2][0])/deltas[2]))
        Bs=np.zeros((self.Nx,self.Ny,self.Nz,3,Ncoils))
        #compute the 3 dimensional field array
        for ncoil in range(0,Ncoils):
            for i in range(0,len(fields[0])):
                #compute the three integer indices of the current measurement point
                c=coords[:,i]
                idx=int(round((c[0]-ranges[0][0])/deltas[0]))
                idy=int(round((c[1]-ranges[1][0])/deltas[1]))
                idz=int(round((c[2]-ranges[2][0])/deltas[2]))
                Bs[idx,idy,idz,:,ncoil]=fields[ncoil][i]
        Bs=Bs/calibration_current
        return (Bs,np.transpose(coords))

    def get_magfield_data_comsol(self,filenames,Nx,Ny,Nz,Ncoils):
        """
        reads magnetic field components from comsol exported files
        Args:
            filenames ([string]): [List of filenames containing the x, y and z components of the magnetic field]
            N ([int]): [Number of samples in each direction (should be the same for all directions)]

        Returns:
            [type]: [Tuple (Bs,coords) where Bs is a `(N,N,N,N,3,Ncoils)` array containing the field data in [T] and coords is a `(N^3,3)` array containing the field positions in [m]]
        """
        Bs=np.zeros((Nx,Ny,Nz,3,Ncoils))    # array containing field data

        for coordidx in range(0,3):
            # open the file containing Bx, By or Bz values
            fptr=open(filenames[coordidx])
            linecounter=-1

            # read the magnetic flux components from the file containing Bx, By and Bz
            for line in fptr:
                linecounter+=1

                #ignore the first 5 header lines
                if linecounter<4:
                    continue

                #third line contains coordinates
                if linecounter==4:
                    split_lines=line.split("\"")
                    coordinates=np.zeros((Nx*Ny*Nz,3))
                    
                    coordcounter=0
                    for n in range(0,len(split_lines)):
                        #the odd line fragment contain the coordinates
                        if n%2==1:
                            #extract coordinate strings
                            coord_strs=split_lines[n][48:-1].split(",")
                            coordinates[coordcounter,:]=np.array([float(coord_strs[0]),float(coord_strs[1]),float(coord_strs[2])])
                            coordcounter+=1
                            if coordcounter>=Nx*Ny*Nz:
                                break

                #the lines contain the actual simulation data
                if linecounter>4:
                    split_frags=line.split(",")
                    for n in range(0,len(split_frags)):
                        if n<Ncoils:
                            pass
                        else:
                            linidx=n-Ncoils
                            nx=self._linidx2volidx(linidx)[0]
                            ny=self._linidx2volidx(linidx)[1]
                            nz=self._linidx2volidx(linidx)[2]
                            coilidx=linecounter-5
                            if(nz>=Nz):
                                break
                            Bs[nx,ny,nz,coordidx,coilidx]=float(split_frags[n])
            fptr.close()
            
        return (Bs,coordinates)

if __name__=="__main__":
    Ncoils=3
    system=magnetic_system(str("config/7x7x5_40mm_40mm_30mm_steelcores_threecoil/calibration.pkl"),"pickle",7,7,7,Ncoils)

    Bact=system.getBact((0,0,0))
    (dBdx,dBdy,dBdz)=system.getDerMatrices((0,0,0))