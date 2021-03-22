import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import systemID.timedataProcessors as sysIDtdp
import numpy.linalg as nplinalg
pd.options.display.float_format = '{:.5f}'.format
from scipy.spatial.transform import Rotation as sctrR
from scipy.optimize import minimize
from scipy import interpolate
from scipy.optimize import least_squares
from scipy.linalg import sqrtm
import scipy.integrate
from scipy.integrate import cumtrapz
import quaternion
import copy

gvec=np.array([0,0,-9.81])


def w_opt_2_imu(wo,cw,R1o,bw):
    if wo.ndim==1:
        wo=wo.reshape(1,-1)
        
    w1=np.zeros_like(wo)
    for i in range(wo.shape[0]):
        w1[i,:]=cw*R1o.dot(wo[i,:])+bw
        
    return w1

def a_opt_2_imu(ao,quat_go,wo,alphao,ro1,ca,R1o,ba):
    if ao.ndim==1:
        ao=ao.reshape(1,-1)
    if quat_go.ndim==1:
        quat_go=quat_go.reshape(1,-1)
    if wo.ndim==1:
        wo=wo.reshape(1,-1)
    if alphao.ndim==1:
        alphao=alphao.reshape(1,-1)
    
    a1 = np.zeros_like(ao)
    for i in range(ao.shape[0]):  
        Rgo = quaternion.as_rotation_matrix(quaternion.from_float_array(quat_go[i,:]))
        Rog = nplinalg.inv(Rgo)
        a1g = ao[i,:]+Rgo.dot( np.cross(wo[i,:],np.cross(wo[i,:],ro1))+np.cross(alphao[i,:],ro1) )
        RR = np.matmul(R1o,Rog)

        a1[i,:]=ca*RR.dot(a1g-gvec)+ba
    
    return a1

def estimate_gyro_params(x,Wo,W1):
    thz = x[0]
    thy = x[1]
    thx = x[2]
    
    cw = x[3]
    bw = x[4:7]
    
    r1 = sctrR.from_euler('zyx', [thz, thy, thx], degrees=False)
    R = r1.as_matrix()
    # e=np.zeros(Wo.shape[1])
    e = []
    for i in range(Wo.shape[0]):
        e1=W1[i,:]-cw*R.dot(Wo[i,:])-bw
        # e=e+e1.dot(e1)
        e.append(e1)
    return np.hstack(e)

def estimate_accel_params(x,A1,Wo,quat_go,Alphao,R1o,aog):
    ro1 = x[0:3]
    
    
    ca = x[3]
    # ca = 1
    # ba =np.zeros(3)
    ba = x[4:7]

    # e=np.zeros(Wo.shape[1])
    e = []
    for i in range(Wo.shape[0]):
        Rgo = quaternion.as_rotation_matrix(quaternion.from_float_array(quat_go[i,:]))
        # r = sctrR.from_euler('zyx', quat_go[i,:], degrees=False)
        # Rgo = r.as_matrix()
        Rog = nplinalg.inv(Rgo)
        wo = Wo[i,:]
        alphao = Alphao[i,:]
        a1g = aog[i,:]+Rgo.dot( np.cross(wo,np.cross(wo,ro1))+np.cross(alphao,ro1) )
        RR = np.matmul(R1o,Rog)
        # print(A1[i,:])
        # print(a1g-gvec)
        # print(RR.dot(a1g-gvec))
        e1 = A1[i,:]-ca*RR.dot(a1g-gvec)-ba
        
        # e=e+e1.dot(e1)
        e.append(e1)
    return np.hstack(e)

class RigidBodyCalibrator:
    # to use quaternion package: 1st elemetn is scalr component
    # in opttrack, qw is the scalar
    # The corresponding Rotation matrix is nothing but Rgo .... o to g
    def __init__(self,dopt,dimus):
        self.dopt=dopt.copy()
        self.dimus=copy.deepcopy(dimus)
        
        self.nimus = len(dimus)
        self.calib=[{} for i in range(self.nimus)]
    
    def load_default_calib(self):
        self.calib[0]={'R_imu_opt':np.identity(3),'cw':1,'bw':np.zeros(3),'Q':0.1**2*np.identity(3),
                       'rvec_opt_to_imu_optframe':np.zeros(3),'ca':1,'ba':np.zeros(3)}
        self.calib[1]={'R_imu_opt':np.identity(3),'cw':1,'bw':np.zeros(3),'Q':0.1**2*np.identity(3),
                       'rvec_opt_to_imu_optframe':np.zeros(3),'ca':1,'ba':np.zeros(3)}
        
    def initialize(self):
        t0=[self.dopt['t'].min()]
        for i in range(self.nimus):
            t0.append(self.dimus[i]['t'].min())
        self.t0=np.min(t0)
        self.dopt['tvec']=self.dopt['t']-self.t0
        
        for i in range(self.nimus):
            self.dimus[i]['tvec']=self.dimus[i]['t']-self.t0
    
    def plotopt(self,method='spline'):
        fig=plt.figure("Angular rates :"+method)
        cols=['qx','qy','qz','wx','wy','wz','alphax','alphay','alphaz']
        for i in range(1,len(cols)+1):
            ax = fig.add_subplot(3,3,i)
            i=i-1
            ax.plot(self.dopt['tvec'],self.dopt[cols[i]+'_'+method],label=cols[i]+'-opt')
            if cols[i] in ['qx','qy','qz']:
                ax.plot(self.dopt['tvec'],self.dopt[cols[i]],label=cols[i]+'-opt-true')
                
            ax.legend()
        
        fig=plt.figure("linear rates: "+method)
        cols=['x','y','z','vx','vy','vz','ax','ay','az']
        for i in range(1,len(cols)+1):
            ax = fig.add_subplot(3,3,i)
            i=i-1
            ax.plot(self.dopt['tvec'],self.dopt[cols[i]+'_'+method],label=cols[i]+'-opt')
            if cols[i] in ['x','y','z']:
                ax.plot(self.dopt['tvec'],self.dopt[cols[i]],label=cols[i]+'-opt-true')
                
            ax.legend()
            
        plt.show()
    
    def plotimu(self):
        for i in range(self.nimus):
            fig=plt.figure("IMU #%d"%i)
            cols = ['ax','ay','az','wx','wy','wz']
            for j in range(1,len(cols)+1):
                ax = fig.add_subplot(2,3,j)
                j=j-1
                ax.plot(self.dimus[i]['tvec'],self.dimus[i][cols[j]],label=cols[j]+'-imu')
                ax.legend()
        plt.show()
        
    def plot_imu_opt_with_calib(self,method='spline'):
        fig=plt.figure("Angular rates :"+method)
        cols=['wx','wy','wz']
        colsmethod = [s+'_'+method for s in cols]
        wo=self.dopt[colsmethod].values
        
        for j in range(self.nimus):
            cw = self.calib[j]['cw']
            bw = self.calib[j]['bw']
            R1o = self.calib[j]['R_imu_opt']
            woimu=w_opt_2_imu(wo,cw,R1o,bw)
            
            for i in range(len(cols)):
                ax = fig.add_subplot(self.nimus,3,(self.nimus+1)*j+i+1)
                ax.plot(self.dimus[j]['tvec'],self.dimus[j][cols[i]],label=cols[i]+'-imu #%d'%j)
                ax.plot(self.dopt['tvec'],woimu[:,i],label=cols[i]+'-opt2imu #%d'%j)
                ax.legend()
        
        
        fig=plt.figure("Linear rates :"+method)
        cols=['ax','ay','az']
        colsmethod = [s+'_'+method for s in cols]
        ao=self.dopt[colsmethod].values
        quat_go = self.dopt[['qw','qx','qy','qz']].values
        alphacols = [s+'_'+method for s in ['alphax','alphay','alphaz']]
        alphao = self.dopt[alphacols].values
        
        for j in range(self.nimus):
            ca = self.calib[j]['ca']
            ba = self.calib[j]['ba']
            R1o = self.calib[j]['R_imu_opt']
            ro1 = self.calib[j]['rvec_opt_to_imu_optframe']
            aoimu = a_opt_2_imu(ao,quat_go,wo,alphao,ro1,ca,R1o,ba)
            
            for i in range(len(cols)):
                ax = fig.add_subplot(self.nimus,3,(self.nimus+1)*j+i+1)
                ax.plot(self.dimus[j]['tvec'],self.dimus[j][cols[i]],label=cols[i]+'-imu #%d'%j)
                ax.plot(self.dopt['tvec'],aoimu[:,i],label=cols[i]+'-opt2imu #%d'%j)
                ax.legend()
                
        plt.show()
        
        
    def get_time_statistics(self):
        print("Optitrack time statistics")
        plt.figure("Optitrack")
        self.dopt['t'].diff().hist(bins=100)
        self.dopt['dt']=self.dopt['t'].diff()
        print("mean dt: ",np.mean(self.dopt['dt']))
        print("min dt: ",min(self.dopt['dt']))
        print("max: ",max(self.dopt['dt']))
        print("#>0.03: ",len(self.dopt['dt'][self.dopt['dt']>0.03]))
        
        for i in range(self.nimus):
            plt.figure("IMU #%d"%i)
            print("IMU #%d time statistics"%i)
            self.dimus[i]['t'].diff().hist(bins=100)
            self.dimus[i]['dt']=self.dimus[i]['t'].diff()
            print("mean dt: ",np.mean(self.dimus[i]['dt']))
            print("min dt: ",min(self.dimus[i]['dt']))
            print("max: ",max(self.dimus[i]['dt']))
            print("#>0.02: ",len(self.dimus[i]['dt'][self.dimus[i]['dt']>0.02]))
        plt.show()
        
    def estimate_true_rates_spline(self,k=3,s=0.002):
        qvec=self.dopt[['qw','qx','qy','qz']].values
        tvec=self.dopt['tvec'].values
        
        tvec,qvec_sp,w,qdot,alpha=sysIDtdp.quat2omega_scipyspline(tvec,qvec,k=k,s=s)
        tag='_spline'
        self.dopt['qw'+tag] = qvec_sp[:,0]
        self.dopt['qx'+tag] = qvec_sp[:,1]
        self.dopt['qy'+tag] = qvec_sp[:,2]
        self.dopt['qz'+tag] = qvec_sp[:,3]
        
        self.dopt['qdotw'+tag] = qdot[:,0]
        self.dopt['qdotx'+tag] = qdot[:,1]
        self.dopt['qdoty'+tag] = qdot[:,2]
        self.dopt['qdotz'+tag] = qdot[:,3]
        
        self.dopt['ww'+tag] = w[:,0]
        self.dopt['wx'+tag] = w[:,1]
        self.dopt['wy'+tag] = w[:,2]
        self.dopt['wz'+tag] = w[:,3]
        
        self.dopt['alphaw'+tag] = alpha[:,0]
        self.dopt['alphax'+tag] = alpha[:,1]
        self.dopt['alphay'+tag] = alpha[:,2]
        self.dopt['alphaz'+tag] = alpha[:,3]
        
        for st in ['x','y','z']:
            Xeval=sysIDtdp.derivative_spline(tvec,self.dopt[st].values,teval=tvec,orders=2,k=k,s=s)
            self.dopt[st+tag]=Xeval[0]
            self.dopt['v'+st+tag]=Xeval[1]
            self.dopt['a'+st+tag]=Xeval[2]
        

        
    def estimate_true_rates_poly(self,win=15,poly=5):
        qvec=self.dopt[['qw','qx','qy','qz']].values
        tvec=self.dopt['tvec'].values
        
        tvec,qvec_sp,w,qdot,alpha=sysIDtdp.quat2omega_poly(tvec,qvec,win=win,poly=poly)
        tag='_poly'
        self.dopt['qw'+tag] = qvec_sp[:,0]
        self.dopt['qx'+tag] = qvec_sp[:,1]
        self.dopt['qy'+tag] = qvec_sp[:,2]
        self.dopt['qz'+tag] = qvec_sp[:,3]
        
        self.dopt['qdotw'+tag] = qdot[:,0]
        self.dopt['qdotx'+tag] = qdot[:,1]
        self.dopt['qdoty'+tag] = qdot[:,2]
        self.dopt['qdotz'+tag] = qdot[:,3]
        
        self.dopt['ww'+tag] = w[:,0]
        self.dopt['wx'+tag] = w[:,1]
        self.dopt['wy'+tag] = w[:,2]
        self.dopt['wz'+tag] = w[:,3]
        
        self.dopt['alphaw'+tag] = alpha[:,0]
        self.dopt['alphax'+tag] = alpha[:,1]
        self.dopt['alphay'+tag] = alpha[:,2]
        self.dopt['alphaz'+tag] = alpha[:,3]
        

        for st in ['x','y','z']:
            Xeval=sysIDtdp.derivative_poly(tvec,self.dopt[st].values,teval=tvec,orders=2,win=win,poly=poly)
            self.dopt[st+tag]=Xeval[0]
            self.dopt['v'+st+tag]=Xeval[1]
            self.dopt['a'+st+tag]=Xeval[2]
    
    def calib_imus(self,t0,tf,method='spline'):
        wcols = [s+'_'+method for s in ['wx','wy','wz']]
        Wo = self.dopt[(self.dopt['tvec']>t0) & (self.dopt['tvec']<tf)][wcols].values
        to = self.dopt[(self.dopt['tvec']>t0) & (self.dopt['tvec']<tf)]['tvec'].values
        
        for i in range(self.nimus):

            dimu = self.dimus[i]
            ti = dimu[(dimu['tvec']>(t0-1)) & (dimu['tvec']<(tf+1) )]['tvec'].values
            W=[]
            for s in ['wx','wy','wz']:
                w = dimu[(dimu['tvec']>(t0-1)) & (dimu['tvec']<(tf+1) )][s].values
                f = interpolate.interp1d(ti, w)
                W.append(f(to))
                        
            W=np.vstack(W).T
            
            x0 = [0,0,0,1,0,0,0]
            # res = minimize(estimate_gyro_params, x0,args=(Wo,W1), method='Nelder-Mead', options={'gtol': 1e-6, 'disp': True})
            res = least_squares(estimate_gyro_params, x0,args=(Wo,W),method='lm')
            print("IMU#%d w: success = "%i,res.success)
            thz = res.x[0]
            thy = res.x[1]
            thx = res.x[2]
            
            cw = res.x[3]
            bw = res.x[4:7]
            
            r = sctrR.from_euler('zyx', [thz, thy, thx], degrees=False)
            R1o = r.as_matrix()
            Q=np.cov(res.fun.reshape(-1,3).T)
            
            e=res.fun.reshape(-1,3)
            fig=plt.figure("Noise Errors IMU #%d"%i)
            for j in range(3):
                ax=fig.add_subplot(self.nimus,3,1+j)
                ax.plot(to,e[:,j],label="w")
                
            self.calib[i]['R_imu_opt'] = R1o
            self.calib[i]['cw'] = cw
            self.calib[i]['bw'] = bw            
            self.calib[i]['Qw'] = Q 
            self.calib[i]['Ew'] = e
            
        # Now accelerations
        wcols = [s+'_'+method for s in ['wx','wy','wz']]
        alphacols = [s+'_'+method for s in ['alphax','alphay','alphaz']]
        accelcols = [s+'_'+method for s in ['ax','ay','az']]
        dd = self.dopt[(self.dopt['tvec']>t0) & (self.dopt['tvec']<tf)]
        Wo = dd[wcols].values
        to = dd['tvec'].values
        quat_go = dd[['qw','qx','qy','qz']].values
        Alphao = dd[alphacols].values
        aog = dd[accelcols].values
        
        

        for i in range(self.nimus):

            dimu = self.dimus[i]
            ti = dimu[(dimu['tvec']>(t0-1)) & (dimu['tvec']<(tf+1) )]['tvec'].values
            A=[]
            for s in ['ax','ay','az']:
                a = dimu[(dimu['tvec']>(t0-1)) & (dimu['tvec']<(tf+1) )][s].values
                f = interpolate.interp1d(ti, a)
                A.append(f(to))
                        
            A = np.vstack(A).T
            
            R_imu_opt = self.calib[i]['R_imu_opt']
            x0 = [0.05,0.05,0.05,1,0,0,0]
            res = least_squares(estimate_accel_params, x0,args=(A,Wo,quat_go,Alphao,R_imu_opt,aog),method='lm')
            print("IMU#%d a: success = "%i,res.success)
            rvec_opt_to_imu_optframe = res.x[0:3]  
            ca = res.x[3]
            ba = res.x[4:7]
            
            Q=np.cov(res.fun.reshape(-1,3).T)
            e=res.fun.reshape(-1,3)
            fig=plt.figure("Noise Errors IMU #%d"%i)
            for j in range(3):
                ax=fig.add_subplot(self.nimus,3,4+j)
                ax.plot(to,e[:,j],label="a")
                
            self.calib[i]['rvec_opt_to_imu_optframe'] = rvec_opt_to_imu_optframe
            self.calib[i]['ca'] = ca
            self.calib[i]['ba'] = ba            
            self.calib[i]['Qa'] = Q 
            self.calib[i]['Ea'] = e
        plt.show()
        