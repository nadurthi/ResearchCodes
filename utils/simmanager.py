#!/usr/bin/env python
"""
Documentation for this imm module

More details.
"""
import loggerconfig as logconf
logger = logconf.getLogger(__name__)

import numpy as np
import uuid
import os
import datetime
import pickle
from git import Repo
import datetime
import os
import json
import pickle
import dill
from shutil import copyfile


# for submodule in repo.submodules:
#     print(submodule)
#     diff=submodule.module().git.diff('HEAD~1..HEAD', name_only=True)
#     print(diff)



class SimManager:
    def __init__(self,t0=0,tf=100,dt=0.5,dtplot=0.1,simname=None,savepath=None,workdir=None):
        self.ID = uuid.uuid4()
        self.simname = simname
        self.savepath = savepath
        self.createdtime  = datetime.datetime.now()
        
        self.data={}
        
        self.workdir = workdir

        self.t0 = t0
        self.tf = tf
        self.dt = dt
        self.tvec = np.arange(self.t0,self.tf,dt)
        self.tvecplot = np.arange(self.t0,self.tf,dtplot)

        logger.critical('critical message') 
        logger.timing('timing message',{'funcName':"funny2",'funcArgstr':"None22",'timeTaken':222.33})
        
        self.repo = Repo(workdir)
        assert not self.repo.bare

    @property
    def ntimesteps(self):
        return len(self.tvec)

    def iteratetimesteps(self,skipto=0):
        for i in range(skipto,len(self.tvec)-1):
            yield self.tvec[i],i,self.tvec[i+1]-self.tvec[i]
    

            
    def initialize(self,repocheck=False):
        self.createdtime  = datetime.datetime.now()
        
        cnt = len([ss for ss in os.listdir(self.savepath) if os.path.isdir(os.path.join(self.savepath,ss)) and self.simname in ss])
        self.simname = "_".join([self.simname,str(cnt),
                        self.createdtime.strftime("%Y-%m-%d-%HH-%MM-%Ss")])

        self.fullpath = os.path.join(self.savepath,self.simname)
        self.figpath = os.path.join(self.fullpath,'figures')
        self.dillsessionpath = os.path.join(self.fullpath,'dillsession.dill')
        self.metafilepath = os.path.join(self.fullpath,"metalog.txt")
        self.debugstatusfilepath = os.path.join(self.fullpath,"debugstatuslog.txt")

        self.debugpath = os.path.join(self.fullpath,"debugData")

        self.repocommitfilepath = os.path.join(self.fullpath,"repocommitlog.txt")
        self.datapath = os.path.join(self.fullpath,'data.dill')
        self.simmanagerpath = os.path.join(self.fullpath,'simmanager.dill')

        if not os.path.exists(self.fullpath):
            os.makedirs(self.fullpath)


        if not os.path.exists(self.figpath):
            os.makedirs(self.figpath)



        # git check
        try:
            diff = self.repo.git.diff('HEAD~1..HEAD', name_only=True)
            difffiles = diff.split('\n')
            print(difffiles)
    
            if repocheck:
                if self.repo.is_dirty():
                    input("Repo is dirty, COmmit and run this again, \n Do you want to continue\n")
        except:
            print('No git in this folder')
            
            
    def finalize(self):
        self.repocommits={}
        try:
            self.repocommits['main'] = str(self.repo.head.commit)
            # repo.submodules[0].module().head.commit
            # for submodule in self.repo.submodules:
            #     self.repocommits[str(submodule)] = str(submodule.module().head.commit)
        except:
            print('No git in this folder to finalize')
    def addfolder(self,foldername):
        pass

    def compress(self):
        pass

    def savefigure(self,fig,pather,fname,data=None):
        pather= [str(pp) for pp in pather]
        patherpath = os.path.join(self.figpath,*pather)
        datapatherpath = os.path.join(self.figpath,*pather,"figdata")
        if not os.path.exists(patherpath):
            os.makedirs(patherpath)
        if not os.path.exists(datapatherpath):
            os.makedirs(datapatherpath)

                
        fpath = os.path.join(patherpath,fname)
        fig.savefig(fpath,format='png',bbox_inches='tight',dpi=600)
        
        if data is not None:
            dpath = os.path.join(datapatherpath,fname)
            with open(dpath,'wb') as FF:
                pickle.dump(data,FF)
                
    def pickledata(self,data,pather,fname):
        pather= [str(pp) for pp in pather]
        patherpath = os.path.join(self.fullpath,*pather)
        if not os.path.exists(patherpath):
            os.makedirs(patherpath)
        fpath = os.path.join(patherpath,fname)
        with open(fpath,'wb') as FF:
            pickle.dump(data,FF)
        
    def save(self,metalog,**kwargs):
        self.savedtime  = datetime.datetime.now()


        with open(self.metafilepath,'w') as f:
            f.write(metalog)


        with open(self.repocommitfilepath,'w') as f:
            f.write(json.dumps(self.repocommits,indent=4, sort_keys=True) )


        with open(self.datapath,'wb') as f:
            dill.dump(kwargs, f, protocol=pickle.HIGHEST_PROTOCOL)


        with open(self.simmanagerpath,'wb') as f:
            dill.dump(self,f,protocol=pickle.HIGHEST_PROTOCOL)
        
        if 'mainfile' in kwargs:
            filename = os.path.split(kwargs['mainfile'])[-1]
            origdir = os.path.split(kwargs['mainfile'])[0]
            copyfile(kwargs['mainfile'], os.path.join(self.fullpath,filename) )
        
    def copyfigs(self,tofolder,pretag):
        pass
    
    def summarize(self):
        print("ID = ",self.ID)
        print("simname = ",self.simname)
        print("savepath = ",self.savepath)
        print("created-time = ",self.createdtime)
        print("working-dir = ", self.workdir)

        print(self.t0)
        print(self.tf)
        print("dt = ",self.dt)
        print("n-timestep = ",len(self.tvec))
        
        print("fullpath = ",self.fullpath)
        print("figpath = ",self.figpath)
        print("dillsessionpath = ",self.dillsessionpath)
        print("metafilepath = ", self.metafilepath)
        print("debugstatusfilepath = ", self.debugstatusfilepath)

        print("debugpath = ",self.debugpath)

        print("repocommitfilepath = ",self.repocommitfilepath)
        print("datapath = ",self.datapath)
        print("simmanagerpath = ",self.simmanagerpath)
        
        
    @staticmethod
    def load(mainfolder,G):
        datafile = os.path.join(mainfolder,'data.dill')
        with open(datafile,'rb') as F:
            D = dill.load(F)

        for key in D:
            G[key] = D[key]





