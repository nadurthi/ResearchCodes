# -*- coding: utf-8 -*-

import numpy as np


class BinArrayCounter:
    def __init__(self,n,constraintfunc=lambda x: np.sum(x)<=1):
        self.n = n
        self.constraintfunc = constraintfunc
        self.a=np.zeros(n)
        self.rstcnt = 0

    def resetcnt(self):
        self.rstcnt = 0


    def read(self):
        return self.a,self.rstcnt

    def getAndIncrement(self):



        b = self.a.copy()
        rstc = self.rstcnt
        self.increment()


        return b,rstc

    def increment(self):
        while True:

            self.a[0]=self.a[0] + 1
            for i in range(self.n-1):
                if self.a[i]>1:
                    self.a[i+1] = self.a[i+1] + 1
                    self.a[i] = 0

            if self.a[self.n-1] > 1:
                self.a[self.n-1] = 0
                self.rstcnt = self.rstcnt + 1

            if self.constraintfunc(self.a):
                break

        return self.a,self.rstcnt

class BinMatrixCounter:
    def __init__(self,n,m,rowconstraintfunc=lambda x: np.sum(x)<=1,colconstraintfunc=lambda x: np.sum(x)<=1):
        """
        binary arrays are created row-wise
        """
        self.n = n
        self.m = m
        self.rowconstraintfunc = rowconstraintfunc
        self.colconstraintfunc = colconstraintfunc
        self.Clist = [BinArrayCounter(m,constraintfunc=rowconstraintfunc) for i in range(n) ]
        self.rstcnt = 0

#        self.C = [bc.getAndIncrement() for bc in self.Clist]

    def read(self):
        C= []
        for bc in self.Clist:
            c,_ = bc.read()
            C.append(c)
        C = np.array(C)
        return C,self.rstcnt

    def increment(self):
        """
        row constraints are already checked, just check for col constraints
        """
        while True:
            self.Clist[0].increment()
            _,rscnt = self.Clist[0].read()

            for i in range(self.n-1):
                if rscnt>0:
                    self.Clist[i].resetcnt()
                    self.Clist[i+1].increment()
                    _,rscnt = self.Clist[i+1].read()

            _,rscnt = self.Clist[self.n-1].read()
            if rscnt>0:
                self.Clist[self.n-1].resetcnt()
                self.rstcnt = self.rstcnt + 1

            C,_ = self.read()
            flg=[]
            for i in range(self.m):
                flg.append( self.colconstraintfunc(C[:,i]) )

            if all(flg):
                break



    def getAndIncrement(self):
        """
        row constraints are already checked, just check for col constraints
        """
        C,rscnt = self.read()
        B=C.copy()
        self.increment()
        return B,rscnt