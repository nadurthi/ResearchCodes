#!/usr/bin/env python

"""
Documentation for this imm module

More details.
"""

from uq.filters._basefilter import IntegratedBaseFilterStateModel, FiltererBase
import logging
import numpy as np
import copy
import uuid
from uq.stats import moments as uqstmoms

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Particles:
    def __init__(self,currt=0):
        self.ID = uuid.uuid4()
        self.currt = currt
        self.wts = []
        self.X = []
    
    def addParticle(self,X,w):
        self.X.append(X)
        self.wts.append(w)
    
    def deleteParticle(self,i):
        del self.X[i]
        del self.wts[i]

    def __getitem__(self,i):
        return self.X[i],self.wts[i]
    
    def makeCopy(self):
        particlescopy =  copy.deepcopy(self)
        particlescopy.ID = uuid.uuid4()
        return particlescopy
    
    def renormlizeWts(self):
        self.wts = self.wts/np.sum(self.wts)
        
    def getEst(self):
        return uqstmoms.MeanCov(self.X, self.wts/np.sum(self.wts))
        
    @property
    def Nsamples(self):
        return len(self.X)
    
    
    def itersamplesIdx(self):
        for i in range(len(self.X)):
            yield i
    
    def bootstrapResample(self,N=None):
        if N is None:
            N=len(self.wts)
        idxs = np.random.choice(len(self.wts),size=N, replace=True, p=self.wts)
        Xnew=[]
        for idx in idxs:
            Xnew.append( copy.deepcopy(self.X[idx]) )
        
        self.wts = np.ones(N)/N
        self.X = Xnew
        
class PF(FiltererBase):
    def __init__(self, dynModel, sensModel, recordfilterstate=False):

        super().__init__(dynModel, sensModel, recordfilterstate=recordfilterstate)


# function[Xnew, wnew] = Reweightpts(
#     Nmoms, Xprior, wprior, hx, R, ymeas, Xpost, wpost)
# % [y, M] give the moments of the posterior to be captured by weights
# % pts position X stays the same.
# % postfunX is posterior pdf evaluations at pts X
# dim = size(Xpost, 2);
# N = length(wpost);


# % % posterior moments
# % y = [];
# % M = [];
# py = 0;
# ss = zeros(size(Xpost, 1), 1);
# invR = inv(R);
# y = zeros(1, size(Xpost, 2));
# M = 1;
# for i = 1: 1: N
#     hh = hx(Xprior(i, : ));
#     ss(i) = -0.5*(ymeas(:)'-hh(:)')*invR*(ymeas(: )'-hh(:)')';
#     py = py+wprior(i)*1/sqrtm(det(2*pi*R))*exp(-0.5*(ymeas(:)'-hh(:)')*invR*(ymeas(: )'-hh(:)')');
# end

# wmod = ((wprior / py) * 1 / sqrt(det(2 * pi * R))).*exp(ss);

# n1 = 0;
# n2 = 0;
# for i = 1: 1: Nmoms
#     [yy, MM] = Cal_moments_samples(Xprior, wmod, i, 'raw');
#     y = vertcat(y, yy);
#     M = vertcat(M, MM);
#     if i == 1
#         n1 = length(MM);
#     end
#     if i == 2
#         n2 = length(MM);
#     end


# end
# [mu, P] = MeanCovPts(wmod, Xprior);
# [Xnew, wnew] = conjugate_dir_gausspts_till_8moment(mu(:), P);

# % [ypost, Mpost] = Cal_moments_samples(X, w, 2, 'raw');
# % keyboard
# % % %
# % [mupost, Ppost] = MeanCovPts(wmod, Xprior);
# % [muprior, Pprior] = MeanCovPts(wprior, Xprior);
# %
# % isqPprior = sqrtm(inv(Pprior));
# % sqPpost = sqrtm(Ppost);
# %
# %
# %
# % % XX = zeros(size(Xprior));
# % % for i = 1: 1: length(wprior)
# % %     XX(i, :) = isqPprior*(Xprior(i, : )'- muprior);
# % %     XX(i, :) = sqPpost*XX(i, : )'+ mupost;
# % % end
# % % X = XX;
# % % w = wprior;
# %
# % X = Xpost;
# % w = wpost;
# %
# % A = zeros(size(y, 1), N);
# % B = zeros(size(y, 1), 1);
# %
# %
# % for i = 1: 1: size(y, 1)
# % A(i, :) = prod(X.^repmat(y(i, : ), N, 1), 2);
# % B(i) = M(i);
# % end
# %
# % % % quad prog
# % % Aeq = A(1: (1+n1+n2), :);
# % % Beq = B(1: (1 + n1 + n2));
# % % A(1: (1+n1+n2), :) = [];
# % % B(1: (1 + n1 + n2)) = [];
# % %
# % % A = A(1: n1+n2, :);
# % % B = B(1: n1 + n2);
# %
# % % opts = optimset('Algorithm', 'interior-point-convex', 'Display', 'on');
# % % wnew = quadprog(A'*A,-A' * B, [], [], Aeq, Beq, zeros(1, N), ones(1, N));
# % wnew = quadprog(eye(N), zeros(N, 1), [], [], A, B, zeros(1, N), ones(1, N));


# # %%


# function[X_new] = PF_time_update1(X_old, model)
# %
# % Particle Filter Time Update - DISCRETE DYNAMICAL SYSTEM
# %
# % X_new - the new set of particles after propagation and noise perturb
# %
# % Gabriel Terejanu(terejanu@buffalo.edu)

# no_particles = size(X_old, 2);
# X_new = zeros(size(X_old));
# % keyboard
# for i = 1: no_particles
#     %     [t yend y] = feval(model.fx, X_old(:, i), time.tspan(k-measTs), time.tspan(k));
#     %  yend = feval(model.fx, X_old(:, i), model.para_dt);
#     if strcmp(model.dynamics, 'discrete')
#         yend = model.fx(X_old(:, i)', model.fx_dt);
#     else
#         [tt, xx] = ode45(@model.fx, [0, model.dt], X_old(: , i));
#         yend = xx(end, : );
#     end
#     [r, c] = size(yend);
#     if r == model.fn
#         yend = yend';
#     end
#     X_new(:, i) = yend' + model.sQ * randn(model.fn, 1);
#     %     Y(i, : , : ) = y;
#     %     i
# end

# % keyboard
# % for j = 1
# % for indt = 1: length(t)
# %         Momx1(indt, j) = sum(Y(: , indt, 1))/no_particles;
# %         Momx2(indt, j) = sum(Y(: , indt, 2))/no_particles;
# %         Momp1(indt, j) = sum(Y(: , indt, 3))/no_particles;
# %         Momp2(indt, j) = sum(Y(: , indt, 4))/no_particles;
# %     end
# % end
# %
# % for j = 2: Nm
# % for indt = 1: length(t)
# %             Momx1(indt, j) = sum((Y(:, indt, 1)-Momx1(indt, 1)).^j)/no_particles;
# %             Momx2(indt, j) = sum((Y(:, indt, 2)-Momx2(indt, 1)).^j)/no_particles;
# %             Momp1(indt, j) = sum((Y(:, indt, 3)-Momp1(indt, 1)).^j)/no_particles;
# %             Momp2(indt, j) = sum((Y(:, indt, 4)-Momp2(indt, 1)).^j)/no_particles;
# %     end
# %     j
# % end


# # %%


# function[X_new, w_new] = PF_meas_update(X_old, w_old, model, pf, ymeas)
# %
# % Particle Filter Measurement Update
# %
# % X_new - the new set of particles after reweighting and resampling if necessary
# %
# % Gabriel Terejanu(terejanu@buffalo.edu)

# % % ------------------------------------------------------------------------
# % weights update
# %--------------------------------------------------------------------------
# mX_pf = zeros(model.hn, pf.no_particles);
# w_new = zeros(size(w_old));
# % keyboard
# for j = 1: pf.no_particles
#     % get the measurement
# %     keyboard
# %     mX_pf(:, j) = feval(model.hx, X_old(: , j), model.para_dt);
#     mX_pf(: , j) = feval(model.hx, X_old(: , j));
#     w_new(j) = w_old(j) * getLikelihood(ymeas - mX_pf(: , j), model.R);
# end
# w_new = w_new. / sum(w_new);
# X_new = X_old;

# [m, P] = MeanCov(X_new, w_new(:));

# % % ------------------------------------------------------------------------
# % resampling
# %--------------------------------------------------------------------------
# if (pf.resample)
# 	Neff = 1 / sum(w_new. ^ 2);
#     if (Neff < pf.neff * pf.no_particles)
#         I = myresampling(w_new);
#         I = round(I);
#         for j = 1 : pf.no_particles
#             X_new(:,j) = X_old(:,I(j));
#         end;
#         % reset weights
#         w_new = ones(1,pf.no_particles)/pf.no_particles;
#     end
# end

# %% ------------------------------------------------------------------------
# % resampling
# %--------------------------------------------------------------------------
# if (pf.regularizepf)
#     D=sqrtm(P);
#     c=1.1;
#     nx=model.fn;
#     cnx=pi^(nx/2)/gamma(nx/2+1);
#     A=((8/cnx)*(nx+4)*(2*sqrt(pi))^nx)^(1/(nx+4));
#     hopt=A*(pf.no_particles)^(1/(nx+4));


#     for j = 1 : pf.no_particles
#         while(1)
#            g=mvnrnd(zeros(model.fn,1),eye(model.fn)) ;
#            u=rand;
#            l=EpanechnikovKernel(g,nx,cnx)/(c*mvnpdf(g,zeros(model.fn,1),eye(model.fn)));
#            if u<=l
#                break
#            end
#         end
#         X_new(:,j) = X_new(:,j) + hopt*D*g;
#     end

# end

# end

# function K=EpanechnikovKernel(x,nx,cnx)
# [r,c]=size(x);
# if r==1 || c==1
#    x=x(:)';
# end
# [N,nx]=size(x);
# % cnx=pi^(nx/2)/gamma(nx/2+1);
# K=zeros(N,1);
# p=(nx+2)/(2*cnx);
# for i=1:N
#     if norm(x)<1
#         K(i)=p*(1-norm(x(i,:))^2);
#     else
#         K(i)=0;
#     end
# end

# end











# # %%




# %% ------------------------------------------------------------------------
# % time
# %--------------------------------------------------------------------------
# time.t0 = 0;
# time.dt = 0.1;
# time.tf = 10;
# time.tspan = time.t0 : time.dt : time.tf;
# time.nSteps = length(time.tspan);
# %_________________________________
# %% ------------------------------------------------------------------------
# % model
# T=time.dt;

# Qf=1*diag([1,1,1,1,1]);

# Qt=0*diag([0,0,2.4064*1e-5,2.4064*1e-5,0]);

# % sigr=100*1e-3;
# % sigth=10*1e-3;

# % R=diag([sigr^2,sigth^2]);
# % R=diag([sigth^2]);



# model.fn = 5;               % state space dimensionality
# model.fx = @lorenz_cont;
# model.fx_dt =time.dt;
# model.dynamics='continuous';

# R=diag([0.5,0.5,0.5]);
# model.hn =3;               % measurement dimensionality
# model.hx =@(x,dt)[x(1);x(2);x(3)];
# model.hx_jac=@(x,dt)[eye(model.hn),zeros(3,2)];


# model.gx=@(x,dt)1;
# model.Q =Qf;
# model.sQ =sqrtm(Qf);
# model.R = R;
# model.sR=sqrtm(R);
# model.Qtruth=Qt;
# model.Qt_sq=sqrtm(Qt);
# model.dt=time.dt;

# model.propagate=1;
# model.para_dt=0;

# % model.x0tr=[6500.4,349.14,-1.8093,-6.7967,0.6932]';
# % model.P0tr=diag([1e-5,1e-5,1e-5,1e-5,0]);
# % [1.50887,-1.531271,25.46091,12,25]';
# model.x0tr=[1.50887,-1.531271,25.46091,10,28]';
# model.P0tr=diag([4,4,4,2,4]);


# %% ------------------------------------------------------------------------
# % particle filter settings
# %--------------------------------------------------------------------------
# pf.no_particles =100;
# pf.no_bins = 100;               % when computing the histogram

# pf.resample = true;             % resampling after measurement update
# pf.neff = 0.8;                  % treshold for the number of effective samples

# filter.paras_ukf_kappa=1;
# filter.paras_gh_pts=2;

# filter.freq=1; %% This is actually the number of 'dt' steps after which a meas updt is done

# % start the filter too at the true estimate
# x0f=model.x0tr;
# P0f=model.P0tr;

# filter.x0_filt_start=x0f;
# filter.P0_filt_start=P0f;

# %% Generate measurements

# % use the model's true initial state to generate measurements
# x0trr = model.x0tr;%mvnrnd(model.x0tr', model.P0tr);
# [t,x_mc]=ode45(model.fx,time.tspan,x0trr',model.Qt_sq);
# ym=zeros(time.nSteps,model.hn);
# for ii=1:1:time.nSteps
#     ym(ii,:)=(model.hx(x_mc(ii,:)')+(sqrtm(R)*randn(model.hn,1)))';
# end

# % start the filter from random point
# filter.x0_filt_start = mvnrnd(model.x0tr', model.P0tr)';


# %     filter.x0_filt_start =[1.50887,-1.531271,25.46091,12,25]';
# filter.ymeas=ym;


# %% running
# % FIlter INIitalization

# mu_pf=filter.x0_filt_start;
# P_pf=filter.P0_filt_start;

# % initial condition for the filter
# X_pf=repmat(mu_pf,1,pf.no_particles)+sqrtm(P_pf)*randn(model.fn,pf.no_particles);
# w_pf = ones(1, pf.no_particles) / pf.no_particles;



# xNNN_mupf=zeros(time.nSteps,model.fn);
# PNNN_pf=zeros(time.nSteps,model.fn^2);


# for k=2:1:time.nSteps
#     k
#     if rem(k,filter.freq)==0
#         zm=filter.ymeas(k,:)';
#     else
#         zm=-1234;
#     end

#     [X_pf,w_pf]=BOOTSTRAP_Pfilter_disc_UPDT_disc_MEAS(model,pf,X_pf,w_pf,zm);

#     [PFx,P_pf,PFminB,PFmaxB,tmp_X_pf] = getPFdata(X_pf, w_pf);

#     xNNN_mupf(k,:)=PFx';
#     PNNN_pf(k,:)=reshape(P_pf,1,model.fn^2);

# end

# plot(time.tspan,xNNN_mupf(:,1),time.tspan,xNNN_mupf(:,1)+sqrt(PNNN_pf(:,1)),time.tspan,xNNN_mupf(:,1)-sqrt(PNNN_pf(:,1)) )










# # %%

# function J = myresampling(q)
# %Systematic Resampling or Deterministic Resampling
# N = length(q);

# c = cumsum(q);

# J = zeros(1,N);


# i = 1;
# u1 = rand/N;

# for j=1:N
#     u = u1 + (j-1)/N;
#     while u>c(i)
#         i = i + 1;
#     end
#     J(j) = i;

# end


# # %%





# function [omu,osig,minB,maxB,tmp_X_pf] = getPFdata(pf_samples,pf_w)
# %
# % Get the estimates for the Particle Filter
# %
# % omu   - estimate the mean
# % osig  - estimate the covariance
# %
# % Gabriel Terejanu (terejanu@buffalo.edu)

# %% ------------------------------------------------------------------------
# % init & resampling
# %--------------------------------------------------------------------------
# no_particles = length(pf_w);

# % resampling
# I = myresampling(pf_w);
# I = round(I);
# tmp_X_pf = zeros(size(pf_samples));
# for j = 1 : no_particles
#     tmp_X_pf(:,j) = pf_samples(:,I(j));
# end;


# %% ------------------------------------------------------------------------
# % omu - compute the mean of the Particle Filter
# %--------------------------------------------------------------------------

# omu = mean(tmp_X_pf,2);
# mu2=sum(repmat(pf_w',1,5).*tmp_X_pf',1);
# XX=tmp_X_pf'-repmat(mu2,length(pf_w),1);
# cc=0;
# for i=1:1:length(pf_w)
#     cc=cc+pf_w(i)*XX(i,:)'*XX(i,:);
# end
# % sum(mu2'-omu)
# %% ------------------------------------------------------------------------
# % osig - compute the covariance of the Particle Filter
# %--------------------------------------------------------------------------
# osig = cov(tmp_X_pf');

# minB = min(tmp_X_pf')';
# maxB = max(tmp_X_pf')';











# # %%



# function l = getLikelihood(r, A)

# l = exp(-r'*inv(A)*r/2)/sqrt(det(2.*pi.*A));











# # %%


# function fs = get_seq ( d, norm )

# %*****************************************************************************80
# %
# %% GET_SEQ generates all positive integer D-vectors that sum to NORM.
# %
# %  Discussion:
# %
# %    This function computes a list, in reverse dictionary order, of
# %    all D-vectors of positive values that sum to NORM.
# %
# %    For example, fs = get_seq ( 3, 5 ) returns
# %
# %      3  1  1
# %      2  2  1
# %      2  1  2
# %      1  3  1
# %      1  2  2
# %      1  1  3
# %
# %  Licensing:
# %
# %    This code is distributed under the GNU LGPL license.
# %
# %  Modified:
# %
# %    30 May 2010
# %
# %  Author:
# %
# %    Original MATLAB version by Florian Heiss, Viktor Winschel.
# %    This MATLAB version by John Burkardt.
# %
# %  Reference:
# %
# %    Florian Heiss, Viktor Winschel,
# %    Likelihood approximation by numerical integration on sparse grids,
# %    Journal of Econometrics,
# %    Volume 144, 2008, pages 62-80.
# %
# %  Parameters:
# %
# %    Input, integer D, the dimension.
# %
# %    Input, integer NORM, the value that each row must sum to.
# %    NORM must be at least D.
# %
# %    Output, integer FS(*,D).  Each row of FS represents one vector
# %    with all elements positive and summing to NORM.
# %
#   if ( norm < d )
#     fprintf ( 1, '\n' );
#     fprintf ( 1, 'GET_SEQ - Fatal error!\n' );
#     fprintf ( 1, '  NORM = %d < D = %d.\n', norm, d );
#     error ( 'GET_SEQ - Fatal error!' );
#   end

#   seq = zeros ( 1, d );
# %
# %  The algorithm is written to work with vectors whose minimum value is
# %  allowed to be zero.  So we subtract D from NORM at the beginning and
# %  then increment the result vectors by 1 at the end!
# %
#   a = norm - d;
#   seq(1) = a;
#   fs = seq;
#   c = 1;

#   while ( seq (d) < a )

#     if ( c == d )
#       for i = (c-1) : -1 : 1
#         c = i;
#         if ( seq(i) ~= 0 )
#           break
#         end
#       end
#     end

#     seq(c) = seq(c) - 1;
#     c = c + 1;
#     seq(c) = a - sum ( seq(1:(c-1)) );

#     if ( c < d )
#       seq((c+1):d) = zeros ( 1, d - c );
#     end

#     fs = [ fs; seq ];

#   end
# %
# %  Increment all entries by 1.
# %
#   fs = fs + 1;

#   return
# end











# # %%

# function [X_pf,w_pf]=BOOTSTRAP_Pfilter_disc_UPDT_disc_MEAS(model,pf,X_pf,w_pf,ym)
# %Modified code of GABRIEL to suit my purpose

# %% Propagate Particles
#  X_pf = PF_time_update1(X_pf, model);
#  w_pf=w_pf;
#  %% CHECK MEasurement update if available
#         if ym==-1234 || any(isnan(ym)==1)
# % [PF.x{k},PF.P{k},PF.minB{k},PF.maxB{k},tmp_X_pf] = getPFdata(X_pf, w_pf);
#          return
#         end

#  %% Meas UPDT
#  [X_pf, w_pf] = PF_meas_update(X_pf, w_pf, model, pf, ym);

# %  %% GET estimates
# %  [PF.x{k},PF.P{k},PF.minB{k},PF.maxB{k},tmp_X_pf] = getPFdata(X_pf, w_pf);
# end
# # %%















# # %%















# # %%













# # %%


# # %%















# # %%















# # %%













# # %%
