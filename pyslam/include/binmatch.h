#pragma once

#include <pybind11/eigen.h>
#include <iostream>
#include <thread>
#include <array>
#include <numeric>
#include <Eigen/Geometry>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <vector>
#include <utility>
#include <map>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <string>
#include <algorithm>
// #include <pcl/registration/gicp.h>
#include <pcl/registration/mygicp.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <cmath>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/octree/octree_search.h>
#include <nlohmann/json.hpp>
#include <queue>
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector6f = Eigen::Matrix<float, 6, 1>;
using Vector4f = Eigen::Matrix<float, 4, 1>;

using json = nlohmann::json;

Eigen::ArrayXi
numba_histogram2D(const Ref<const Eigen:MatrixXf>& X,const Ref<const VectorXf>& xedges,const Ref<const VectorXf>& yedges):
    float x_min = xedges.minCoeff();
    float x_max = xedges.maxCoeff();
    nx = xedges.size();


    y_min = yedges.minCoeff();
    y_max = yedges.maxCoeff();
    ny = yedges.size();

    H = Eigen::ArrayXi::Zero(nx-1,ny-1);

    Eigen::Array<float, 1, 2> dxy({x_max-x_min,y_max-y_min});
    Eigen::Array<float, 1, 2> xymin({x_min,y_min});
    Eigen::Array<float, 1, 2> xymax({x_max,y_max});

    Eigen::Array<float, 1, 2>  nxy({nx-1,ny-1});

    Eigen::ArrayXb inbnd= ((X.array().rowwise()-xymin)>=0 ) & ((xymax-X.array().rowwise())>=0 ).rowwise().all();

    Eigen::ArrayXi dd = nxy*((X.array().rowwise()-xymin).rowwise()/dxy).rowwise();

    for(int i=0; i< X.rows(); ++i){
        if (inbnd(i))
            H[dd(i,0),dd(i,1)]+=1
    }

    return H




@njit(cache=True)
def UpsampleMax(Hup,n):
    H=np.zeros((int(np.ceil(Hup.shape[0]/2)),int(np.ceil(Hup.shape[1]/2))),dtype=np.int32)
    for j in range(H.shape[0]):
        for k in range(H.shape[1]):
            lbx=max([2*j,0])
            ubx=min([2*j+n,Hup.shape[0]-1])+1
            lby=max([2*k,0])
            uby=min([2*k+n,Hup.shape[1]-1])+1
            H[j,k] = np.max( Hup[lbx:ubx,lby:uby] )
    return H



# @jit(int32(int32[:,:], float32[:], float32[:,:], float32[:], float32[:]),nopython=True, nogil=True,cache=True)
@njit(cache=True)
def getPointCost(H,dx,X,Oj,Tj):
    # Tj is the 2D index of displacement
    # X are the points
    # dx is 2D
    # H is the probability histogram

    Pn=np.floor((X+Oj)/dx)
    # j=np.floor(Oj/dx)
    # Pn=P+j
    # Idx = np.zeros(Pn.shape[0],dtype=np.int32)
    # for i in range(Pn.shape[0]):
    #     if Pn[i,0]<0 or Pn[i,0]>H.shape[0]-1 or Pn[i,1]<0 or Pn[i,1]>H.shape[1]-1 :
    #         Pn[i,0]=H.shape[0]-1
    #         Pn[i,1]=H.shape[1]-1

        # elif Pn[i,0]>H.shape[0]-1:
        #     Pn[i,0]=H.shape[0]-1
        # if Pn[i,1]<0:
        #     Pn[i,1]=0
        # elif Pn[i,1]>H.shape[1]-1:
        #     Pn[i,1]=H.shape[1]-1

        # Idx[i]= Pn[i,1]*(H.shape[0]-1)+Pn[i,0]



    # c=np.sum(np.take(H, Idx))
    # c=np.sum(np.take(H, np.ravel_multi_index(Pn.T, H.shape,mode='clip')))
    # idx1 = np.all(Pn>=np.zeros(2),axis=1)
    # Pn=Pn[idx1]
    # idx2 = np.all(Pn<H.shape,axis=1)
    # Pn=Pn[idx2]
    idx1=np.logical_and(Pn[:,0]>=0,Pn[:,0]<H.shape[0])
    idx2=np.logical_and(Pn[:,1]>=0,Pn[:,1]<H.shape[1])
    idx=np.logical_and(idx1,idx2 )
    c=0
    # idx=np.all(np.logical_and(Pn>=np.zeros(2) , Pn<H.shape),axis=1 )
    Pn=Pn[idx]
    # if Pn.size>0:
    #     values, counts = np.unique(Pn, axis=0,return_counts=True)
    #     c=np.sum(counts*H[values[:,0],values[:,1]])
        # c=np.sum(H[Pn[:,0],Pn[:,1]])
    for k in range(Pn.shape[0]):
        c+=H[int(Pn[k,0]),int(Pn[k,1])]

    return c


Eigen::MaxtrixXf computeHitogram2D{

}

class BinMatch{
public:
  BinMatch();
  int getcost(H,X,Oj,dx){

  }
  void computeHlevels(){

  }
  getmatch(){
    std::priority_queue<int> q;

    n=histsmudge =2 # how much overlap when computing max over adjacent hist for levels
    H21comp=np.identity(3)

    mn=np.zeros(2)
    mx=np.zeros(2)
    mn_orig=np.zeros(2)
    mn_orig[0] = np.min(X11[:,0])
    mn_orig[1] = np.min(X11[:,1])

    mn_orig=mn_orig-dxMatch



    H12mn = H12.copy()
    H12mn[0:2,2]=H12mn[0:2,2]-mn_orig
    X1=X11-mn_orig


    mn[0] = np.min(X1[:,0])
    mn[1] = np.min(X1[:,1])
    mx[0] = np.max(X1[:,0])
    mx[1] = np.max(X1[:,1])

    t0 = H12mn[0:2,2]
    L0=t0-Lmax
    L1=t0+Lmax



    P = mx-mn

    mxlvl=0
    dx0=mx+dxMatch
    dxs = []
    XYedges=[]
    for i in range(0,100):
        f=2**i

        xedges=np.linspace(0,mx[0]+1*dxMatch[0],f+1)
        yedges=np.linspace(0,mx[1]+1*dxMatch[0],f+1)
        XYedges.append((xedges,yedges))
        dx=np.array([xedges[1]-xedges[0],yedges[1]-yedges[0]])

        dxs.append(dx)

        if np.any(dx<=dxMatch):
            break

    mxlvl=len(dxs)


    dxs=[dx.astype(np.float64) for dx in dxs]

    H1match=nbpt2Dproc.numba_histogram2D(X1, XYedges[-1][0],XYedges[-1][1])
    H1match = np.sign(H1match)

    levels=[]
    HLevels=[H1match]

    for i in range(1,mxlvl):

        Hup = HLevels[i-1]
        H=nbpt2Dproc.UpsampleMax(Hup,n)
        HLevels.append(H)

    mxLVL=len(HLevels)-1
    HLevels=HLevels[::-1]
    HLevels=[np.ascontiguousarray(H).astype(np.int32) for H in HLevels]


    SolBoxes_init=[]
    for xs in np.arange(0,dxs[0][0],dxs[0][0]):
        for ys in np.arange(0,dxs[0][1],dxs[0][1]):
            SolBoxes_init.append( (xs,ys,dxs[0][0],dxs[0][1]) )

    h=[(100000.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)]
    lvl=0
    dx=dxs[lvl]
    H=HLevels[lvl]

    Xth= Dict.empty(
        key_type=types.float64,
        value_type=float_2Darray,
    )
    ii=0

    thfineRes = thmin
    thL=np.arange(-thmax,thmax+thfineRes,thfineRes,dtype=np.float64)

    for j in range(len(thL)):
        th=thL[j]
        R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        XX=np.transpose(R.dot(X22.T))
        XX=np.transpose(H12mn[0:2,0:2].dot(XX.T))#+H12mn[0:2,2]
        Xth[th]=XX


        for solbox in SolBoxes_init:
            xs,ys,d0,d1 = solbox
            Tj=np.array((d0,d1))
            Oj = np.array((xs,ys))

            cost2=nbpt2Dproc.getPointCost(H,dx,Xth[th],Oj,Tj)
            h.append((-cost2-np.random.rand()/1e10,xs,ys,d0,d1,lvl,th,mxLVL))

    heapq.heapify(h)

    XX0=np.transpose(H12mn[0:2,0:2].dot(X22.T))+H12mn[0:2,2]
    zz=np.zeros(2,dtype=np.float64)
    cost0=nbpt2Dproc.getPointCost(HLevels[-1],dxs[-1],XX0,zz,dxs[-1])

    bb1={'x1':t0[0]-Lmax[0],'y1':t0[1]-Lmax[1],'x2':t0[0]+Lmax[0],'y2':t0[1]+Lmax[1]}
    if dxBase[0]>=0:
        while(1):
            HH=[]
            flg=False
            for i in range(len(h)):

                if h[i][3]>dxBase[0] or h[i][4]>dxBase[1]:
                    flg=True
                    (cost,xs,ys,d0,d1,lvl,th,mxLVL)=h[i]

                    nlvl = int(lvl)+1
                    dx=dxs[nlvl]
                    H=HLevels[nlvl]
                    Tj=np.array((d0,d1))
                    Oj = np.array((xs,ys))



                    Xg=np.arange(Oj[0],Oj[0]+Tj[0],dx[0])
                    Yg=np.arange(Oj[1],Oj[1]+Tj[1],dx[1])

                    d0,d1=dx[0],dx[1]
                    Tj=np.array((d0,d1))

                    for xs in Xg[:2]:
                        for ys in Yg[:2]:
                            bb2={'x1':xs,'y1':ys,'x2':xs+d0,'y2':ys+d1}
                            x_left = max(bb1['x1'], bb2['x1'])
                            y_top = max(bb1['y1'], bb2['y1'])
                            x_right = min(bb1['x2'], bb2['x2'])
                            y_bottom = min(bb1['y2'], bb2['y2'])

                            if x_right < x_left or y_bottom < y_top:
                                continue

                            Oj = np.array((xs,ys))
                            cost3=nbpt2Dproc.getPointCost(H,dx,Xth[th],Oj,Tj)
                            HH.append((-cost3-np.random.rand()/1e10,xs,ys,d0,d1,float(nlvl),th,mxLVL))
                else:
                    HH.append(h[i])
            h=HH
            if flg==False:
                break

        heapq.heapify(h)

    while(1):
        (cost,xs,ys,d0,d1,lvl,th,mxLVL)=heapq.heappop(h)
        mainSolbox = (cost,xs,ys,d0,d1,lvl,th,mxLVL)
        if lvl==mxLVL:
            break
            cnt=0
            for jj in range(len(h)):
                if np.floor(-h[jj][0])==np.floor(-cost) and h[jj][5]<lvl:
                    cnt+=1
            if cnt==0:
                break
            else:
                continue

        nlvl = int(lvl)+1
        dx=dxs[nlvl]
        H=HLevels[nlvl]
        Tj=np.array((d0,d1))
        Oj = np.array((xs,ys))



        Xg=np.arange(Oj[0],Oj[0]+Tj[0],dx[0])
        Yg=np.arange(Oj[1],Oj[1]+Tj[1],dx[1])

        d0,d1=dx[0],dx[1]
        Tj=np.array((d0,d1))

        for xs in Xg[:2]:
            for ys in Yg[:2]:
                bb2={'x1':xs,'y1':ys,'x2':xs+d0,'y2':ys+d1}
                x_left = max(bb1['x1'], bb2['x1'])
                y_top = max(bb1['y1'], bb2['y1'])
                x_right = min(bb1['x2'], bb2['x2'])
                y_bottom = min(bb1['y2'], bb2['y2'])

                if x_right < x_left or y_bottom < y_top:
                    continue

                Oj = np.array((xs,ys))
                cost3=nbpt2Dproc.getPointCost(H,dx,Xth[th],Oj,Tj)
                heapq.heappush(h,(-cost3-np.random.rand()/1e10,xs,ys,d0,d1,float(nlvl),th,mxLVL))


    t=np.array(mainSolbox[1:3])-t0
    th = mainSolbox[6]
    cost=np.floor(-mainSolbox[0])
    d0,d1=mainSolbox[3:5]

    HcompR=np.identity(3)
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    HcompR[0:2,0:2]=R

    Ht=np.identity(3)
    Ht[0:2,2]=t

    H12comp=np.dot(Ht.dot(H12mn),HcompR)
    H12comp[0:2,2]=H12comp[0:2,2]+mn_orig

    H21comp=nplinalg.inv(H12comp)
    hh=0
    hR=0
  }

  std::vector<Eigen::MaxtrixXf> Hlevels;
  std::vector<float> dxlevels;
  json options;
}
