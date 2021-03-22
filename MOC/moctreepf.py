from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import numpy.linalg as nplg
import scipy.linalg as sclg
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from uq.stats import pdfs as uqstpdf
from scipy.stats import multivariate_normal

from uq.stats import moments as uqstmom
# clf.predict([[2., 2.]])
    # X = [[0, 0], [1, 1]]
    # Y = [0, 1]
plt.close('all')
def fit_tree(X,Y,lb,ub,max_depth=10,min_samples_split=6,min_samples_leaf=6):

    clf = tree.DecisionTreeRegressor(max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)
    clf = clf.fit(X, Y)
    boxes = get_boxes(clf,lb,ub)
    
    return clf,boxes

def get_boxes(estimator, lb,ub):
    lb=lb.astype(float)
    ub=ub.astype(float)
    v=np.prod(ub-lb)
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold
    extracted_MSEs = estimator.tree_.impurity
    values = estimator.tree_.value
    
    boxes=[]
    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [ [0, -1,[lb.copy(),ub.copy(),extracted_MSEs[0],values[0]]] ]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth, box = stack.pop()
        node_depth[node_id] = parent_depth + 1
        
        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            lbnl = box[0].copy()
            ubnl = box[1].copy()
            lbnr = box[0].copy()
            ubnr = box[1].copy()
            ubnl[feature[node_id]]=threshold[node_id]
            lbnr[feature[node_id]]=threshold[node_id]
            lmse = extracted_MSEs[children_left[node_id]]
            rmse = extracted_MSEs[children_right[node_id]]
            lval = values[children_left[node_id]]
            rval = values[children_right[node_id]]
            stack.append([children_left[node_id], parent_depth + 1,[lbnl,ubnl,lmse,lval]])
            stack.append([children_right[node_id], parent_depth + 1,[lbnr,ubnr,rmse,rval]])
        else:
            is_leaves[node_id] = True
            box[2]=extracted_MSEs[node_id]
            box[3]=values[node_id]
            boxes.append(box)
    nboxes=[]
    for i in range(len(boxes)):
        nboxes.append({'lb':boxes[i][0],'ub':boxes[i][1],'mse':boxes[i][2],'m':boxes[i][3]} )    


        
    # print("The binary tree structure has %s nodes and has "
    #       "the following tree structure:"% n_nodes)
    # nlfnodes=0
    # for i in range(n_nodes):
    #     if is_leaves[i]:
    #         nlfnodes +=1
    #         print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
    #     else:
    #         print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
    #               "node %s."
    #               % (node_depth[i] * "\t",
    #                  i,
    #                  children_left[i],
    #                  feature[i],
    #                  threshold[i],
    #                  children_right[i],
    #                  ))
    
    # vb=0
    # for box in boxes:
    #     vb+=np.prod(box[1]-box[0])
    # print(v,vb)
    # print("nlfnodes = ",nlfnodes)        
    return nboxes

def getmeshpoints(N,box=None,lb=None,ub=None):
    if box is None:
        box={'lb':lb,'ub':ub}
    w=box['ub']-box['lb']
    d=5*w/100;
    S=[]
    for i in range(len(box['lb'])):
        S.append(np.linspace(box['lb'][i]+d[i],box['ub'][i]-d[i],N))
    X=np.meshgrid(*S)
    X=np.vstack([x.reshape(-1) for x in X]).T
    return X

def pointsIdxInBox(X,box):
    lb=box['lb']
    ub=box['ub']
    
    return np.all((X>lb)&(X<ub),axis=1)

def plot_boxes(boxes,stateslist,c='r',fill =False):
    # stateslist = [[0,1],[1,2]] etc
    figureslist=[]
    for states in stateslist:
        fig=plt.figure()
        figureslist.append(fig)
        ax=fig.add_subplot(111)
        for box in boxes:
            lb=box['lb'][states]
            ub=box['ub'][states]
            rect = patches.Rectangle(lb,ub[0]-lb[0],ub[1]-lb[1],linewidth=1,edgecolor='r',fill =fill ,alpha=0.4)
            ax.add_patch(rect)
    return figureslist


def randomSampleBox(N,box):
    dim = len(box['lb'])
    w = box['ub'] - box['lb']
    X = box['lb']+w*np.random.rand(N,dim)
    return X

class BoxPdf:
    def __init__(self,boxes):
        self.boxes = boxes
        self.XboxesLB = np.array([box['lb'] for box in self.boxes])
        self.XboxesUB = np.array([box['ub'] for box in self.boxes])
        self.XboxesV = np.array([box['m'] for box in self.boxes])
        self.II = np.arange(0,len(self.boxes)).astype(int)
    
    def pdf(self,X):
        f = np.zeros(X.shape[0])
        for i in range(len(self.boxes)):
            idx = pointsIdxInBox(X,self.boxes[i])
            f[self.II[idx]] = self.boxes[i]['m']
        
        return f
    
    
def gaussianPDF(X,m,P):
    Pinv = nplalg.inv(P)
    a = X-m
    c = 1/np.sqrt(nplalg.det(2*np.pi*P))
    if X.ndim==1:
        y = c*np.exp(-0.5*multi_dot([a,Pinv,a]))
    else:
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y[i] = np.exp(-0.5*multi_dot([a[i],Pinv,a[i]]))
        y = c*y
    return y

def gaussianPDF1D(X,m,var):
    Pinv = 1/var
    a = X-m
    c = 1/np.sqrt(2*np.pi*var)
    y = c*np.exp(-0.5*Pinv*a**2)
    return y

def hfunc(x):
    p0=np.array([60,0])
    if x.ndim>1:
        r=nplg.norm( x-p0,axis=1)
    else:
        r= nplg.norm(x-p0)
    return r

def problike(z,x):
    # single meas z and multiple x
    var = 5**2
    return gaussianPDF1D(z-hfunc(x),0,var)   


# Algorithms - using MC
"""
- use initial MC-pf to estimate mean and cov
- prior:
    - build tree with depth d=3 centered at mu, 6sig
    - iterate boxes with var>thresh_var
        - get more points for this box



"""

zstar = 70

def Algo1PFprior(pdf0,X0,p0,Fpforw,Fpback,mseIterRel = 10,max_depth = 3,
                 min_samples_split = 6, min_samples_leaf = 6,Nmcbox = 100):
    # upgraded pf
    # pdf0 is the PDF that can be evaluated
    # X0 are some initial points from pdf0
    # return pdf1 
    # Fpforw: propagated forward both the point x and its probability
    # Fpback: propagated backward both the point x and its probability
    
    # mseIterRel = 10 # 10% relative error between parent box and max of child boxes  
    # max_depth = 3
    # min_samples_split = 6 
    # min_samples_leaf = 6
    # Nmcbox = 100
    

    X1,p1 = Fpforw(X0,p0)
    p1 = p1*problike(zstar,X1)
    m1,P1=uqstmom.MeanCov(X1, p1/np.sum(p1))
    sig = np.diag(sclg.sqrtm(P1))
    boxes=[{'lb':m1-6*sig,'ub':m1+6*sig,'m':np.mean(p1),'mse':np.std(p1),'status':'splitit'}] # lb, ub, mean, var, parsed?
    # bxmse= np.max(np.abs((np.log(p1bo)-np.log(box['m']))/np.log(p1bo) ))
    cntiter = 0
    while True:
        nb = len(boxes)
        newboxes=[]
        for i in range(nb):
            box = boxes[i]
            if box['status'] == 'stop':
                newboxes.append(box)
                continue
            
            bxmse = box['mse']
            idx = pointsIdxInBox(X1,box)
            Ninbox = np.sum(idx)

            if Ninbox<Nmcbox:
                X1b = randomSampleBox(Nmcbox-Ninbox,box)
            
                X0b,_ = Fpback(X1b,None)
                p0b = pdf0.pdf(X0b)
                _,p1b = Fpforw(X0b,p0b)                
                
                p1b = p1b*problike(zstar,X1b)
                
                X1bo = np.vstack([X1b,X1[idx,:]])
                p1bo = np.hstack([p1b,p1[idx]])
                
                X1 = np.vstack([X1,X1b])
                p1 = np.hstack([p1,p1b])                
                
            else:
                X1bo = X1[idx,:]
                p1bo = p1[idx]
            
            idx=pointsIdxInBox(X1bo,box)

            clf = tree.DecisionTreeRegressor(max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)
            clf = clf.fit(X1bo, p1bo)
            subboxes = get_boxes(clf,box['lb'],box['ub'])
            subbxmse = np.max([b['mse'] for b in subboxes])
            for j in range(len(subboxes)):
                subboxes[j]['status']='splitit'
            
            bxmse= np.mean(np.abs((np.log(p1bo)-np.log(box['m']))/np.log(box['m']) ))
            subboxesM=clf.predict(X1bo)
            subbxmse = np.mean(np.abs((np.log(p1bo)-np.log(subboxesM))/np.log(subboxesM)))
            print(len(p1bo),np.mean(p1bo),box['m'],bxmse,subbxmse,100*np.abs(subbxmse-bxmse)/bxmse)
            # newboxes.extend(subboxes)
            if 100*np.abs(subbxmse-bxmse)/bxmse >0.01: # if greater than relerror
                newboxes.extend(subboxes)
            else:
                box['status'] = 'stop'
                newboxes.append(box)
        
        cntiter += 1
        boxes = newboxes
        if np.all([box['status']=='stop' for box in boxes]):
            break
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X1[:,0],X1[:,1],p1,alpha=0.3)
        verts=[]
        for box in boxes:
            verts.append(lbubbox2cuboid2D(box))
        ax.add_collection3d(Poly3DCollection(verts,alpha=0.4,color='r'))
        
        plt.show()
        cntiter
        if cntiter > 25:
            break
        
    print("box-splitter took ", cntiter, " iterations and a total of ",len(p1), " points and ",len(boxes), " boxes")
    pdf1 = BoxPdf(boxes)
    return pdf1,X1,p1,boxes
    
def lbubbox2cuboid2D(box):
    lb=box['lb']
    ub=box['ub']
    Y=[]
    Y.append([lb[0],lb[1],box['m'] ])
    Y.append([ub[0],lb[1],box['m'] ])
    Y.append([ub[0],ub[1],box['m'] ])
    Y.append([lb[0],ub[1],box['m'] ])
    
    return Y



#%%


    
def func2(x):
    if x.ndim==1:
        r=x[0]
        th=x[1]
        xk1 = r*np.array([np.cos(th),np.sin(th)])
        return xk1
    else:
        xk1 = np.vstack([x[:,0]*np.cos(x[:,1]),x[:,0]*np.sin(x[:,1])]).T
        return xk1
    
def func2jac(x):
    if x.ndim==1:
        r=x[0]
        th=x[1]
        jac = np.array([[np.cos(th),-r*np.sin(th)],[np.sin(th),r*np.cos(th)]])
        return jac
    else:
        jac=[]
        for i in range(x.shape[0]):
            r=x[i,0]
            th=x[i,1]
            jac.append( np.array([[np.cos(th),-r*np.sin(th)],[np.sin(th),r*np.cos(th)]]) )

        return jac


def func2back(x):
    if x.ndim==1:
        x1=x[0]
        y1=x[1]
        r=nplg.norm(x)
        th = np.arctan2(y1,x1)
        xk1 = np.array([r,th])
        return xk1
    else:
        r=nplg.norm(x,axis=1)
        th = np.arctan2(x[:,1],x[:,0])
        xk1 = np.vstack([r,th]).T
        return xk1
    
def func2jacback(x):
    if x.ndim==1:
        x1=x[0]
        y1=x[1]
        r=nplg.norm(x)
        jac = np.array([[x1/r,y1/r],[-y1/r**2,x1/r**2]])
        return jac
    else:
        jac=[]
        for i in range(x.shape[0]):
            x1=x[i,0]
            y1=x[i,1]
            r=nplg.norm(x[i,:])
            jac.append( np.array([[x1/r,y1/r],[-y1/r**2,x1/r**2]]) )

        return jac


def p2cFpforw(X,p):
    Y=None
    if X is not None:
        Y=func2(X)
    P=None
    if p is not None:
        jacprop = func2jac(X)
        P = p/np.array([nplg.det(jacprop[i]) for i in range(X.shape[0])])
    return Y,P

def p2cFpback(X,p):
    Y=None
    if X is not None:
        Y=func2back(X)
    P=None
    if p is not None:
        jacprop = func2jacback(X)
        P = p/np.array([nplg.det(jacprop[i]) for i in range(X.shape[0])])
    return Y,P



x0eg2 = np.array([30,np.pi/2])
P0eg2 = np.array([[2**2,0],[0,(30*np.pi/180)**2]])
pdf0 = multivariate_normal(x0eg2,P0eg2)    #lambda X:uqstpdf.gaussianPDF(X,x0eg2,P0eg2)
X0 = np.random.multivariate_normal(x0eg2,P0eg2,100)
p0 = multivariate_normal(x0eg2,P0eg2).pdf(X0)   #uqstpdf.gaussianPDF(X0,x0eg2,P0eg2)

X1tr = func2(X0)
jacprop = func2jac(X0)
p1tr = np.array( [p0[i]/nplg.det(jacprop[i]) for i in range(X0.shape[0]) ] )

pdf1,X1,p1,boxes=Algo1PFprior(pdf0,X0,p0,p2cFpforw,p2cFpback,mseIterRel = 20,max_depth = 1,
                 min_samples_split = 10, min_samples_leaf = 3,Nmcbox = 50)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X1[1:100,0],X1[1:100,1],p1[1:100])
# verts=[]
# for box in boxes:
#     verts.append(lbubbox2cuboid2D(box))
# ax.add_collection3d(Poly3DCollection(verts,alpha=0.4))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X1[:,0],X1[:,1],p1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1tr[:,0],X1tr[:,1],p1tr)
#%%
# lb=np.array([-5,-5])
# ub=np.array([5,5])
# X = getmeshpoints(35,box=None,lb=lb,ub=ub)
# X=X+0.1*np.random.randn(*(X.shape))
# Y=uqstpdf.gaussianPDF(X,np.zeros(2),np.identity(2))

# clf = tree.DecisionTreeRegressor(max_depth=7,min_samples_split=3,min_samples_leaf=3)
# clf = clf.fit(X, Y)
# boxes=get_boxes(clf,lb,ub)

# stateslist=[[0,1]]
# figureslist = plot_boxes(boxes,stateslist,c='r',fill =False)
# figureslist[0].axes[0].plot(X[:,0],X[:,1],'.')
# # figureslist[1].axes[0].plot(X[:,1],X[:,2],'.')
# plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:,0],X[:,1],Y)
# verts=[]
# for box in boxes:
#     verts.append(lbubbox2cuboid2D(box))
# ax.add_collection3d(Poly3DCollection(verts))
    


