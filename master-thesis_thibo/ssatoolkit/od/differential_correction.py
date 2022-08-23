import numpy as np
import scipy.optimize
# import toolkit as tl
# from prop_odeint import propagate
# from laplace_method import laplace_orbit_fit
import ssatoolkit.propagators as ssaprop
from ssatoolkit import coords

def zkDistErrs(rv0,t0, data,mu):
    """ Compile the differentiate.

        Parameters
        ----------

        rv0: numpy.array (6x1)
          Initial state of the orbit after IOD.
        pos_obs: numpy.array (3xn)
          an array of observed position.
        L_obs: numpy.array (3xm)
          an array of angle of observation.


        Returns
        -------

        sum: TBD
          A single element that reprensent the sum to be minimized.

        """

    # sum = 0
    tvec = data[:,6]
    orb = coords.from_RV_to_OE(rv0[0:3],rv0[3:],mu)
    # rv0 = coords.from_OE_to_RV(orb0,mu)
    if orb[1]<0 or orb[1]>1:
        _,X=ssaprop.propagate_rv_tvec(t0, rv0, tvec, mu)
    else:
        _,X=ssaprop.propagate_FnG_mixed(t0,rv0,tvec, mu, tol = 1e-12)
    
    L=X[:,0:3]-data[:,3:6]  
    L=L/np.linalg.norm(L,axis=1).reshape(-1,1)
    
    # rmserr = np.sqrt(np.mean(np.sum(np.power(L - data[:,0:3],2),axis=1)))
    rmserr = np.sqrt(np.sum(np.power(L - data[:,0:3],2),axis=1))
    
    return rmserr    
    # for row in pos_obs:
    #     pos_pred = propagate(step*i, rv0[0:3], rv0[3:6], step=step)[-1]
    #     #pos_pred = tl.propagate_kep(rv0[0:3], rv0[3:6], step*i)[-1]
    #     sum = sum + (np.linalg.norm(row - pos_pred[0:3]))**2
    #     i=i+1
    '''
    for row in L_obs[6:, :]:
        pos_pred = propagate(step*i, rv0[0:3], rv0[3:6], step=step)[-1]

        #r_site = tl.LongLatHeigth_to_cart(observatory_pos[int(obs_station[i]), :])
        r_site = observatory_pos
        angle = pos_pred[0:3] - r_site
        L_pred = angle / np.linalg.norm(angle) # Big Warning here, still some work to do
        sum = sum + (np.linalg.norm(row - L_pred))**2
        i=i+1
    '''
    # return sum
    
def M2data(M,sensors,Tvec):
    """
     M=[(s1,'a',L,k1),(s2,'ra',r,k2),..]

    """
    data = np.zeros((len(M),7))
    for i in range(len(M)):
        sensId = M[i].sensID
        data[i,0:3] = M[i].zk
        data[i,3:6] = sensors[sensId].r_site
        data[i,6] = Tvec[M[i].k]
    
    return data

def orbit_det(t0,rv0,data,planet):
    """
     M=[(s1,'a',L,k1),(s2,'ra',r,k2),..]

    """
    # data = M2data(M,sensors,Tvec)
    
    # tvec = np.sort(list(set(data[:,6])))
    # data_dict={t:[] for t in tvec}
    # for i in range(data.shape[0]):
    #     data_dict[data[i,6]].append(data[i])
    
    dct = coords.DimensionalConverter(planet)
    # rvdbr_can = dct.true2can_posvel(rvdbr)  
    # t1_can = dct.true2can_time(T[1])

    
    # data_can=np.zeros_like(data)
    # for i in range(data.shape[0]):
    #     data_can[i,0:3] = data[i,0:3]
    #     data_can[i,3:6] = dct.true2can_pos(data[i,3:6])
    #     data_can[i,6] = dct.true2can_time(data[i,6])
        
        
    # rv0_can = np.zeros_like(rv0)
    # rv0_can[0:3] = dct.true2can_pos( rv0[0:3])
    # rv0_can[3:6] = dct.true2can_pos( rv0[3:6])
    # t0_can = dct.true2can_time(t0)
    # res = scipy.optimize.least_squares(zkDistErrs, rv0_can, args=( t0_can,data_can,1 ),method='lm',ftol=1e-3,xtol=1e-4)
    
    # rv_can = res['x']
    # rv = np.zeros_like(rv_can)
    # rv[0:3] = dct.normX2trueX*rv0[0:3]
    # rv[3:6] = dct.true2can_pos( rv0[3:6])
    
    res = scipy.optimize.least_squares(zkDistErrs, rv0, args=( t0,data,planet.mu ),method='lm',ftol=1e-2,xtol=1e-3)
    # res = scipy.optimize.least_squares(zkDistErrs, orb0, args=( t0,data,mu ),bounds=[(7000,-1,0,0,0,0),(30000,2,2*np.pi,2*np.pi,np.pi,2*np.pi)])
    
    # res = scipy.optimize.minimize(func, rv0, args=( t0,data,mu ), method="BFGS")
    return res


# if __name__=='__main__':
#     obs_data = np.loadtxt('obs_data1.csv', delimiter=',')
#     #omega_e = np.array([0, 0, 7.292115 * 10 ** (-5)]) # omega_earth
#     step = 5
#     #print(obs_data[np.where(obs_data[:, 0] == 154)])
#     data = obs_data[np.where(obs_data[:, 0] == 5)]
#     actual_pos = data[:, 1:4]
#     obs_station = data[:, 4]
#     actual_pos = actual_pos + 0.2 * np.random.randn(np.size(actual_pos, axis = 0), np.size(actual_pos, axis = 1))
#     obs =  np.array([[-23.02331*tl.to_rad(), -67.75379*tl.to_rad(), 6378],   # Attacama Observatory
#                        [49.133949*tl.to_rad(), 1.441930*tl.to_rad(), 6378],        # Pressagny l'Orgeuilleux Observatory
#                        [35.281287*tl.to_rad(), -116.783051*tl.to_rad(), 6378]])
#     r_site = tl.LongLatHeigth_to_cart(obs)
#     L = get_L_from_Pos(actual_pos, obs, obs_station)
#     T = np.array([0, step, 2*step, 3*step, 4*step])
#     rv = laplace_orbit_fit(tl.LongLatHeigth_to_cart(obs[0, :]), T, L[0:4], step=5) # Warning, some work to do here
#     r = rv[0:3]
#     v = rv[3:6]
#     Orbit1 = tl.from_RV_to_OE(r, v)
#     print(Orbit1)
    
#     od = scipy.optimize.minimize(func, rv, args=(data[:, 1:4], L, tl.LongLatHeigth_to_cart(obs[2, :]), obs_station), method="Nelder-Mead")
    
#     new = tl.from_RV_to_OE(od.x[0:3], od.x[3:6])
#     print(new)
