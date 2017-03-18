import os
import os.path
import sys
import logging
import numpy as np
import scipy as sp
from copy import deepcopy
from math import ceil
from scipy.stats import multivariate_normal
from numpy.linalg import LinAlgError
from gps.algorithm.config import ALG
from gps.utility.general_utils import extract_condition

LOGGER = logging.getLogger(__name__)


class MpcTrajOpt(object):
    def __init__(self, hyperparams, cond):
        config = deepcopy(ALG)
        config.update(hyperparams)
        self._hyperparams = config
        
        agent = self._hyperparams['agent']
        self.T = self._hyperparams['T'] = agent.T
        self.M = self._hyperparams['M'] = config['init_mpc']['T']
        self.dU = self._hyperparams['dU'] = agent.dU
        self.dX = self._hyperparams['dX'] = agent.dX
        del self._hyperparams['agent']  # Don't want to pickle this
        
        self.N = int(ceil(self.T/(self.M-1.)))
        
        # It will update wwith different X_t in update function
        # Note that is different from M in last n-th MPC
        self.T_mpc = self.M
        
        # Setup policy
        init_mpc = config['init_mpc']
        init_mpc['x0'] = agent.x0
        init_mpc['dX'] = agent.dX
        init_mpc['dU'] = agent.dU
        
        self.mpc_pol = []
        for n in range(self.N):
            init_mpc = extract_condition(
                config['init_mpc'], cond
            )
            self.mpc_pol.append(init_mpc['type'](init_mpc))
        
    def update(self, n, X_t, prior, traj_distr, traj_info, cur_t):
        self.T = traj_distr.T
        dX = traj_distr.dX
        dU = traj_distr.dU
        
        # Make a copy
        trajinfo = deepcopy(traj_info)
        trajinfo.x0mu = X_t
        trajinfo.x0sigma = 1e-6*np.eye(dX)
        
        if cur_t+self.M > self.T:
            X_ref = prior[cur_t:,:dX]
        else:
            X_ref = prior[cur_t:cur_t+self.M,:dX]
        
        # Reset T_mpc
        self.T_mpc = X_ref.shape[0]    
            
        mu, sigma = self.forward(traj_distr, trajinfo, cur_t)
        new_mpc = self.backward(self.mpc_pol[n], traj_distr, trajinfo, X_ref, mu, sigma, cur_t)
        
        # Store mpc
        self.mpc_pol[n] = new_mpc
        
        return new_mpc
        
    def forward(self, traj_distr, traj_info, cur_t):
        """
        Perform LQR forward pass. Computes state-action marginals from
        dynamics and policy.
        Args:
            traj_distr: A linear Gaussian policy object.
            traj_info: A TrajectoryInfo object.
        Returns:
            mu: A T x dX mean action vector.
            sigma: A T x dX x dX covariance matrix.
        """
        # Compute state-action marginals from specified conditional
        # parameters and current traj_info.
        T = self.M
        dU = traj_distr.dU
        dX = traj_distr.dX
    
        # Constants.
        idx_x = slice(dX)
    
        # Allocate space.
        sigma = np.zeros((T, dX+dU, dX+dU))
        mu = np.zeros((T, dX+dU))
    
        # Pull out dynamics.
        Fm = traj_info.dynamics.Fm
        fv = traj_info.dynamics.fv
        dyn_covar = traj_info.dynamics.dyn_covar
    
        # Set initial covariance (initial mu is always zero).
        sigma[0, idx_x, idx_x] = traj_info.x0sigma
        mu[0, idx_x] = traj_info.x0mu
    
        for t in range(T):
            t_traj = cur_t+t
            if t_traj > self.T-1:
                break
            sigma[t, :, :] = np.vstack([
                np.hstack([
                    sigma[t, idx_x, idx_x],
                    sigma[t, idx_x, idx_x].dot(traj_distr.K[t_traj, :, :].T)
                ]),
                np.hstack([
                    traj_distr.K[t_traj, :, :].dot(sigma[t, idx_x, idx_x]),
                    traj_distr.K[t_traj, :, :].dot(sigma[t, idx_x, idx_x]).dot(
                        traj_distr.K[t_traj, :, :].T
                    ) + traj_distr.pol_covar[t_traj, :, :]
                ])
            ])
            mu[t, :] = np.hstack([
                mu[t, idx_x],
                traj_distr.K[t_traj, :, :].dot(mu[t, idx_x]) + traj_distr.k[t_traj, :]
            ])
            if t < T - 1:
                sigma[t+1, idx_x, idx_x] = \
                        Fm[t_traj, :, :].dot(sigma[t, :, :]).dot(Fm[t_traj, :, :].T) + \
                        dyn_covar[t_traj, :, :]
                mu[t+1, idx_x] = Fm[t_traj, :, :].dot(mu[t, :]) + fv[t_traj, :]
        return mu, sigma
    
    def backward(self, prev_mpc_traj_distr, traj_distr, traj_info, x0, mu, sigma, cur_t):
        """
        Perform LQR backward pass. This computes a new linear Gaussian
        policy object.
        Args:
            prev_mpc_traj_distr: previous MPC 
            traj_distr: A linear Gaussian policy object from
                previous iteration.
            traj_info: A TrajectoryInfo object.
            x0: State independent from action generate by forward
                pass from offline trajectory distribution.
            mu, sigma: Parameter of forward pass independent from 
                action, start from current real state.
            cur_t: current time of agent
        Returns:
            traj_distr: A new MPC linear Gaussian policy.
        """
        # Constants.
        T = self.T_mpc
        dU = prev_mpc_traj_distr.dU
        dX = prev_mpc_traj_distr.dX
        
        mpc_traj_distr = prev_mpc_traj_distr.nans_like(zeros=True)
        
        """
        # TODO: Check BADMM need this?
        # Store pol_wt if necessary
        if type(algorithm) == AlgorithmBADMM:
            pol_wt = algorithm.cur[m].pol_info.pol_wt
        """

        idx_x = slice(dX)
        idx_u = slice(dX, dX+dU)

        # Pull out dynamics.
        Fm = traj_info.dynamics.Fm
        fv = traj_info.dynamics.fv

        # Non-SPD correction terms.
        eta = 0
        del_ = 1e-4
        eta0 = eta

        # Run dynamic programming.
        fail = True
        while fail:
            fail = False  # Flip to true on non-symmetric PD.

            # Allocate.
            Vxx = np.zeros((T, dX, dX))
            Vx = np.zeros((T, dX))

            fCm, fcv = self.compute_costs(traj_distr, x0, mu, sigma, cur_t)

            # Compute state-action-state function at each time step.
            for t in range(T - 1, -1, -1):
                t_traj = cur_t+t
                # Add in the cost.
                Qtt = fCm[t, :, :]  # (X+U) x (X+U)
                Qt = fcv[t, :]  # (X+U) x 1

                # Add in the value function from the next time step.
                if t < T - 1:
                    """
                    # TODO: Check BADMM need this?
                    if type(algorithm) == AlgorithmBADMM:
                        multiplier = (pol_wt[t+1] + eta)/(pol_wt[t] + eta)
                    else:
                        multiplier = 1.0
                    """
                    multiplier = 1.0
                    Qtt = Qtt + multiplier * \
                            Fm[t_traj, :, :].T.dot(Vxx[t+1, :, :]).dot(Fm[t_traj, :, :])
                    Qt = Qt + multiplier * \
                            Fm[t_traj, :, :].T.dot(Vx[t+1, :] +
                                            Vxx[t+1, :, :].dot(fv[t_traj, :]))

                # Symmetrize quadratic component.
                Qtt = 0.5 * (Qtt + Qtt.T)
                
                # Regularization to make sure Quu is PD
                Qtt[idx_u, idx_u] += eta*np.eye(dU)
                
                # Compute Cholesky decomposition of Q function action
                # component.
                try:
                    U = sp.linalg.cholesky(Qtt[idx_u, idx_u])
                    L = U.T
                except LinAlgError as e:
                    # Error thrown when Qtt[idx_u, idx_u] is not
                    # symmetric positive definite.
                    #LOGGER.debug('MPC LinAlgError: %s', e)
                    fail = True
                    break

                # Store conditional covariance, inverse, and Cholesky.
                mpc_traj_distr.inv_pol_covar[t, :, :] = Qtt[idx_u, idx_u]
                mpc_traj_distr.pol_covar[t, :, :] = sp.linalg.solve_triangular(
                    U, sp.linalg.solve_triangular(L, np.eye(dU), lower=True)
                )
                mpc_traj_distr.chol_pol_covar[t, :, :] = sp.linalg.cholesky(
                    mpc_traj_distr.pol_covar[t, :, :]
                )

                # Compute mean terms.
                mpc_traj_distr.k[t, :] = -sp.linalg.solve_triangular(
                    U, sp.linalg.solve_triangular(L, Qt[idx_u], lower=True)
                )
                mpc_traj_distr.K[t, :, :] = -sp.linalg.solve_triangular(
                    U, sp.linalg.solve_triangular(L, Qtt[idx_u, idx_x],
                                                  lower=True)
                )

                # Compute value function.
                Vxx[t, :, :] = Qtt[idx_x, idx_x] + \
                        Qtt[idx_x, idx_u].dot(mpc_traj_distr.K[t, :, :])
                Vx[t, :] = Qt[idx_x] + Qtt[idx_x, idx_u].dot(mpc_traj_distr.k[t, :])
                Vxx[t, :, :] = 0.5 * (Vxx[t, :, :] + Vxx[t, :, :].T)

            # Increment eta on non-SPD Q-function.
            if fail:
                old_eta = eta
                eta = eta0 + del_
                LOGGER.debug('Increasing eta: %f -> %f', old_eta, eta)
                del_ *= 2  # Increase del_ exponentially on failure.
                if eta >= 1e16:
                    if np.any(np.isnan(Fm)) or np.any(np.isnan(fv)):
                        raise ValueError('NaNs encountered in dynamics!')
                    raise ValueError('Failed to find PD solution even for very \
                            large eta (check that dynamics and cost are \
                            reasonably well conditioned)!')
        return mpc_traj_distr
    
    def _eval_cost(self, x0, mu, sigma):
        T = self.T_mpc
        dX = self.dX
        dU = self.dU
        
        fCm = np.zeros([T, dX+dU, dX+dU])
        fcv = np.zeros([T, dX+dU])
        
        idx_x = slice(dX)
        idx_u = slice(dX, dX+dU)
        
        # Cost = -logp(x_t'|x_t)
        # Gradient = inv(Sigma)*(x0 - mu)
        # Hessian = inv(Sigma)
        for t in range(T - 1, -1, -1):
            inv_sigma = np.linalg.inv(sigma[t,idx_x,idx_x])
            fcv[t, idx_x] = inv_sigma.dot(x0[t] - mu[t,idx_x]) # Gradient
            fCm[t, idx_x, idx_x] = inv_sigma # Hessian
        
        #yhat = np.c_[x0, u0]
        yhat = np.c_[x0]
        rdiff = -yhat
        rdiff_expand = np.expand_dims(rdiff, axis=2)
        cv_update = np.sum(fCm[:,idx_x,idx_x] * rdiff_expand, axis=1)
        fcv[:,idx_x] += cv_update
                
        return fCm, fcv
    
    def compute_costs(self, traj_distr, x0, mu, sigma, cur_t):
        T = self.T_mpc
        fCm, fcv = self._eval_cost(x0, mu, sigma)
        
        K, ipc, k = traj_distr.K, traj_distr.inv_pol_covar, traj_distr.k
        # Add in the trajectory divergence term.
        for t in range(T - 1, -1, -1):
            t_traj = cur_t+t
            fCm[t, :, :] += np.vstack([
                np.hstack([
                    K[t_traj, :, :].T.dot(ipc[t_traj, :, :]).dot(K[t_traj, :, :]),
                    -K[t_traj, :, :].T.dot(ipc[t_traj, :, :])
                ]),
                np.hstack([
                    -ipc[t_traj, :, :].dot(K[t_traj, :, :]), ipc[t_traj, :, :]
                ])
            ])
            fcv[t, :] += np.hstack([
                K[t_traj, :, :].T.dot(ipc[t_traj, :, :]).dot(k[t_traj, :]),
                -ipc[t_traj, :, :].dot(k[t_traj, :])
            ])
                            
        return fCm, fcv