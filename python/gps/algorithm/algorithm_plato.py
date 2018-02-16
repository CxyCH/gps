""" This file defines the PIGPS algorithm.

Reference:
Y. Chebotar, M. Kalakrishnan, A. Yahya, A. Li, S. Schaal, S. Levine.
Path Integral Guided Policy Search. 2016. https://arxiv.org/abs/1610.00529.
"""
import copy
import logging
import numpy as np

from gps.algorithm.algorithm import Algorithm
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.algorithm_utils import PolicyInfo
from gps.algorithm.config import ALG_PLATO
from gps.sample.sample_list import SampleList

LOGGER = logging.getLogger(__name__)


class AlgorithmPLATO(AlgorithmMDGPS):
    """
    Sample-based joint policy learning and trajectory optimization with
    path integral guided policy search algorithm.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(ALG_PLATO)
        config.update(hyperparams)
        AlgorithmMDGPS.__init__(self, config)

        # TODO: Append sample & get expert info here
        # Compute target mean, cov, and weight for each sample.
        dU, dO, T = self.dU, self.dO, self.T
        self.obs_data, self.tgt_mu = np.zeros((0, T, dO)), np.zeros((0, T, dU))
        self.tgt_prc, self.tgt_wt = np.zeros((0, T, dU, dU)), np.zeros((0, T))

    def iteration(self, sample_lists):
        """
        Run iteration of PI-based guided policy search.

        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        # Store the samples and evaluate the costs.
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]
            self._eval_cost(m)

        # On the first iteration, need to catch policy up to init_traj_distr.
        if self.iteration_count == 0:
            self.new_traj_distr = [
                self.cur[cond].traj_distr for cond in range(self.M)
            ]
            self._update_policy()

        # C-step
        self._update_trajectories()

        # S-step
        self._update_policy()

        # Prepare for next iteration
        self._advance_iteration_variables()

    def _update_policy(self):
        """ Compute the new policy. """
        dU, dO, T = self.dU, self.dO, self.T

        for m in range(self.M):
            samples = self.cur[m].sample_list
            X = samples.get_X()
            N = len(samples)
            traj, pol_info = self.new_traj_distr[m], self.cur[m].pol_info
            mu = np.zeros((N, T, dU))
            prc = np.zeros((N, T, dU, dU))
            wt = np.zeros((N, T))
            for n in range(N):
                mu[n] = samples[n].get(self._hyperparams['expert_type'])
            # Get time-indexed actions.
            for t in range(T):
                # Compute actions along this trajectory.
                prc[:, t, :, :] = np.tile(traj.inv_pol_covar[t, :, :],
                                          [N, 1, 1])
                # for i in range(N):
                #     mu[i, t, :] = (traj.K[t, :, :].dot(X[i, t, :]) + traj.k[t, :])
                wt[:, t].fill(pol_info.pol_wt[t])
            print "Latest expert ", mu[-1], " Sample ", self.tgt_mu.shape
            self.tgt_mu = np.concatenate((self.tgt_mu, mu))
            self.tgt_prc = np.concatenate((self.tgt_prc, prc))
            self.tgt_wt = np.concatenate((self.tgt_wt, wt))
            self.obs_data = np.concatenate((self.obs_data, samples.get_obs()))
        self.policy_opt.update(self.obs_data, self.tgt_mu, self.tgt_prc, self.tgt_wt)
