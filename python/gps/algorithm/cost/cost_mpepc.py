import copy

import numpy as np
import math
from gps.algorithm.cost.config import COST_MPEPC
from gps.algorithm.cost.cost import Cost
from gps.algorithm.cost.cost_utils import evalhinglel2loss, get_ramp_multiplier

class CostMPEPC(Cost):
    """ Computes hingle l2 loss for a given closest obstacle position. """
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_MPEPC)
        config.update(hyperparams)
        Cost.__init__(self, config)

    def eval(self, sample):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample:  A single sample
        """
        T = sample.T
        dX = sample.dX
        dU = sample.dU

        # Initialize terms.
        l = np.zeros(T)
        lu = np.zeros((T, dU))
        lx = np.zeros((T, dX))
        luu = np.zeros((T, dU, dU))
        lxx = np.zeros((T, dX, dX))
        lux = np.zeros((T, dU, dX))

        wp = self._hyperparams['wp']

        obs = sample.get(self._hyperparams['obstacle_type'])
        x = sample.get(self._hyperparams['position_type'])
        pot = sample.get(self._hyperparams['potential_type'])

        wpm = get_ramp_multiplier(
            self._hyperparams['ramp_option'], T,
            wp_final_multiplier=self._hyperparams['wp_final_multiplier']
        )
        wp = wp * np.expand_dims(wpm, axis=-1)

        # Compute state penalty.
        obs_dist = x - obs

        sigma_square = self._hyperparams['sigma_square']
        p_collision = np.exp(- np.sum(obs_dist ** 2, axis=1) / sigma_square)
        l_collision = p_collision * self._hyperparams['wp_col'] * wpm

        pot_prev = np.zeros_like(pot[:, 0])
        pot_prev[0] = 0 # Penalty for first
        pot_prev[1:] = pot[:-1,0]
        p_survivability = 1 - p_collision
        l_progress = p_survivability * (pot[:, 0] - pot_prev) * self._hyperparams['wp_nf'] * wpm

        l = l_collision + l_progress

        # Evaluate penalty term.
        # l, ls, lss = evalhinglel2loss(
        #     wp, dist, self._hyperparams['d_safe'], self._hyperparams['l2'],
        # )
        #
        # # Add to current terms.
        # sample.agent.pack_data_x(lx, ls, data_types=[data_type])
        # sample.agent.pack_data_x(lxx, lss,
        #                          data_types=[data_type, data_type])

        return l, lx, lu, lxx, luu, lux
