import copy

import numpy as np

from gps.algorithm.cost.config import COST_OBSTACLE
from gps.algorithm.cost.cost import Cost
from gps.algorithm.cost.cost_utils import evalhinglel2loss, get_ramp_multiplier

class CostObstacle(Cost):
    """ Computes hingle l2 loss for a given closest obstacle position. """
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_OBSTACLE)
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
        
        data_type = self._hyperparams['position_type']

        wp = self._hyperparams['wp']
        obs = sample.get(self._hyperparams['obstacle_type'])
        x = sample.get(data_type)

        wpm = get_ramp_multiplier(
            self._hyperparams['ramp_option'], T,
            wp_final_multiplier=self._hyperparams['wp_final_multiplier']
        )
        wp = wp * np.expand_dims(wpm, axis=-1)    
        # Compute state penalty.
        dist = x - obs

        # Evaluate penalty term.
        l, ls, lss = evalhinglel2loss(
            wp, dist, self._hyperparams['d_safe'], self._hyperparams['l2'],
        )

        # Add to current terms.
        sample.agent.pack_data_x(lx, ls, data_types=[data_type])
        sample.agent.pack_data_x(lxx, lss,
                                 data_types=[data_type, data_type])

        return l, lx, lu, lxx, luu, lux
