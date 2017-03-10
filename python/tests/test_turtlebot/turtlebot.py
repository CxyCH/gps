#import matplotlib as mpl
#mpl.use('Qt4Agg')

import os
import os.path
import sys
import numpy as np
import imp
import Box2D as b2
from copy import deepcopy
# Add gps/python to path so that imports work.
gps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', ''))
sys.path.append(gps_path)

from gps.agent.ros.agent_turtlebot import AgentTurtlebot
from gps.gui.gps_training_gui import GPSTrainingGUI
from gps.utility.data_logger import DataLogger
from gps.sample.sample_list import SampleList
from gps.proto.gps_pb2 import END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, POSITION_NEAREST_OBSTACLE, ACTION
from gps.sample.sample import Sample
from gps.algorithm.cost.cost_obstacles import CostObstacle
from gps.algorithm.cost.cost_state import CostState
from gps.utility.data_logger import DataLogger
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT
from gps.algorithm.cost.cost_utils import evalhinglel2loss, evall1l2term, evallogl2term
from scipy.stats import multivariate_normal
from numpy.linalg import LinAlgError
from gps.algorithm.traj_opt.mpc_traj_opt import MpcTrajOpt
from gps.algorithm.policy.lin_gauss_init import init_pd
from math import ceil
import scipy as sp
import time

config = None

def loadExperiment(exp_name):
	from gps import __file__ as gps_filepath
	gps_filepath = os.path.abspath(gps_filepath)
	gps_dir = '/'.join(str.split(gps_filepath, '/')[:-3]) + '/'
	exp_dir = gps_dir + 'experiments/' + exp_name + '/'
	hyperparams_file = exp_dir + 'hyperparams.py'
	
	print hyperparams_file
	
	if not os.path.exists(hyperparams_file):
	    sys.exit("Experiment '%s' does not exist.\nDid you create '%s'?" %
	             (exp_name, hyperparams_file))
	
	hyperparams = imp.load_source('hyperparams', hyperparams_file)
	
	return hyperparams
    
def runTest(itr_load):
	data_files_dir = config['common']['data_files_dir']
	data_logger = DataLogger()
	
	algorithm_file = data_files_dir + 'algorithm_itr_%02d.pkl' % itr_load
	algorithm = data_logger.unpickle(algorithm_file)
	if algorithm is None:
	    print("Error: cannot find '%s.'" % algorithm_file)
	    os._exit(1) # called instead of sys.exit(), since this is in a thread
	
	pol = algorithm.cur[0].traj_distr
	#pol = algorithm.policy_opt.policy
	agent_hyperparams = deepcopy(AGENT)
	agent_hyperparams.update(config['agent'])
	
	x0s = agent_hyperparams["x0"]
	for cond in range(len(x0s)):
		T = agent_hyperparams['T']
		dX = x0s[cond].shape[0]
		dU = agent_hyperparams['sensor_dims'][ACTION]
		
		agent_hyperparams['render'] = True
		agent = config['agent']['type'](agent_hyperparams)
		
		'''
		while True:
			sample = agent.get_data()		
			raw_input("Get data")
		'''
		# Sample using offline trajectory distribution.
		for i in range(config['num_samples']):
			agent.sample(pol, cond)
	
def main():
	print 'running ros'
	
	exp_name = "turtlebot_example"
	hyperparams = loadExperiment(exp_name)
	global config
	config = hyperparams.config
	
	#runExperiment()
	argv = sys.argv
	runTest(int(argv[1]))
	print "pass test"

if __name__ == '__main__':
    main()
