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

from gps.agent.box2d.agent_box2d import AgentBox2D
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

def runExperiment():
	agent_hyperparams = config['agent']
	agent_hyperparams['render'] = False
	agent = config['agent']['type'](agent_hyperparams)
	x0 = agent_hyperparams["x0"]
	T = agent_hyperparams['T']
	world = agent_hyperparams["world"](x0, agent_hyperparams["target_state"], agent_hyperparams["render"])
	
	cost_obstacle = CostObstacle(config['algorithm']['cost']['costs'][2])
	cost_state = CostState(config['algorithm']['cost']['costs'][1])
	world.run()
	world.reset_world()
	b2d_X = world.get_state()
	sample = Sample(agent)
	set_sample(sample, b2d_X, -1)
	for t in range(T):
		world.run_next([1, 1])
		b2d_X = world.get_state()
		set_sample(sample, b2d_X, t)
		"""
		l, lx, lu, lxx, luu, lux = cost_obstacle.eval(sample)
		sl, slx, slu, slxx, sluu, slux = cost_state.eval(sample)
		
		print sample.get(DISTANCE_TO_NEAREST_OBSTACLE, t), l[t], sl[t]
		print lx[t]
		#print lxx[t]
		"""
	
	obs = sample.get(DISTANCE_TO_NEAREST_OBSTACLE)
	x = sample.get(END_EFFECTOR_POINTS)
	dist = x - obs
	#"""
	f = lambda dist: evalhinglel2loss(
	    0.3*np.ones([T,3]), dist, 1.0, 0, 0
	)[0]
	al, alx, alxx = evalhinglel2loss(
	    0.3*np.ones([T,3]), dist, 1.0, 0, 0
	)
	#"""
	
	"""
	_, dim_sensor = x.shape
	f = lambda dist: evallogl2term(
	    0.5*np.ones([T, 3]), dist, np.tile(np.eye(dim_sensor), [T, 1, 1]),
        np.zeros((1, dim_sensor, dim_sensor, dim_sensor)),
        0.0, 1.0, 1e-2
	)[0]
	al, alx, alxx = evallogl2term(
	    0.5*np.ones([T, 3]), dist, np.tile(np.eye(dim_sensor), [T, 1, 1]),
        np.zeros((T, dim_sensor, dim_sensor, dim_sensor)),
        0.0, 1.0, 1e-2
	)
	"""
	
	nlx = evalGradient(f, dist)
	nlxx = evalHessian(f, dist)
	"""
	f2 = lambda dist: np.sum(0.5 * dist ** 2)
	x0 = dist[T-1]
	print x0
	eps=1e-1
	print f2(x0)
	for i in range(3):
		x_p = np.array(x0)
		x_p[i] = x0[i] + eps
		x_p = x_p.reshape(1,3)
		f_xp = f2(x_p)
		
		x_s = np.array(x0)
		x_s[i] = x0[i] - eps
		x_s = x_s.reshape(1,3)
		f_xs = f2(x_s)
		
		print (f_xp-f_xs) / (2 * eps)
	"""
	for t in range(48):
		#print lx[t,:3],
		print alx[t],
		print nlx[t]
		#print "Hessian"
		print alxx[t]
		print nlxx[t]
	#"""

def set_sample(sample, b2d_X, t):
	for sensor in b2d_X.keys():
		sample.set(sensor, np.array(b2d_X[sensor]), t)
		
def evalGradient(func, x, eps=1e-5):
	"""
	Inputs: 
	- func: function output scalar
	- x: matrix array TxdX
	"""
	dimx1, dimx2 = x.shape
	grad_x = np.zeros([dimx1, dimx2])
	
	deps = eps*np.ones([dimx1])
	for i in range(dimx2):
		oldVal = np.copy(x[:, i])
		x[:, i] = oldVal + deps 
		fxp = func(x)
		x[:, i] = oldVal - deps
		fxm = func(x)
		x[:, i] = oldVal
	
		grad_x[:,i] = (fxp - fxm) / (2*deps)
	return grad_x

def evalHessian(func, x, eps=1e-5):
	dimx1, dimx2 = x.shape
	h = np.zeros([dimx1, dimx2, dimx2])
	
	deps = eps*np.ones([dimx1])
	# Compute all h(i,i)
	for i in range(dimx2):
		x_p = np.copy(x)
		x_p[:, i] = x[:, i]+deps
		x_s = np.copy(x)
		x_s[:, i] = x[:, i]-deps
		
		h[:,i,i] = (func(x_p) - 2*func(x) + func(x_s)) / (eps**2)
	# Compute the rest h(i,j)
	for i in range(dimx2):
		for j in range(dimx2):
			 if i != j:
			 	x_p = np.copy(x)
				x_p[:, i] = x[:, i]+deps
				x_p[:, j] = x[:, j]+deps
				
				x_s = np.copy(x)
				x_s[:, i] = x[:, i]-deps
				x_s[:, j] = x[:, j]-deps
				
				h[:,i,j] = (func(x_p) - 2*func(x) + func(x_s)) / (2*eps**2) - (h[:,i,i] + h[:,j,j])/2
    
	return h

def runTest(itr_load):
	data_files_dir = config['common']['data_files_dir']
	data_logger = DataLogger()
	
	algorithm_file = data_files_dir + 'algorithm_itr_%02d.pkl' % itr_load
	algorithm = data_logger.unpickle(algorithm_file)
	if algorithm is None:
	    print("Error: cannot find '%s.'" % algorithm_file)
	    os._exit(1) # called instead of sys.exit(), since this is in a thread
	
	pol = algorithm.cur[0].traj_distr
	
	agent_hyperparams = deepcopy(AGENT)
	agent_hyperparams.update(config['agent'])
	
	x0 = agent_hyperparams["x0"]
	T = agent_hyperparams['T']
	dX = x0.shape[0]
	dU = agent_hyperparams['sensor_dims'][ACTION]
	
	traj_info = algorithm.cur[0].traj_info
	traj_info.x0mu = x0
	traj_info.x0sigma = np.zeros([dX, dX])
	
	# Setup for MPC policy
	M = agent_hyperparams['M'] # Short Horizon
	mpc = MpcTrajOpt(M)
	mpc_params = {
	    'type': init_pd,
	    'init_var': 5.0,
	    'pos_gains': 0.0,
	    'dQ': 2,
	    'dt': agent_hyperparams['dt'],
	    'T': M,
	    'x0': x0,
        'dX': dX,
        'dU': dU
	}
	mpc_pol = init_pd(mpc_params)
	
	agent_hyperparams['render'] = True
	agent = config['agent']['type'](agent_hyperparams)
	
	agent_hyperparams['T'] = agent_hyperparams['M']
	mpc_agent = config['agent']['type'](agent_hyperparams)
	
	# Sample using online MPC trajectory distribution.
	X_t = x0
	N = int(ceil(T/(M-1.)))
	for n in range(N):
		# Note: M-1 because action[M] = [0,0].
		t_traj = n*(M-1)
		reset = True if(n == 0) else False
		
		mpc.update(mpc_pol, X_t, pol, traj_info, t_traj)
		new_sample = mpc_agent.sample(mpc_pol, 0, reset=reset, noisy=False)
		X_t = new_sample.get_X(t=M-1)
	
	"""
	 Merge sample for optimize offline trajectory distribution
	"""
	full_sample = Sample(agent)
	sample_lists = mpc_agent.get_samples(0)
	
	keys = sample_lists[0]._data.keys()
	t = 0
	for sample in sample_lists:
		for m in range(sample.T-1):
			for sensor in keys:
				full_sample.set(sensor, sample.get(sensor, m), t)
			t = t+1
			if t+1 > T:	break

	print full_sample.get_X()
	
	# TODO: Save sample here
	
	"""
	agent_hyperparams = deepcopy(AGENT)
	agent_hyperparams.update(config['agent'])
	agent_hyperparams['render'] = True
	agent_hyperparams['T'] = 100
	agent = config['agent']['type'](agent_hyperparams)
	
	poseArray = []
	agent._worlds[0].run()
	agent._worlds[0].reset_world()
	b2d_X = agent._worlds[0].get_state()
	sample = agent._init_sample(b2d_X)
	U = np.zeros([T, dU])

	noise = generate_noise(5, dU, agent_hyperparams)
	#noise = np.zeros((T, dU)) 
	print noise
	raw_input("33ff")
	for t in range(T):
		X_t = sample.get_X(t=t)
		obs_t = sample.get_obs(t=t)
		
		# Note: M-1 because action[M] = [0,0].
		t_mpc = t % (M-1)
		if t_mpc == 0:
			# Find out MPC using sample from offline trajectory distribtion.
			mpc_pol, mu, sigma = mpc.update(mpc_pol, X_t, pol, traj_info, t)
			for ti in range(M-1):
				agent._worlds[0].drawPose(poseArray, mu[ti,:2])
				agent._worlds[0].run_next(np.zeros(dU))

		U[t, :] = mpc_pol.act(X_t, obs_t, t_mpc, noise[t, :])
		
		if (t+1) < T:
			print t, X_t, U[t, :]
			for _ in range(agent_hyperparams['substeps']):
				agent._worlds[0].run_next(U[t, :])
			b2d_X = agent._worlds[0].get_state()
			agent._set_sample(sample, b2d_X, t)
			sample.set(ACTION, U)
			
	agent._worlds[0].clearPose(poseArray)
	"""
    
def main():
	print 'running box2d'
	
	exp_name = "box2d_pointmass_example"
	#exp_name = "box2d_arm_example"
	hyperparams = loadExperiment(exp_name)
	global config
	config = hyperparams.config
	
	#runExperiment()
	argv = sys.argv
	runTest(int(argv[1]))
	print "pass test"

if __name__ == '__main__':
    main()
