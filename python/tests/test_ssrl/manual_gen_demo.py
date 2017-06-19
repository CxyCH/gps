import random

import logging
import imp
import os
import os.path
import sys
import copy
import argparse
import threading
import time
import scipy.io
import numpy as np
import numpy.matlib
from random import shuffle

LOGGER = logging.getLogger(__name__)

# Add gps/python to path so that imports work.
gps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', ''))
sys.path.append(gps_path)

from gps.utility.data_logger import DataLogger
from gps.sample.sample_list import SampleList
from gps.utility.demo_utils import get_demos, extract_samples
from gps.utility.general_utils import disable_caffe_logs, Timer, mkdir_p, compute_distance

class ManGenDemo(object):
    def __init__(self, config, old_cond, i_model):
        self._hyperparams = config
        '''
        self._conditions = config['common']['conditions']
        if 'train_conditions' in config['common']:
            if type(config['common']['train_conditions']) is list:
                self._train_idx = config['common']['train_conditions']
                self._test_idx = config['common']['test_conditions']
            else:
                self._train_idx = range(config['common']['train_conditions'])
                self._test_idx = range(config['common']['test_conditions'])
            if not self._test_idx:
                self._test_idx = self._train_idx
        else:
            self._train_idx = range(self._conditions)
            config['common']['train_conditions'] = config['common']['conditions']
            self._hyperparams=config
            self._test_idx = self._train_idx

        self._data_files_dir = config['common']['data_files_dir']
        '''
        '''
        with Timer('init agent'):
            self.agent = config['agent']['type'](config['agent'])
        if 'test_agent' in config:
            self.test_agent = config['test_agent']['type'](config['test_agent'])
        else:
            self.test_agent = self.agent
        '''
        
        self.old_cond = old_cond
        self.i_model = i_model
        self.data_logger = DataLogger()
        #with Timer('init GUI'):
        #    self.gui = GPSTrainingGUI(config['common'], gui_on=config['gui_on'])

        #config['algorithm']['agent'] = self.agent

        #if self.using_ioc() and not test_pol:
        if True:
            if config['demo_agent'].get('eval_only', False):
                from gps.utility.visualization import run_alg
                record_gif = config.get('record_gif', None)
                run_alg(config['demo_agent'], config['demo_agent']['algorithm_file'], record_gif=record_gif, verbose=True)
            else:
                with Timer('loading demos'):
                    # demos = get_demos(self)
                    # Modified
                    self.get_demos()
                    '''
                    ## Modified
                    self.algorithm.demoX = demos['demoX']
                    self.algorithm.demoU = demos['demoU']
                    self.algorithm.demoO = demos['demoO']
                    if 'demo_conditions' in demos.keys() and 'failed_conditions' in demos.keys():
                        self.algorithm.demo_conditions = demos['demo_conditions']
                        self.algorithm.failed_conditions = demos['failed_conditions']
                    '''
        else:
            with Timer('init algorithm'):
                self.algorithm = config['algorithm']['type'](config['algorithm'])
        
    def get_demos(self):
        """
        Gather the demos for IOC algorithm. If there's no demo file available, generate it.
        Args:
            gps: the gps object.
        Returns: the demo dictionary of demo tracjectories.
        """
        gps = self
        from gps.utility.generate_demo import GenDemo
    
        if gps._hyperparams['common'].get('nn_demo', False):
            demo_file = gps._hyperparams['common']['NN_demo_file'] # using neural network demos
        else:
            demo_file = gps._hyperparams['common']['LG_demo_file'] # using linear-Gaussian demos
        
        """ 
         Modified here:
         If it is not 1st model: Unpickle demos, Generate, and combine  
        """
        gps.demo_gen = GenDemo(gps._hyperparams)
        if self.i_model == 0: # 1st model
            demos = self.generate(gps.demo_gen, demo_file) 
        else:
            demos = gps.data_logger.unpickle(demo_file)
            dem = self.generate(gps.demo_gen, demo_file)
            
            demos['demoX'] = np.concatenate((demos['demoX'], dem['demoX']), axis=0)
            demos['demoU'] = np.concatenate((demos['demoU'], dem['demoU']), axis=0)
            demos['demoO'] = np.concatenate((demos['demoO'], dem['demoO']), axis=0)
            demos['demoConditions'] += dem['demoConditions']
                    
        # Store temp data
        self.data_logger.pickle(
            demo_file,
            copy.copy(demos)
        )

        print 'Num demos:', demos['demoX'].shape[0]
        
        '''
        ### Modified
        gps._hyperparams['algorithm']['init_traj_distr']['init_demo_x'] = np.mean(demos['demoX'], 0)
        gps._hyperparams['algorithm']['init_traj_distr']['init_demo_u'] = np.mean(demos['demoU'], 0)
        gps.algorithm = gps._hyperparams['algorithm']['type'](gps._hyperparams['algorithm'])
        '''
        
    def generate(self, GenDemo, demo_file):
        print "Generate Demo"
        
        """
         Generate demos and save them in a file for experiment.
         Args:
             demo_file - place to store the demos
             ioc_agent - ioc agent, for grabbing the observation using the ioc agent's observation data types
         Returns: None.
        """
        # Load the algorithm

        self.algorithms = GenDemo.load_algorithms()
        self.algorithm = self.algorithms[0]

        # Keep the initial states of the agent the sames as the demonstrations.
        agent_config = self._hyperparams['demo_agent']
        self.agent = agent_config['type'](agent_config)

        # Roll out the demonstrations from controllers
        # var_mult = self._hyperparams['algorithm']['demo_var_mult']
        # Modified
        var_mult = 1.0
        T = self.algorithms[0].T
        demos = []
        demo_idx_conditions = []  # Stores conditions for each demo

        M = agent_config['conditions']
        N = self._hyperparams['algorithm']['num_demos']
        if not GenDemo.nn_demo:
            controllers = {}

            # Store each controller under M conditions into controllers.
            for i in xrange(M):
                controllers[i] = self.algorithm.cur[i].traj_distr
            controllers_var = copy.copy(controllers)
            for i in xrange(M):
                # Increase controller variance.
                controllers_var[i].chol_pol_covar *= var_mult
                # Gather demos.
                for j in xrange(N):
                    demo = self.agent.sample(
                        controllers_var[i], i,
                        verbose=(j < self._hyperparams['verbose_trials']), noisy=True,
                        save = True
                    )
                    demos.append(demo)
                    demo_idx_conditions.append(i)
        else:
            all_pos_body_offsets = []
            # Gather demos.
            for a in xrange(len(self.algorithms)):
                pol = self.algorithms[a].policy_opt.policy
                for i in xrange(M / len(self.algorithms) * a, M / len(self.algorithms) * (a + 1)):
                    for j in xrange(N):
                        demo = self.agent.sample(
                            pol, i,
                            verbose=(j < self._hyperparams['verbose_trials']), noisy=True
                            )
                        demos.append(demo)
                        demo_idx_conditions.append(i)
        # Modified
        return self.filter(demos, demo_idx_conditions, agent_config, self.agent, demo_file)
        
        #return demos, demo_idx_conditions
    def filter(self, demos, demo_idx_conditions, agent_config, ioc_agent, demo_file):
        """
        Filter out failed demos.
        Args:
            demos: generated demos
            demo_idx_conditions: the conditions of generated demos
            agent_config: config of the demo agent
            ioc_agent: the agent for ioc
            demo_file: the path to save demos
        """
        M = agent_config['conditions']
        N = self._hyperparams['algorithm']['num_demos']
        
        # Filter failed demos
        if 'filter_demos' in agent_config:
            filter_options = agent_config['filter_demos']
            filter_type = filter_options.get('type', 'min')
            targets = filter_options['target']
            pos_idx = filter_options.get('state_idx', range(4, 7))
            end_effector_idx = filter_options.get('end_effector_idx', range(0, 3))
            max_per_condition = filter_options.get('max_demos_per_condition', 999)
            dist_threshold = filter_options.get('success_upper_bound', 0.01)
            cur_samples = SampleList(demos)
            dists = compute_distance(targets, cur_samples, pos_idx, end_effector_idx, filter_type=filter_type)
            failed_idx = []
            for i, distance in enumerate(dists):
                if (distance > dist_threshold):
                    failed_idx.append(i)

            LOGGER.debug("Removing %d failed demos: %s", len(failed_idx), str(failed_idx))
            demos_filtered = [demo for (i, demo) in enumerate(demos) if i not in failed_idx]
            demo_idx_conditions = [cond for (i, cond) in enumerate(demo_idx_conditions) if i not in failed_idx]
            demos = demos_filtered


            # Filter max demos per condition
            condition_to_demo = {
                cond: [demo for (i, demo) in enumerate(demos) if demo_idx_conditions[i]==cond][:max_per_condition]
                for cond in range(M)
            }
            LOGGER.debug('Successes per condition: %s', str([len(demo_list) for demo_list in condition_to_demo.values()]))
            demos = [demo for cond in condition_to_demo for demo in condition_to_demo[cond]]
            shuffle(demos)

            for demo in demos: demo.reset_agent(ioc_agent)
            demo_list = SampleList(demos)
            demo_store = {'demoX': demo_list.get_X(),
                          'demoU': demo_list.get_U(),
                          'demoO': demo_list.get_obs(),
                          'demoConditions': demo_idx_conditions}
        else:
            shuffle(demos)
            for demo in demos: demo.reset_agent(ioc_agent)
            demo_list = SampleList(demos)
            demo_store = {'demoX': demo_list.get_X(), 'demoU': demo_list.get_U(), 'demoO': demo_list.get_obs()}
        
        '''
        # Save the demos.
        self.data_logger.pickle(
            demo_file,
            copy.copy(demo_store)
        )
        '''
            
        return demo_store
        

def loadExperiment(exp_name):
    from gps import __file__ as gps_filepath
    gps_filepath = os.path.abspath(gps_filepath)
    gps_dir = '/'.join(str.split(gps_filepath, '/')[:-3]) + '/'
    exp_dir = gps_dir + 'experiments/' + exp_name + '/'
    hyperparams_file = exp_dir + 'hyperparams.py'
        
    if not os.path.exists(hyperparams_file):
        sys.exit("Experiment '%s' does not exist.\nDid you create '%s'?" %
                 (exp_name, hyperparams_file))
    
    hyperparams = imp.load_source('hyperparams', hyperparams_file)
    
    return hyperparams
        
def main():
    parser = argparse.ArgumentParser(description='Run the Guided Policy Search algorithm.')
    parser.add_argument('experiment', type=str,
                        help='experiment name')
    parser.add_argument('-m', '--model', metavar='N', type=int,
                        help='Demo model')
    parser.add_argument('-c', '--combine', metavar='N', type=int,
                    help='combine with original agent')
    
    args = parser.parse_args()
    exp_name = args.experiment
    i_model = args.model
    
    hyperparams = loadExperiment(exp_name)
    config = hyperparams.config
    
    if args.combine:
        print "Combine"
        
        data_logger = DataLogger()
        demo_file = config['common']['LG_demo_file']
        unpck_data = data_logger.unpickle(demo_file)
        demo_agent, demos, demo_idx_conditions = unpck_data[0], unpck_data[1], unpck_data[2]
        
        # Copy demo agent
        for i in range(len(demos)):
            demos[i].agent = demo_agent
        
        print "Agent ", demo_agent.x_data_types
        
        agent_config = config['agent']
        ioc_agent = agent_config['type'](agent_config)
        
        from gps.utility.generate_demo import GenDemo
        demo_gen = GenDemo(config)
        demo_gen.filter(demos, demo_idx_conditions, config['demo_agent'], ioc_agent, demo_file)
        
    else:
        # Sample corresponding to model number
        old_cond = config['demo_agent']['conditions']
        config['demo_agent']['conditions'] = 1
        old_models = config['demo_agent']['models']
        config['demo_agent']['models'] = old_models[i_model]
        
        manGen = ManGenDemo(hyperparams.config, old_cond, i_model)

    
if __name__ == '__main__':
    main()