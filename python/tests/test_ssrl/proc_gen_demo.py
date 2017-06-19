import os
import signal
import subprocess
import time
import argparse

def wait_timeout(proc, seconds):
    """Wait for a process to finish, or raise exception after timeout"""
    start = time.time()
    end = start + seconds
    interval = min(seconds / 1000.0, .25)

    while True:
        result = proc.poll()
        if result is not None:
            return result
        if time.time() >= end:
            os.killpg(os.getpgid(pro.pid), signal.SIGTERM)  # Send the signal to all the process groups
            #raise RuntimeError("Process timed out")
        time.sleep(interval)

def run_gen_demo(exp_name, iteration):
    # The os.setsid() is passed in the argument preexec_fn so
    # it's run after the fork() and before  exec() to run the shell.
    cmd = "python python/tests/test_ssrl/manual_gen_demo.py "+exp_name+" -m " + str(iteration)
    pro = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                           shell=True, preexec_fn=os.setsid) 
    
    print i, ", ", wait_timeout(pro, 20.0)

def run_merge(exp_name):
    cmd = "python python/tests/test_ssrl/manual_gen_demo.py "+exp_name+" --combine 1"
    pro = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                           shell=True, preexec_fn=os.setsid) 
    
    print i, ", ", wait_timeout(pro, 20.0)


parser = argparse.ArgumentParser(description='Run the Guided Policy Search algorithm.')
parser.add_argument('experiment', type=str,
                    help='experiment name')
parser.add_argument('-m', '--model', metavar='N', type=int,
                        help='Demo model')
args = parser.parse_args()
exp_name = args.experiment
n_model = args.model

for i in range(n_model):
    run_gen_demo(exp_name,i)