import math

class ControlLaw(object):
    def __init__(self):
        pass

    @staticmethod
    def wrap_pos_neg_pi(angle):
        return math.fmod(angle + math.pi, math.pi*2) - math.pi;

    @staticmethod
    def convert_to_egopolar(current_pose, current_goal_pose):
        dx = current_goal_pose[0] - current_pose[0]
        dy = current_goal_pose[1] - current_pose[1]
        obs_heading = math.atan2(dy, dx)
        current_yaw = current_pose[2]
        goal_yaw = current_goal_pose[2]

        # calculate r
        r = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
        # calculate delta
        delta = ControlLaw.wrap_pos_neg_pi(current_yaw - obs_heading)
        # calculate theta
        theta = ControlLaw.wrap_pos_neg_pi(goal_yaw - obs_heading)

        return (r, delta, theta)
