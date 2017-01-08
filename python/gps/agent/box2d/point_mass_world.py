""" This file defines an environment for the Box2D PointMass simulator. """
import numpy as np
import Box2D as b2
from framework import Framework

from gps.agent.box2d.settings import fwSettings
from gps.proto.gps_pb2 import END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, POSITION_NEAREST_OBSTACLE

class PointMassWorld(Framework):
    """ This class defines the point mass and its environment."""
    name = "PointMass"
    def __init__(self, x0, target, render):
        self.render = render
        if self.render:
            super(PointMassWorld, self).__init__()
        else:
            self.world = b2.b2World(gravity=(0, -10), doSleep=True)
        self.world.gravity = (0.0, 0.0)
        self.initial_position = (x0[0], x0[1])
        self.initial_angle = b2.b2_pi
        self.initial_linear_velocity = (x0[2], x0[3])
        self.initial_angular_velocity = 0

        ground = self.world.CreateBody(position=(0, 20))
        ground.CreateEdgeChain(
            [(-20, -20),
             (-20, 20),
             (20, 20),
             (20, -20),
             (-20, -20)]
            )

        xf1 = b2.b2Transform()
        xf1.angle = 0.3524 * b2.b2_pi
        xf1.position = b2.b2Mul(xf1.R, (1.0, 0.0))

        xf2 = b2.b2Transform()
        xf2.angle = -0.3524 * b2.b2_pi
        xf2.position = b2.b2Mul(xf2.R, (-1.0, 0.0))
        
        self.body_shape = [b2.b2PolygonShape(vertices=[xf1*(-1, 0),
                                                xf1*(1, 0), xf1*(0, .5)]),
                    b2.b2PolygonShape(vertices=[xf2*(-1, 0),
                                                xf2*(1, 0), xf2*(0, .5)])]
        self.body = self.world.CreateDynamicBody(
            position=self.initial_position,
            angle=self.initial_angle,
            linearVelocity=self.initial_linear_velocity,
            angularVelocity=self.initial_angular_velocity,
            angularDamping=5,
            linearDamping=0.1,
            shapes=self.body_shape,
            shapeFixture=b2.b2FixtureDef(density=1.0),
        )
        self.target = self.world.CreateStaticBody(
            position=target[:2],
            angle=self.initial_angle,
            shapes=[b2.b2PolygonShape(vertices=[xf1*(-1, 0), xf1*(1, 0),
                                                xf1*(0, .5)]),
                    b2.b2PolygonShape(vertices=[xf2*(-1, 0), xf2*(1, 0),
                                                xf2*(0, .5)])],
        )
        
        self.obstacle_shape = [b2.b2PolygonShape(box=(3,1))]
        self.obstacle = self.world.CreateStaticBody(
            position=np.array([0, 10]),
            angle=self.initial_angle,
            shapes=self.obstacle_shape
        )
        
        
        self.target.active = False

    def run(self):
        """Initiates the first time step
        """
        if self.render:
            super(PointMassWorld, self).run()
        else:
            self.run_next(None)

    def run_next(self, action):
        """Moves forward in time one step. Calls the renderer if applicable."""
        if self.render:
            super(PointMassWorld, self).run_next(action)
            """
            hist = self.world.CreateStaticBody(
                position=self.body.position,
                angle=self.body.angle,
                shapes=self.body_shape
            )
            hist.active = False
            """
            #self.world.DestroyBody(hist)
        else:
            if action is not None:
                self.body.linearVelocity = (action[0], action[1])
            self.world.Step(1.0 / fwSettings.hz, fwSettings.velocityIterations,
                            fwSettings.positionIterations)

    def Step(self, settings, action):
        """Called upon every step. """
        self.body.linearVelocity = (action[0], action[1])

        super(PointMassWorld, self).Step(settings)

    def reset_world(self):
        """ This resets the world to its initial state"""
        self.world.ClearForces()
        self.body.position = self.initial_position
        self.body.angle = self.initial_angle
        self.body.angularVelocity = self.initial_angular_velocity
        self.body.linearVelocity = self.initial_linear_velocity

    def get_nearest_dist_obs(self):
        distanceInput = b2.b2DistanceInput();
        distanceInput.transformA = self.body.transform
        distanceInput.transformB = self.obstacle.transform
        # TODO: Show how multi polygon to one shape instance
        distanceInput.proxyA = b2.b2DistanceProxy(self.body_shape[0])
        distanceInput.proxyB = b2.b2DistanceProxy(self.obstacle_shape[0])
        distanceInput.useRadii = True
        
        distanceOutput = b2.b2Distance(distanceInput)
        
        """
        # euclidian_dist == distanceOutput.distance
        euclidian_dist = np.sqrt((self.body.position[0] - distanceOutput.pointB[0]) ** 2 + \
                (self.body.position[1] - distanceOutput.pointB[1]) ** 2)
        
        # gradient of euclidian_dist. For simplific, I power 2 the distance.
        grad = np.append(np.array(self.body.position), [0])
        
        return np.array([0.5 * euclidian_dist ** 2])
        """
        
        return np.append(np.array(distanceOutput.pointB), [0])

    def drawPose(self, poseArray, position):
        hist = self.world.CreateStaticBody(
            position=position,
            angle=self.body.angle,
            shapes=self.body_shape
        )
        hist.active = False
        poseArray.append(hist)
    
    def clearPose(self, poseArray):
        for hist in poseArray:
            self.world.DestroyBody(hist)

    def get_state(self):
        """ This retrieves the state of the point mass"""
        state = {END_EFFECTOR_POINTS: np.append(np.array(self.body.position), [0]),
                 END_EFFECTOR_POINT_VELOCITIES: np.append(np.array(self.body.linearVelocity), [0]),
                 POSITION_NEAREST_OBSTACLE: self.get_nearest_dist_obs()}

        return state
