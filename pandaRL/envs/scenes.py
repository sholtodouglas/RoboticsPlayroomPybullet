import numpy as np
import os
def nothing_scene(bullet_client, offset, flags):

    return []
def default_scene(bullet_client, offset, flags, env_range_low, env_range_high):

    
    # bullet_client.loadURDF("tray/traybox.urdf", [0 + offset[0], -0.1 + offset[1], -0.6 + offset[2]],
    #                             [-0.5, -0.5, -0.5, 0.5], flags=flags)
    plane_extent = 2
    colcubeId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX,
                                                   halfExtents=[plane_extent, 0.0001, plane_extent])
    visplaneId = bullet_client.createVisualShape(bullet_client.GEOM_BOX,
                                                 halfExtents=[plane_extent, 0.0001, plane_extent],
                                                 rgbaColor=[1, 1, 1, 1])
    plane = bullet_client.createMultiBody(0, colcubeId, visplaneId, [0, -0.2, -0.6])

    return []


def push_scene(bullet_client, offset, flags, env_range_low, env_range_high):

    plane_extent = 2
    colcubeId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX,
                                                   halfExtents=[plane_extent, 0.0001, plane_extent])
    visplaneId = bullet_client.createVisualShape(bullet_client.GEOM_BOX,
                                                 halfExtents=[plane_extent, 0.0001, plane_extent],
                                                 rgbaColor=[1, 1, 1, 1])
    plane = bullet_client.createMultiBody(0, colcubeId, visplaneId, [0, -0.07, -0.6])
    bullet_client.loadURDF("tray/traybox.urdf", [0 + offset[0], -0.1 + offset[1], -0.6 + offset[2]],
                                [-0.5, -0.5, -0.5, 0.5], flags=flags)

    legos = []
    side = 0.025
    colcubeId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX, halfExtents=[side, side, side])
    visplaneId = bullet_client.createVisualShape(bullet_client.GEOM_BOX, halfExtents=[side, side, side],
                                                 rgbaColor=[1, 1, 1, 1])
    block = bullet_client.createMultiBody(0.1, colcubeId, visplaneId, [0, -0.06, -0.6])

    legos.append(block)
        #bullet_client.loadURDF(os.path.dirname(os.path.abspath(__file__)) + "/lego/lego.urdf", np.array([0.1, 0.3, -0.5]) + offset, flags=flags))
    add_hinge(bullet_client, offset, flags)
    return legos


def complex_scene(bullet_client, offset, flags, env_range_low, env_range_high):
    plane_extent = 2
    colcubeId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX,
                                                   halfExtents=[plane_extent, 0.0001, plane_extent])
    visplaneId = bullet_client.createVisualShape(bullet_client.GEOM_BOX,
                                                 halfExtents=[plane_extent, 0.0001, plane_extent],
                                                 rgbaColor=[1, 1, 1, 1])
    plane = bullet_client.createMultiBody(0, colcubeId, visplaneId, [0, -0.07, -0.6])
    bullet_client.loadURDF("tray/traybox.urdf", [0 + offset[0], -0.1 + offset[1], -0.6 + offset[2]],
                                [-0.5, -0.5, -0.5, 0.5], flags=flags)

    legos = []
    side = 0.025
    colcubeId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX, halfExtents=[side, side, side])
    visplaneId = bullet_client.createVisualShape(bullet_client.GEOM_BOX, halfExtents=[side, side, side],
                                                 rgbaColor=[0, 0, 1, 1])
    block = bullet_client.createMultiBody(0.1, colcubeId, visplaneId, [0, -0.06, -0.6])

    visplaneId2 = bullet_client.createVisualShape(bullet_client.GEOM_BOX, halfExtents=[side, side, side],
                                                 rgbaColor=[1, 0, 0, 1])
    block2 = bullet_client.createMultiBody(0.1, colcubeId, visplaneId2, [-0.6, -0.06, -0.0])
    legos.append(block)
    legos.append(block2)



    return legos


def add_hinge(bullet_client, offset, flags):
    sphereRadius = 0.05
    colBoxId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX,
                                      halfExtents=[sphereRadius, sphereRadius, sphereRadius])

    mass = 0
    visualShapeId = -1

    link_Masses = [1]
    linkCollisionShapeIndices = [colBoxId]
    linkVisualShapeIndices = [-1]
    linkPositions = [[0.0,0.0, -0.5]]
    linkOrientations = [bullet_client.getQuaternionFromEuler([0,0,np.pi/2])]
    linkInertialFramePositions = [[0, 0, 0]]
    linkInertialFrameOrientations = [[0, 0, 0, 1]]
    indices = [0]
    # jointTypes = [bullet_client.JOINT_REVOLUTE]
    jointTypes = [bullet_client.JOINT_PRISMATIC]
    axis = [[0, 0, 1]]

    basePosition = np.array([0, 0, 0])+offset
    baseOrientation = [0, 0, 0, 1]

    sphereUid = bullet_client.createMultiBody(mass,
                                  colBoxId,
                                  visualShapeId,
                                  basePosition,
                                  baseOrientation,
                                  linkMasses=link_Masses,
                                  linkCollisionShapeIndices=linkCollisionShapeIndices,
                                  linkVisualShapeIndices=linkVisualShapeIndices,
                                  linkPositions=linkPositions,
                                  linkOrientations=linkOrientations,
                                  linkInertialFramePositions=linkInertialFramePositions,
                                  linkInertialFrameOrientations=linkInertialFrameOrientations,
                                  linkParentIndices=indices,
                                  linkJointTypes=jointTypes,
                                  linkJointAxis=axis)

    bullet_client.changeDynamics(sphereUid,
                     -1,
                     spinningFriction=0.001,
                     rollingFriction=0.001,
                     linearDamping=0.0)
