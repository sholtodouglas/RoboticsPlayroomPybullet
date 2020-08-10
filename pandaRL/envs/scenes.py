import numpy as np
import os
def nothing_scene(bullet_client, offset, flags):

    return []
def default_scene(bullet_client, offset, flags, env_range_low, env_range_high):

    
    # bullet_client.loadURDF("tray/traybox.urdf", [0 + offset[0], -0.1 + offset[1], -0.6 + offset[2]],
    #                             [-0.5, -0.5, -0.5, 0.5], flags=flags)
    plane_extent = 2
    colcubeId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX,
                                                   halfExtents=[plane_extent,plane_extent, 0.0001])
    visplaneId = bullet_client.createVisualShape(bullet_client.GEOM_BOX,
                                                 halfExtents=[plane_extent, plane_extent, 0.0001],
                                                 rgbaColor=[1, 1, 1, 1])
    plane = bullet_client.createMultiBody(0, colcubeId, visplaneId, [0, 0, -0.07])

    return []

def tray_box(bullet_client, offset, flags, env_range_low, env_range_high):
    bullet_client.loadURDF("tray/traybox.urdf", [0 + offset[0], 0.0 + offset[1], -0.1 + offset[2]],
                                [0,0,0,1], flags=flags)


def push_scene(bullet_client, offset, flags, env_range_low, env_range_high):

    default_scene(bullet_client, offset, flags, env_range_low, env_range_high)
    tray_box(bullet_client, offset, flags, env_range_low, env_range_high)

    legos = []
    side = 0.025
    colcubeId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX, halfExtents=[side, side, side])
    visplaneId = bullet_client.createVisualShape(bullet_client.GEOM_BOX, halfExtents=[side, side, side],
                                                 rgbaColor=[1, 1, 1, 1])
    block = bullet_client.createMultiBody(0.1, colcubeId, visplaneId, [0, -0.06, -0.06])

    legos.append(block)
        #bullet_client.loadURDF(os.path.dirname(os.path.abspath(__file__)) + "/lego/lego.urdf", np.array([0.1, 0.3, -0.5]) + offset, flags=flags))
    add_hinge(bullet_client, offset, flags)
    return legos


def complex_scene(bullet_client, offset, flags, env_range_low, env_range_high):
    #default_scene(bullet_client, offset, flags, env_range_low, env_range_high)
    #tray_box(bullet_client, offset, flags, env_range_low, env_range_high)
    plane_extent = 2
    colcubeId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX,
                                                   halfExtents=[plane_extent, plane_extent, 0.0001])
    visplaneId = bullet_client.createVisualShape(bullet_client.GEOM_BOX,
                                                 halfExtents=[plane_extent, plane_extent, 0.0001],
                                                 rgbaColor=[1, 1, 1, 1])
    plane = bullet_client.createMultiBody(0, colcubeId, visplaneId, [0, 0, -0.27])

    legos = []
    side = 0.025
    colcubeId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX, halfExtents=[side*2, side, side])
    visplaneId = bullet_client.createVisualShape(bullet_client.GEOM_BOX, halfExtents=[side*2, side, side],
                                                 rgbaColor=[0, 0, 1, 1])
    block = bullet_client.createMultiBody(0.1, colcubeId, visplaneId, [0, -0.06, -0.06])

    visplaneId2 = bullet_client.createVisualShape(bullet_client.GEOM_BOX, halfExtents=[side*2, side, side],
                                                 rgbaColor=[1, 0, 0, 1])
    block2 = bullet_client.createMultiBody(0.1, colcubeId, visplaneId2, [-0.6, -0.06, -0.006])
    legos.append(block)
    legos.append(block2)

    door = add_door(bullet_client)
    drawer = add_drawer(bullet_client)
    dial, toggleGrill = add_dial(bullet_client)
    button, toggleSphere = add_button(bullet_client)
    add_static(bullet_client)


    return legos, [door, drawer, button, dial], {button: ('button', toggleSphere), dial: ('dial', toggleGrill)} # return the toggle sphere with it's joint index

def add_static(bullet_client):
    # TableTop
    width = 0.35
    colcubeId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX, halfExtents=[width, 0.28, 0.005])
    visplaneId = bullet_client.createVisualShape(bullet_client.GEOM_BOX, halfExtents=[width, 0.28, 0.005],
                                                 rgbaColor=[0.75, 0.4, 0.2, 1])
    block = bullet_client.createMultiBody(0.0, colcubeId, visplaneId, [0, 0.25, -0.03])

    # Cabinet back
    colcubeId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX, halfExtents=[width, 0.01, 0.235])
    visplaneId = bullet_client.createVisualShape(bullet_client.GEOM_BOX, halfExtents=[width, 0.01, 0.235],
                                                 rgbaColor=[0.75, 0.4, 0.2, 1])
    block = bullet_client.createMultiBody(0.0, colcubeId, visplaneId, [0., 0.52, -0.00])

    # Cabinet top
    width = 0.37
    colcubeId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX, halfExtents=[width, 0.065, 0.005])
    visplaneId = bullet_client.createVisualShape(bullet_client.GEOM_BOX, halfExtents=[width, 0.065, 0.005],
                                                 rgbaColor=[0.75, 0.4, 0.2, 1])
    block = bullet_client.createMultiBody(0.0, colcubeId, visplaneId, [0., 0.45, 0.24])

    # Cabinet sides
    width = 0.03
    colcubeId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX, halfExtents=[width, 0.065, 0.235])
    visplaneId = bullet_client.createVisualShape(bullet_client.GEOM_BOX, halfExtents=[width, 0.065, 0.235],
                                                 rgbaColor=[0.75, 0.4, 0.2, 1])
    block = bullet_client.createMultiBody(0.0, colcubeId, visplaneId, [-0.34, 0.45, -0.00])

    colcubeId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX, halfExtents=[width, 0.065, 0.235])
    visplaneId = bullet_client.createVisualShape(bullet_client.GEOM_BOX, halfExtents=[width, 0.065, 0.235],
                                                 rgbaColor=[0.75, 0.4, 0.2, 1])
    block = bullet_client.createMultiBody(0.0, colcubeId, visplaneId, [0.34, 0.45, -0.00])

def add_door(bullet_client, offset=np.array([0, 0, 0]), flags=None):
    sphereRadius = 0.1
    colBoxId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX,
                                                  halfExtents=[sphereRadius, sphereRadius, sphereRadius])

    #     wallid = bullet_client.createCollisionShape(bullet_client.GEOM_BOX,
    #                                       halfExtents=[sphereRadius*4, sphereRadius/4, sphereRadius*4])
    wallid = bullet_client.createCollisionShape(bullet_client.GEOM_MESH, fileName=os.path.dirname(os.path.abspath(__file__)) + '/door.obj', meshScale=[0.0015] * 3,
                                        flags=bullet_client.GEOM_FORCE_CONCAVE_TRIMESH)

    mass = 0
    visualShapeId = -1

    link_Masses = [0.1]
    linkCollisionShapeIndices = [wallid]
    linkVisualShapeIndices = [-1]
    linkPositions = [[0.0, 0.0, 0.27]]
    linkOrientations = [bullet_client.getQuaternionFromEuler([0, np.pi / 2, 0])]
    linkInertialFramePositions = [[0, 0, 0.0]]
    linkInertialFrameOrientations = [[0, 0, 0, 1]]
    indices = [0]
    # jointTypes = [bullet_client.JOINT_REVOLUTE]
    jointTypes = [bullet_client.JOINT_PRISMATIC]
    axis = [[0, 0, 1]]

    basePosition = np.array([0, 0.4, -0.2]) + offset
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
    return sphereUid

def add_button(bullet_client, offset=np.array([0, 0, 0])):
    sphereRadius = 0.02
    colBoxId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX,
                                                  halfExtents=[sphereRadius, sphereRadius, sphereRadius / 4])

    mass = 0
    visualShapeId = bullet_client.createVisualShape(bullet_client.GEOM_BOX,
                                                    halfExtents=[sphereRadius, sphereRadius, sphereRadius / 4],
                                                    rgbaColor=[1, 0, 0, 1])

    link_Masses = [0.1]
    linkCollisionShapeIndices = [colBoxId]
    linkVisualShapeIndices = [visualShapeId]
    x = -0.25
    y = 0.45
    linkPositions = [[x,y, 0.70]]
    linkOrientations = [bullet_client.getQuaternionFromEuler([0, 0, 0])]
    linkInertialFramePositions = [[0, 0, 0]]
    linkInertialFrameOrientations = [[0, 0, 0, 1]]
    indices = [0]
    # jointTypes = [bullet_client.JOINT_REVOLUTE]
    jointTypes = [bullet_client.JOINT_PRISMATIC]
    axis = [[0, 0, 1]]

    basePosition = np.array([0, 0, -0.7]) + offset
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
    bullet_client.setJointMotorControl2(sphereUid, 0, bullet_client.POSITION_CONTROL, targetPosition=0.03, force=1)

    # create a little globe to turn on and off
    sphereRadius = 0.03
    colSphereId = bullet_client.createCollisionShape(bullet_client.GEOM_SPHERE, radius=sphereRadius)
    visualShapeId = bullet_client.createVisualShape(bullet_client.GEOM_SPHERE,
                                                    radius=sphereRadius,
                                                    rgbaColor=[1, 1, 1, 1])
    toggleSphere = bullet_client.createMultiBody(0.0, colSphereId, visualShapeId, [x,y,0.24],
                                  baseOrientation)

    return sphereUid, toggleSphere


def add_drawer(bullet_client, offset=np.array([0, 0, 0]), flags=None):
    sphereRadius = 0.001
    colBoxId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX,
                                                  halfExtents=[sphereRadius, sphereRadius, sphereRadius])

    #     wallid = bullet_client.createCollisionShape(bullet_client.GEOM_BOX,
    #                                       halfExtents=[sphereRadius*4, sphereRadius/4, sphereRadius*4])
    wallid = bullet_client.createCollisionShape(bullet_client.GEOM_MESH, fileName=os.path.dirname(os.path.abspath(__file__)) + '/drawer.obj', meshScale=[0.0015] * 3,
                                        flags=bullet_client.GEOM_FORCE_CONCAVE_TRIMESH)

    mass = 0
    visualShapeId = -1

    link_Masses = [1]
    linkCollisionShapeIndices = [wallid]
    linkVisualShapeIndices = [-1]
    linkPositions = [[0.0, 0.0, 0.1]]
    linkOrientations = [bullet_client.getQuaternionFromEuler([np.pi / 2, 0, 0])]
    linkInertialFramePositions = [[0, 0, 0.0]]
    linkInertialFrameOrientations = [[0, 0, 0, 1]]
    indices = [0]
    # jointTypes = [bullet_client.JOINT_REVOLUTE]
    jointTypes = [bullet_client.JOINT_PRISMATIC]
    axis = [[0, 0, 1]]

    basePosition = np.array([-0.1, 0.0, -0.15]) + offset
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
    return sphereUid

def dial_to_0_1_range(data):
    return (data % 2*np.pi ) / (2.2*np.pi)

def add_dial(bullet_client, offset=np.array([0, 0, 0]), flags=None):
    sphereRadius = 0.0075
    colBoxId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX,
                                                  halfExtents=[sphereRadius, sphereRadius, sphereRadius])

    wallid = bullet_client.createCollisionShape(bullet_client.GEOM_BOX,
                                                halfExtents=[sphereRadius * 4, sphereRadius / 4, sphereRadius * 4])
    #  wallid =env.p.createCollisionShape(env.p.GEOM_MESH, fileName='drawer.obj', meshScale=[0.0015]*3, flags=env.p.GEOM_FORCE_CONCAVE_TRIMESH)

    mass = 0
    visualShapeId = -1

    link_Masses = [0.1]
    linkCollisionShapeIndices = [wallid]
    linkVisualShapeIndices = [-1]
    linkPositions = [[0.20, -0.055, 0.0]]
    linkOrientations = [bullet_client.getQuaternionFromEuler([np.pi / 2, 0, 0])]
    linkInertialFramePositions = [[0, 0, 0.0]]
    linkInertialFrameOrientations = [[0, 0, 0, 1]]
    indices = [0]
    jointTypes = [bullet_client.JOINT_REVOLUTE]
    # jointTypes = [bullet_client.JOINT_PRISMATIC]
    axis = [[0, 0, 1]]

    basePosition = np.array([0, 0.0, -0.07]) + offset
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

    # create a little globe to turn on and off
    width = 0.07
    # create a grill to turn on/off
    colSphereId = bullet_client.createCollisionShape(bullet_client.GEOM_BOX,
                                                  halfExtents=[width, width, 0.01])
    visualShapeId = bullet_client.createVisualShape(bullet_client.GEOM_BOX,
                                                  halfExtents=[width, width, 0.01],
                                                    rgbaColor=[1, 1, 1, 1])
    toggleGrill = bullet_client.createMultiBody(0.0, colSphereId, visualShapeId, [0.2, 0.1, -0.03],
                                                 baseOrientation)

    return sphereUid, toggleGrill

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
    linkOrientations = [bullet_client.getQuaternionFromEuler([0,np.pi/2,0])]
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
    return sphereUid
