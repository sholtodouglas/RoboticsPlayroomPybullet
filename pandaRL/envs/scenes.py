import numpy as np
import os
def nothing_scene(bullet_client, offset, flags):

    return []
def default_scene(bullet_client, offset, flags):

    bullet_client.loadURDF("tray/traybox.urdf", [0 + offset[0], -0.1 + offset[1], -0.6 + offset[2]],
                                [-0.5, -0.5, -0.5, 0.5], flags=flags)


    return []


def push_scene(bullet_client, offset, flags):
    default_scene(bullet_client, offset, flags)
    legos = []
    legos.append(
        bullet_client.loadURDF(os.path.dirname(os.path.abspath(__file__)) + "/lego/lego.urdf", np.array([0.1, 0.3, -0.5]) + offset, flags=flags))

    return legos


def complex_scene(bullet_client, offset, flags):
    default_scene(bullet_client, offset, flags)
    legos = []
    legos.append(
        bullet_client.loadURDF("lego/lego.urdf", np.array([-0.1, 0.3, -0.5]) + offset, flags=flags))
    legos.append(
        bullet_client.loadURDF("lego/lego.urdf", np.array([0.1, 0.3, -0.7]) + offset, flags=flags))
    sphereId = bullet_client.loadURDF("sphere_small.urdf", np.array([0, 0.3, -0.6]) + offset,
                                      flags=flags)
    bullet_client.loadURDF("sphere_small.urdf", np.array([0, 0.3, -0.5]) + offset, flags=flags)
    bullet_client.loadURDF("sphere_small.urdf", np.array([0, 0.3, -0.7]) + offset, flags=flags)
    return legos