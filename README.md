
Environments to support https://github.com/sholtodouglas/learning_from_play and reinforcement learning for robotic manipulation.

[An example playing out the teleop data](https://github.com/sholtodouglas/learning_from_play/blob/ecd16531422e6e123d22aa58a6abd5d9dc08abfa/notebooks/Minimal%20Example.ipynb), which also showcases the functionality of the env as an RL environment (reset goal, visualise goal, reward function etc).

[An example where you can use the GUI to control the robot.](https://github.com/sholtodouglas/RoboticsPlayroomPybullet/blob/280818586cf001599110acaddb78d216d5056914/roboticsPlayroomPybullet/envs/interactive.py)

[A more involved example](https://github.com/sholtodouglas/learning_from_play/blob/ecd16531422e6e123d22aa58a6abd5d9dc08abfa/notebooks/Deploy.ipynb), which shows how a test rig is used to reset the environment to specific locations and test goals, and plays a pretrained model.

[The file to run if you want to teleoperate new trajectories](https://github.com/sholtodouglas/learning_from_play/blob/77be8ecd9f6c5a730c49b502ca56f2f26f938d6c/data_collection/vr_data_collection.py), this works with any VR headset which can interface with steamVR. Instructions are within the file.

A high level overview of the code can be found [here.](https://github.com/sholtodouglas/RoboticsPlayroomPybullet/blob/8a7fba19502e7a7024708c6ce5d6e1f546c18364/roboticsPlayroomPybullet/readme_assets/code.png)



### UR5PlayAbsRPY1Obj-v0

![alt-text-1](https://github.com/sholtodouglas/learning_from_play/blob/d17bf6dc0fafc74b7a3d63977efc41c67f1640c6/media/headline.gif)


The environment also supports a number of more basic test environments, and other arms. The full list of these is defined in the gym registry [here.](https://github.com/sholtodouglas/RoboticsPlayroomPybullet/blob/2fef21ed1683de7fb9b9b60e8e96ff3a082a6499/roboticsPlayroomPybullet/__init__.py)

### UR5Reach-v0

![alt-text-1](/roboticsPlayroomPybullet/readme_assets/urreach.png "side by side comparison")

### pandaPick-v0
![alt-text-1](/roboticsPlayroomPybullet/readme_assets/pandaPick.png "side by side comparison")
