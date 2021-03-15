
Environments to support https://github.com/sholtodouglas/learning_from_play

[An example playing out the teleop data is here](https://github.com/sholtodouglas/learning_from_play/blob/ecd16531422e6e123d22aa58a6abd5d9dc08abfa/notebooks/Minimal%20Example.ipynb) ,this example replays the teleop data and also showcases the functionality of the env as an RL environment (reset goal, visualise goal, sparse reward function etc).

[An example where you can use the GUI to control the robot is here](https://github.com/sholtodouglas/RoboticsPlayroomPybullet/blob/280818586cf001599110acaddb78d216d5056914/roboticsPlayroomPybullet/envs/interactive.py)

[A more involved example is here](https://github.com/sholtodouglas/learning_from_play/blob/ecd16531422e6e123d22aa58a6abd5d9dc08abfa/notebooks/Deploy.ipynb), which shows how a test rig is used to reset the environment to specific locations and test goals. 

This is an overview of the code, the comments should guide you into understanding any pieces you wish to modify. 
![alt-text-1](/roboticsPlayroomPybullet/readme_assets/code.png "side by side comparison")


### UR5PlayAbsRPY1Obj-v0
<p align="center">
	<a href="https://sholtodouglas.github.io/images/play/headline.gif">
		<img src="media/headline.gif">
	</a>
</p>


The environment also supports a number of more basic test environments, and other arms. The full list of these is defined in the gym registry here at - roboticsPlayroomPybullet/__init__.py. E.g.

### UR5Reach-v0

![alt-text-1](/roboticsPlayroomPybullet/readme_assets/urreach.png "side by side comparison")

### pandaPick-v0
![alt-text-1](/roboticsPlayroomPybullet/readme_assets/pandaPick.png "side by side comparison")
