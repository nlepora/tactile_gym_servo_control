import pybullet as p
import pybullet_utils.bullet_client as bc
import pkgutil
import time 

from tactile_gym_servo_control.utils_robot_sim.robot_embodiment import POSE_UNITS
from tactile_gym_servo_control.utils_robot_sim.robot_embodiment import RobotEmbodiment
from tactile_gym_servo_control.utils.pose_transforms import transform_pose, inv_transform_pose
from tactile_gym.assets import add_assets_path


def setup_pybullet_env(
    workframe,
    stim_path,
    stim_pose,
    stim_scale,
    fix_stim,
    sensor_params,
    cam_params,
    show_gui,
    show_tactile,
    work_target_pose=[],
    quick_mode=False,
):

    # ========= environment set up ===========
    time_step = 1.0 / 240

    if show_gui:
        pb = bc.BulletClient(connection_mode=p.GUI)
    else:
        pb = bc.BulletClient(connection_mode=p.DIRECT)
        egl = pkgutil.get_loader("eglRenderer")
        if egl:
            p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        else:
            p.loadPlugin("eglRendererPlugin")

    pb.setGravity(0, 0, -10)
    pb.setPhysicsEngineParameter(
        fixedTimeStep=time_step,
        numSolverIterations=300,
        numSubSteps=1,
        contactBreakingThreshold=0.0005,
        erp=0.05,
        contactERP=0.05,
        # need to enable friction anchors (something to experiment with)
        frictionERP=0.2,
        solverResidualThreshold=1e-7,
        contactSlop=0.001,
        globalCFM=0.0001,
    )

    pb.loadURDF(
        add_assets_path("shared_assets/environment_objects/plane/plane.urdf"),
        [0, 0, -0.625],
    )
    pb.loadURDF(
        add_assets_path("shared_assets/environment_objects/table/table.urdf"),
        [0.50, 0.00, -0.625],
        [0.0, 0.0, 0.0, 1.0],
    )

    # add stimulus
    stim_pose *= POSE_UNITS
    stim_pos, stim_rpy = stim_pose[:3], stim_pose[3:] 
    p.loadURDF(
        stim_path,
        stim_pos,
        p.getQuaternionFromEuler(stim_rpy),
        useFixedBase=fix_stim,
        globalScaling=stim_scale
    )

    #  load goal indicator if goal exist
    if len(work_target_pose) > 0:
        base_taget_pose = inv_transform_pose(work_target_pose, workframe.copy()) 
        base_taget_pose*= POSE_UNITS
        traj_point_id = p.loadURDF(
                    add_assets_path("shared_assets/environment_objects/goal_indicators/sphere_indicator.urdf"),
                    base_taget_pose[:3],
                    [0, 0, 0, 1],
                    useFixedBase=True,
        )
        p.changeVisualShape(traj_point_id, -1, rgbaColor=[0, 1, 0, 0.5])
        p.setCollisionFilterGroupMask(traj_point_id, -1, 0, 0)
    
    if show_gui:
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.resetDebugVisualizerCamera(
            cam_params['dist'],
            cam_params['yaw'],
            cam_params['pitch'],
            cam_params['pos']
        )

    # set the workrame of the robot (relative to world frame)
    workframe *= POSE_UNITS
    workframe_pos, workframe_rpy = workframe[:3], workframe[3:] 


    # create the robot and sensor embodiment
    embodiment = RobotEmbodiment(
        pb,
        workframe_pos=workframe_pos,
        workframe_rpy=workframe_rpy,
        image_size=sensor_params["image_size"],
        arm_type="ur5",
        t_s_params=sensor_params,
        cam_params=cam_params,
        show_gui=show_gui,
        show_tactile=show_tactile,
        quick_mode=quick_mode
    )

    return embodiment
