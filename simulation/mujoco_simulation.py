import os
import mujoco
import mujoco.viewer
import zmq
import numpy as np
import time

def print_model_joints_and_actuators(model):
    """辅助调试工具：打印 MuJoCo 场景中所有关节和电机的名字"""
    print("\n--- 场景中的所有关节名字 ---")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"Joint ID {i}: {name}")
    print("\n--- 场景中的所有电机名字 ---")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"Actuator ID {i}: {name}")
    print("---------------------------\n")

def main():
    # 1. ZMQ 订阅端初始化
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.setsockopt(zmq.CONFLATE, 1) # 抛弃过期帧，保持最实时
    socket.connect("tcp://localhost:5555")
    print("[MuJoCo] ZMQ 订阅端启动...")

    # 2. 加载场景包装器 XML
    model = mujoco.MjModel.from_xml_path(os.path.expanduser("../assets/xml/scene.xml"))
    data = mujoco.MjData(model)

    # 调试：打印所有发现的关节与电机
    print_model_joints_and_actuators(model)

    # 3. 遍历寻找关节与对应的电机 ID
    real_joint_indices = []
    real_actuator_indices = []

    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if joint_name:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            
            actuator_name = f"act_{joint_name}"
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            
            if joint_id != -1 and act_id != -1:
                # 添加到数组中
                real_joint_indices.append(joint_id)
                real_actuator_indices.append(act_id)
                print(f"[Mapper] 映射匹配成功: 关节 '{joint_name}' -> 驱动 '{actuator_name}'")


    # 4. 主仿真与渲染循环
    physics_steps_per_render = 10  # 物理执行 10 次，步长 0.002s，刚好刷新一次渲染 0.02s
    time_step = model.opt.timestep

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # 接收目标角度并传给电机
            try:
                msg = socket.recv_string(flags=zmq.NOBLOCK)
                angles = [float(x) for x in msg.split(',')]
                if len(angles) == 16:
                    for idx, act_id in enumerate(real_actuator_indices):
                        target_angle = angles[idx]
                        data.ctrl[act_id] = target_angle
                    print(f"[MuJoCo] 接收到目标角度: {angles}")
                mujoco.mj_forward(model, data) 
            except zmq.Again:
                pass


            # 物理步进
            for _ in range(physics_steps_per_render):
                mujoco.mj_step(model, data)

            # 同步画面
            viewer.sync()

            # 保持 50Hz 刷新率
            time_until_next_render = time_step * physics_steps_per_render - (time.time() - step_start)
            if time_until_next_render > 0:
                time.sleep(time_until_next_render)
            else:
                time.sleep

if __name__ == '__main__':
    main()