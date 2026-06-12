import os
import cv2
import time
import zmq
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

    
# 定义手部的 21 根骨骼连接关系
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # 大拇指
    (0, 5), (5, 6), (6, 7), (7, 8),         # 食指
    (5, 9), (9, 10), (10, 11), (11, 12),    # 中指
    (9, 13), (13, 14), (14, 15), (15, 16),  # 无名指
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # 小指及手掌边缘
]

pi = 3.14

def rad2deg(rad):
    return rad * 180 / pi

def vector_angle(v1, v2):
    """计算两个向量之间的夹角，返回弧度值"""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)  # 防止数值误差导致的越界
    angle = np.arccos(dot_product)
    return angle


def main():
    # 1. ZMQ 初始化
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5555")
    print("[Vision] ZMQ Publisher 启动，绑定端口: 5555")

    # 2. MediaPipe 任务初始化
    base_options = python.BaseOptions(model_asset_path=os.path.expanduser('../perception/model/gesture_recognizer.task'))
    options = vision.GestureRecognizerOptions(
        base_options=base_options, 
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1 
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)

    # 3. 摄像头配置
    cap = cv2.VideoCapture(0)

    num = 0

    # 4. 主循环：捕获视频帧，处理手部识别，绘制骨骼和关节点，计算关节角度并发送
    while True:
        num += 1
        success, frame = cap.read()
        if not success: continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)

        results = recognizer.recognize_for_video(mp_image, timestamp_ms)

        if results.hand_world_landmarks:
            # 1. 绘制骨骼和关节点
            for idx, hand_landmarks in enumerate(results.hand_landmarks):
                
                # 绘制骨骼连线 (Bones)
                for connection in HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    start_pt, end_pt = hand_landmarks[start_idx], hand_landmarks[end_idx]
                    
                    # 将归一化坐标(0~1)转换为像素坐标
                    start_x, start_y = int(start_pt.x * w), int(start_pt.y * h)
                    end_x, end_y = int(end_pt.x * w), int(end_pt.y * h)
                    
                    # 用白线连接两点，线宽设为 2
                    cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)

                # 绘制关节点 (Joints)
                for lm_idx, landmark in enumerate(hand_landmarks):
                    cx, cy = int(landmark.x * w), int(landmark.y * h)

                    # 画关节点
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

                    # 显示 landmark 编号
                    cv2.putText(
                        frame,
                        str(lm_idx),          # 0~20
                        (cx + 5, cy - 5),     # 文字位置，可自行调整
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),          # 绿色
                        1,
                        cv2.LINE_AA
                    )

            # 2. 计算关节角度
            wl = results.hand_world_landmarks[0]

            target_angle_indices = [0, 0, 0, 0, 
                                    0, 0, 0, 0, 
                                    0, 0, 0, 0, 
                                    0, 0, 0, 0] 

            def angle1(x):
                x_pt = np.array([wl[x].x, wl[x].y, wl[x].z])
                x1_pt = np.array([wl[x-1].x, wl[x-1].y, wl[x-1].z])
                x2_pt = np.array([wl[x+1].x, wl[x+1].y, wl[x+1].z])

                v1 = x1_pt - x_pt
                v2 = x2_pt - x_pt

                angle = vector_angle(v1, v2)
                angle = pi - abs(angle)

                return angle
            
            def angle2(x):
                if x == 17:
                    x_pt = np.array([wl[x].x, wl[x].y, wl[x].z])
                    x1_pt = np.array([wl[x-4].x, wl[x-4].y, wl[x-4].z])
                    x2_pt = np.array([wl[x+1].x, wl[x+1].y, wl[x+1].z])

                    v1 = x1_pt - x_pt
                    v2 = x2_pt - x_pt

                    angle = vector_angle(v1, v2)
                    angle = angle - pi / 2

                else:
                    x_pt = np.array([wl[x].x, wl[x].y, wl[x].z])
                    x1_pt = np.array([wl[x+4].x, wl[x+4].y, wl[x+4].z])
                    x2_pt = np.array([wl[x+1].x, wl[x+1].y, wl[x+1].z])

                    v1 = x1_pt - x_pt
                    v2 = x2_pt - x_pt

                    angle = vector_angle(v1, v2)
                    angle = angle - pi / 2
                return angle
            
            # 食指
            target_angle_indices[4] = angle2(5) + 0.1
            target_angle_indices[5] = angle1(6) - 0.15
            target_angle_indices[6] = angle1(7) - 0.5

            # 中指
            target_angle_indices[7] = angle2(9) - 0.15
            target_angle_indices[8] = angle1(10) - 0.15
            target_angle_indices[9] = angle1(11) - 0.5

            # 无名指
            target_angle_indices[10] = angle2(13) - 0.5
            target_angle_indices[11] = angle1(14) - 0.15
            target_angle_indices[12] = angle1(15) - 0.5

            # 小拇指
            target_angle_indices[13] = - ( angle2(17) + 0.6 )
            target_angle_indices[14] = angle1(18) - 0.15
            target_angle_indices[15] = angle1(19) - 0.5

            # 大拇指
            target_angle_indices[1] = pi / 2
            target_angle_indices[3] = - angle1(3)
            
            # 打包并通过 ZMQ 发送 (16个浮点数)
            msg = ",".join([f"{val:.5f}" for val in target_angle_indices])
            socket.send_string(msg)

        # 可视化
        cv2.imshow("Vsion", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 关闭
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()