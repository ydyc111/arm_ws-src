#!/usr/bin/env python3
"""
视觉触发 + 关节空间抓取演示（带冷却时间）。
订阅 /detection，检测到红色物体后触发抓取动作，
抓取完成后冷却 N 秒，避免连续触发刷屏。
"""
import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from vision_msgs.msg import Detection2D

from pymoveit2 import MoveIt2


GROUP_NAME   = 'dofbot_arm'
BASE_LINK    = 'base_link'
END_EFFECTOR = 'arm_link5'
JOINT_NAMES  = ['arm_joint1', 'arm_joint2', 'arm_joint3',
                'arm_joint4', 'arm_joint5']

# ====== 抓取动作参数 ======
COOLDOWN_SEC = 5.0      # 抓取完成后冷却时间（秒）


def deg2rad(d):
    return d * math.pi / 180.0

# RViz 里拖出来的抓取姿态（度）
GRASP_JOINTS_DEG = [10.0, 25.0, 17.0, 30.0, 0.0]

# 接近姿态：抬起一点，先到这儿再到抓取位
APPROACH_JOINTS_DEG = [10.0, 12.0, 5.0, 18.0, 0.0]

GRASP_JOINTS    = [deg2rad(x) for x in GRASP_JOINTS_DEG]
APPROACH_JOINTS = [deg2rad(x) for x in APPROACH_JOINTS_DEG]


class GraspPlanner(Node):
    def __init__(self):
        super().__init__('grasp_planner')
        self.cb_group = ReentrantCallbackGroup()

        self.sub = self.create_subscription(
            Detection2D, '/detection', self.detection_callback, 10,
            callback_group=self.cb_group)

        self.moveit2 = MoveIt2(
            node=self,
            joint_names=JOINT_NAMES,
            base_link_name=BASE_LINK,
            end_effector_name=END_EFFECTOR,
            group_name=GROUP_NAME,
            callback_group=self.cb_group,
        )
        self.moveit2.max_velocity = 0.3
        self.moveit2.max_acceleration = 0.3

        self.lock = threading.Lock()
        self.busy = False
        self.last_grasp_time = 0.0     # 上次抓取的时刻（秒）

        self.get_logger().info(
            f'Grasp planner started. GROUP={GROUP_NAME} '
            f'TIP={END_EFFECTOR} COOLDOWN={COOLDOWN_SEC}s')
        self.get_logger().info(
            f'APPROACH joints (deg): {APPROACH_JOINTS_DEG}')
        self.get_logger().info(
            f'GRASP joints (deg):    {GRASP_JOINTS_DEG}')

    # ------------------------------------------------------------------
    def detection_callback(self, msg):
        now = time.time()
        with self.lock:
            # 1) 正在抓取 → 忽略
            if self.busy:
                return
            # 2) 还在冷却 → 忽略
            elapsed = now - self.last_grasp_time
            if elapsed < COOLDOWN_SEC:
                return
            # 3) 允许触发
            self.busy = True
            self.last_grasp_time = now

        cx = msg.bbox.center.position.x
        cy = msg.bbox.center.position.y
        self.get_logger().info(
            f'Red detected at ({cx:.1f},{cy:.1f}) '
            f'-> triggering grasp sequence')
        threading.Thread(target=self._grasp_sequence, daemon=True).start()

    # ------------------------------------------------------------------
    def _move_to(self, joints, label):
        self.get_logger().info(f'[MOVE] {label}')
        self.moveit2.move_to_configuration(joints)
        ok = self.moveit2.wait_until_executed()
        if not ok:
            self.get_logger().error(f'{label} failed')
        return ok

    # ------------------------------------------------------------------
    def _grasp_sequence(self):
        try:
            if not self._move_to(APPROACH_JOINTS, '1/3 approach pose'):
                return
            if not self._move_to(GRASP_JOINTS, '2/3 grasp pose'):
                return

            self.get_logger().info(
                '3/3 grasp reached (gripper not controlled)')
            self._move_to(APPROACH_JOINTS, 'lift up')
            self.get_logger().info(
                f'Grasp sequence finished. Cooldown {COOLDOWN_SEC}s.')
        except Exception as e:
            self.get_logger().error(f'Grasp sequence exception: {e}')
        finally:
            with self.lock:
                self.busy = False


# ----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = GraspPlanner()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
