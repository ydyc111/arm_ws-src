#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2D, BoundingBox2D


class ColorDetector(Node):
    def __init__(self):
        super().__init__('color_detector')
        self.subscription = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        self.publisher_ = self.create_publisher(Detection2D, '/detection', 10)
        self.bridge = CvBridge()

        # 回到最初能用的 HSV 参数
        self.lower_red1 = np.array([0, 100, 100])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 100, 100])
        self.upper_red2 = np.array([180, 255, 255])

        self.min_area = 300

        # 调试计数器
        self._fc = 0

        self.get_logger().info(
            f'Color Detector started. min_area={self.min_area}')

    def image_callback(self, msg):
        self._fc += 1

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.lower_red1, self.upper_red1) + \
               cv2.inRange(hsv, self.lower_red2, self.upper_red2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 每 30 帧保存原图和掩码
        if self._fc % 30 == 0:
            cv2.imwrite('/tmp/red_frame.png', frame)
            cv2.imwrite('/tmp/red_mask.png', mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # 每 30 帧打印一次诊断
        if self._fc % 30 == 0:
            red_pixels = int(cv2.countNonZero(mask))
            max_area = 0.0
            if contours:
                max_area = cv2.contourArea(
                    max(contours, key=cv2.contourArea))
            bgr_mean = frame.mean(axis=(0, 1))
            self.get_logger().info(
                f'[DBG] frame={self._fc} '
                f'red_pixels={red_pixels} '
                f'max_area={max_area:.0f} '
                f'BGR_mean=({bgr_mean[0]:.0f},{bgr_mean[1]:.0f},{bgr_mean[2]:.0f})')

        if not contours:
            return

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < self.min_area:
            return

        x, y, w, h = cv2.boundingRect(c)

        detection = Detection2D()
        detection.bbox = BoundingBox2D()
        detection.bbox.center.position.x = float(x + w / 2)
        detection.bbox.center.position.y = float(y + h / 2)
        detection.bbox.size_x = float(w)
        detection.bbox.size_y = float(h)

        self.publisher_.publish(detection)
        self.get_logger().info(
            f'Red detected at '
            f'({detection.bbox.center.position.x:.1f}, '
            f'{detection.bbox.center.position.y:.1f}) '
            f'size={w}x{h} area={area:.0f}')


def main(args=None):
    rclpy.init(args=args)
    node = ColorDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
