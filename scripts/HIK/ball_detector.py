import ctypes
import numpy as np
import cv2
import time
from MvCameraControl_class import *

class BallDetector:
    def __init__(self, arm_x=9*25, arm_z=-27*25, 
                 camera_focal_length=6.0, pixel_size=0.0024, 
                 ball_diameter=120.0):
        """
        初始化小球检测器
        
        参数:
        arm_x, arm_z: 机械臂参考点（单位mm）
        camera_focal_length: 相机焦距（mm）
        pixel_size: 像素尺寸（mm/pixel）
        ball_diameter: 小球实际直径（mm）
        """
        self.ARM_X = arm_x
        self.ARM_Z = arm_z
        self.CAMERA_FOCAL_LENGTH = camera_focal_length
        self.PIXEL_SIZE = pixel_size
        self.BALL_ACTUAL_DIAMETER = ball_diameter
        
        self.cam = None
        self.stFrameInfo = None
        self.pData = None
        self.data_buf_size = None
        
        # 相机参数
        self.CAMERA_RES_WIDTH = 3072
        self.CAMERA_RES_HEIGHT = 2048
        
        # 红色检测参数
        self.lower_red1 = np.array([0, 60, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([160, 60, 50])
        self.upper_red2 = np.array([180, 255, 255])
        
        self.relative_x = None
        self.relative_y = None
        
    def initialize_camera(self):
        """初始化相机"""
        try:
            self.cam = MvCamera()
            device_list = MV_CC_DEVICE_INFO_LIST()
            n_ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
            if n_ret != MV_OK or device_list.nDeviceNum == 0:
                raise RuntimeError("未找到相机设备！")

            device_info = device_list.pDeviceInfo[0].contents
            n_ret = self.cam.MV_CC_CreateHandle(device_info)
            if n_ret != MV_OK:
                raise RuntimeError(f"创建设备句柄失败: {n_ret:#x}")

            n_ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if n_ret != MV_OK:
                raise RuntimeError(f"打开设备失败: {n_ret:#x}")

            # -------------------- 相机模式设置 --------------------
            self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", False)

            # 自动曝光 & 自动增益
            self.cam.MV_CC_SetEnumValue("ExposureAuto", 2)
            self.cam.MV_CC_SetFloatValue("ExposureTimeUpper", 100000.0)
            self.cam.MV_CC_SetEnumValue("GainAuto", 2)
            self.cam.MV_CC_SetFloatValue("GainUpperLimit", 18.0)

            # Gamma
            self.cam.MV_CC_SetFloatValue("Gamma", 1)

            # 像素格式
            self.cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_BayerRG8)

            # 自动白平衡
            self.cam.MV_CC_SetEnumValue("BalanceWhiteAuto", 2)

            # 锐化
            self.cam.MV_CC_SetBoolValue("SharpnessEnable", True)

            # 开始采集
            self.cam.MV_CC_StartGrabbing()
            
            # 初始化帧信息结构体
            self.stFrameInfo = MV_FRAME_OUT_INFO_EX()
            ctypes.memset(ctypes.byref(self.stFrameInfo), 0, ctypes.sizeof(self.stFrameInfo))
            self.data_buf_size = 3072 * 2048 * 2
            self.pData = (ctypes.c_ubyte * self.data_buf_size)()
            
            print("[INFO] 相机初始化完成")
            return True
            
        except Exception as e:
            print(f"[ERROR] 相机初始化失败: {e}")
            return False
    
    def get_frame(self):
        """获取一帧图像，返回RGB图像和红色掩码以及轮廓"""
        if self.cam is None:
            print("[ERROR] 相机未初始化")
            return None, None, None
            
        # 获取一帧图像
        nRet = self.cam.MV_CC_GetOneFrameTimeout(self.pData, self.data_buf_size, self.stFrameInfo, 1000)
        if nRet != MV_OK:
            print(f"[ERROR] 获取帧失败: {nRet:#x}")
            return None, None, None
        
        # 转换图像数据
        image_array = np.frombuffer(self.pData, dtype=np.uint8, count=self.stFrameInfo.nFrameLen)
        
        if self.stFrameInfo.enPixelType == PixelType_Gvsp_BayerRG8:
            bayer_2d = image_array.reshape((self.stFrameInfo.nHeight, self.stFrameInfo.nWidth))
            output_image = cv2.cvtColor(bayer_2d, cv2.COLOR_BAYER_RG2RGB)
        else:
            print("[ERROR] 不支持的像素格式")
            return None, None, None
        
        # 红色检测
        hsv_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv_image, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv_image, self.lower_red2, self.upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2) 

        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return output_image, red_mask, contours
    
    def analyze_ball_frame(self, image, mask, contours):
        """分析截取帧，计算小球的x,y,z坐标（相机坐标系）"""
        if not contours:
            print("[ERROR] 截取帧中未检测到红色小球！")
            return None, None
        
        # 1. 找到最大轮廓（假设是红色小球）
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 2. 计算边界框和中心点（像素坐标）
        x, y, w, h = cv2.boundingRect(largest_contour)
        cx_pixel = x + w // 2
        cy_pixel = y + h // 2
        
        # 4. 计算小球在图像中的像素直径（使用平均直径）
        ball_pixel_diameter = (w + h) / 2.0
        
        if ball_pixel_diameter == 0:
            print("[ERROR] 小球像素直径为0，无法计算深度！")
            return None, None
        
        # 5. 使用相似三角形原理计算深度（Z坐标，单位：mm）
        depth_mm = (self.BALL_ACTUAL_DIAMETER * self.CAMERA_FOCAL_LENGTH) / (ball_pixel_diameter * self.PIXEL_SIZE)
        
        # 3. 计算物理平面坐标（以图像中心为原点，单位：mm）
        cx_physical = (cx_pixel - self.stFrameInfo.nWidth/2) * depth_mm * self.PIXEL_SIZE / self.CAMERA_FOCAL_LENGTH
        cy_physical = (cy_pixel - self.stFrameInfo.nHeight/2) * depth_mm * self.PIXEL_SIZE / self.CAMERA_FOCAL_LENGTH
        
        # 计算相对机械臂坐标
        relative_x = depth_mm - self.ARM_Z
        relative_y = -(cx_physical - self.ARM_X)
        
        # 输出结果
        print("="*50)
        print("【小球3D坐标分析结果】")
        print(f"1. 像素坐标 (X, Y)：({cx_pixel}, {cy_pixel})")
        print(f"边界框：w={w}, h={h}")
        print(f"2. 物理坐标 (X, Y)：({cx_physical:.2f} mm, {cy_physical:.2f} mm)")
        print(f"3. 小球像素直径：{ball_pixel_diameter:.2f} 像素")
        print(f"4. 深度（Z坐标）：{depth_mm:.2f} mm")
        print(f"5. 3D坐标 (X,Y,Z)：({cx_physical:.2f}, {cy_physical:.2f}, {depth_mm:.2f}) mm")
        print(f"相对机械臂坐标：X=({relative_x:.2f}, Y=({relative_y:.2f}))")
        print("="*50)
        
        # 在图像上标注结果
        cv2.putText(image, f"Center: ({cx_pixel}, {cy_pixel})", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(image, f"3D: ({cx_physical:.1f}, {cy_physical:.1f}, {depth_mm:.1f}) mm", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(image, f"Diameter: {ball_pixel_diameter:.1f} px", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # 绘制中心点和轮廓
        cv2.circle(image, (cx_pixel, cy_pixel), 5, (0, 255, 255), -1)  # 黄色中心点
        cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)  # 绿色轮廓
        
        # 保存结果图像
        cv2.imwrite("ball_analysis_result.png", image)
        cv2.imwrite("ball_mask.png", mask)
        print("✅ 分析结果已保存：ball_analysis_result.png / ball_mask.png")
        
        return relative_x, relative_y
    
    def get_ball_position(self, display=False):
        """
        获取小球的相对坐标
        
        参数:
        display: 是否显示图像窗口
        
        返回:
        tuple: (relative_x, relative_y) 或 (None, None) 如果失败
        """
        if display:
            cv2.namedWindow("Camera Stream", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.namedWindow("Red Mask", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        
        frame_captured = False
        relative_x = None
        relative_y = None
        
        while not frame_captured:
            # 获取一帧
            output_image, red_mask, contours = self.get_frame()
            if output_image is None:
                continue
            
            # 如果检测到红色小球，绘制边界框
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)

                cv2.rectangle(
                    output_image,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cx = x + w // 2
                cy = y + h // 2
                cv2.circle(output_image, (cx, cy), 6, (0, 0, 255), -1)
            
            # 显示图像
            if display:
                cv2.imshow("Camera Stream", output_image)
                cv2.imshow("Red Mask", red_mask)
            
            # 检测按键
            if display:
                key = cv2.waitKey(1) & 0xFF
                
                if key == 32:  # 空格键
                    if contours:
                        relative_x, relative_y = self.analyze_ball_frame(
                            output_image.copy(),
                            red_mask.copy(),
                            contours
                        )
                        frame_captured = True
                    else:
                        print("[WARNING] 当前帧未检测到红色小球，请重新尝试")
                
                elif key == ord('q') or key == 27:  # 'q' 或 ESC
                    print("[INFO] 用户退出")
                    break
            else:
                # 如果不显示窗口，直接分析第一帧
                if contours:
                    relative_x, relative_y = self.analyze_ball_frame(
                        output_image.copy(),
                        red_mask.copy(),
                        contours
                    )
                frame_captured = True
        
        if display:
            cv2.destroyAllWindows()
        
        return relative_x, relative_y
    
    def close(self):
        """关闭相机并释放资源"""
        if self.cam:
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            print("[INFO] 相机资源已释放")

# # -------------------- 主函数（测试用） --------------------
def main():
    """主函数，用于测试"""
    detector = BallDetector()
    
    try:
        # 初始化相机
        if not detector.initialize_camera():
            return
        
        # 获取小球位置（显示图像窗口）
        relative_x, relative_y = detector.get_ball_position(display=True)
        
        if relative_x is not None and relative_y is not None:
            print(f"[RESULT] 最终结果: relative_x = {relative_x:.2f}, relative_y = {relative_y:.2f}")
        else:
            print("[RESULT] 未获取到有效结果")
            
    except Exception as e:
        print(f"[ERROR] 程序运行出错: {e}")
        
    finally:
        # 确保资源被释放
        detector.close()

if __name__ == "__main__":
    main()