"""
完整的海康工业相机视觉引导系统
功能：相机控制、Bayer RG8转换、实时参数调整、红色目标检测、三维坐标计算
作者：基于海康机器人SDK和OpenCV
"""

import cv2
import numpy as np
import ctypes
import time
import sys
from MvCameraControl_class import *

# 定义一些SDK常量（如果MvCameraControl_class中未导入）
try:
    from MvCameraControl_class import (
        MV_OK, MV_TRIGGER_MODE_OFF, MV_ACCESS_Exclusive,
        MV_GIGE_DEVICE, MV_USB_DEVICE,
        PixelType_Gvsp_BayerRG8
    )
except ImportError:
    # 如果导入失败，定义常用常量
    MV_OK = 0x00000000
    MV_TRIGGER_MODE_OFF = 0
    MV_ACCESS_Exclusive = 0
    MV_GIGE_DEVICE = 0x00000001
    MV_USB_DEVICE = 0x00000004
    PixelType_Gvsp_BayerRG8 = 0x01080001


class RealTimeCameraProcessor:
    """实时相机处理器：完整的相机控制和视觉处理流程"""
    
    def __init__(self):
        self.cam = None
        self.is_grabbing = False
        
        # ========== 1. 相机硬件参数 (初始值) ==========
        self.cam_params = {
            'ExposureAuto': 2,          # 0=Off, 1=Once, 2=Continuous
            'ExposureTime': 20000.0,    # 微秒，手动曝光时有效
            'GainAuto': 2,              # 0=Off, 1=Once, 2=Continuous
            'Gain': 10.0,               # dB，手动增益时有效
            'Gamma': 2.2,               # Gamma值
            'AcquisitionFrameRate': 30.0, # 帧率
            'BalanceWhiteAuto': 2,      # 0=Off, 1=Once, 2=Continuous
        }
        
        # ========== 2. 视觉算法参数 (初始值) ==========
        # HSV红色阈值 (两个范围)
        self.hsv_lower1 = np.array([0, 100, 100])
        self.hsv_upper1 = np.array([10, 255, 255])
        self.hsv_lower2 = np.array([160, 100, 100])
        self.hsv_upper2 = np.array([180, 255, 255])
        
        # 形态学核大小 (用于去噪)
        self.morph_kernel_size = 5
        
        # ========== 3. 标定参数 (需要您后续替换!) ==========
        # 相机内参矩阵 [fx, 0, cx; 0, fy, cy; 0, 0, 1]
        self.camera_matrix = np.array([
            [2500.0, 0,     1536.0],  # 对于3072x2048分辨率，cx≈1536
            [0,     2500.0, 1024.0],  # cy≈1024
            [0,     0,     1.0]
        ])
        
        # 手眼标定矩阵 (相机到机械臂基座的变换)
        self.hand_eye_matrix = np.eye(4)  # 单位矩阵，需要替换
        
        # 已知目标物理尺寸 (单位：米)
        self.known_width = 0.05  # 假设红色框宽度为50cm
        
        # 状态变量
        self.last_params_hash = None
        self.frame_count = 0
        self.start_time = time.time()
        
    def setup_camera(self):
        """完整的相机初始化流程"""
        print("=" * 50)
        print("步骤1: 初始化海康工业相机")
        print("=" * 50)
        
        # 1.1 枚举设备
        device_list = MV_CC_DEVICE_INFO_LIST()
        n_ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
        
        if n_ret != MV_OK:
            print(f"[错误] 枚举设备失败！错误码: {n_ret:#x}")
            return False
        
        if device_list.nDeviceNum == 0:
            print("[错误] 未找到任何相机设备！")
            print("可能原因:")
            print("  1. 相机未通过USB连接或未上电")
            print("  2. 相机驱动未正确安装")
            print("  3. USB线可能有问题")
            return False
        
        print(f"[成功] 找到 {device_list.nDeviceNum} 个相机设备")
        
        # 1.2 获取第一个设备信息
        single_device_info = MV_CC_DEVICE_INFO()
        ctypes.memmove(
            ctypes.byref(single_device_info),
            device_list.pDeviceInfo[0],
            ctypes.sizeof(single_device_info)
        )
        
        # 显示设备信息
        if single_device_info.nTLayerType == MV_GIGE_DEVICE:
            dev_type = "GigE"
            name_bytes = bytes(single_device_info.SpecialInfo.stGigEInfo.chUserDefinedName).split(b'\x00')[0]
        elif single_device_info.nTLayerType == MV_USB_DEVICE:
            dev_type = "USB"
            name_bytes = bytes(single_device_info.SpecialInfo.stUsb3VInfo.chUserDefinedName).split(b'\x00')[0]
        else:
            dev_type = "Unknown"
            name_bytes = b""
        
        try:
            dev_name = name_bytes.decode('utf-8')
        except:
            dev_name = str(name_bytes)
        
        print(f"  设备类型: {dev_type}")
        print(f"  设备名称: {dev_name}")
        
        # 1.3 创建设备句柄
        self.cam = MvCamera()
        n_ret = self.cam.MV_CC_CreateHandle(single_device_info)
        if n_ret != MV_OK:
            print(f"[错误] 创建设备句柄失败！错误码: {n_ret:#x}")
            return False
        print("  设备句柄创建成功")
        
        # 1.4 打开设备
        n_ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 1000)
        if n_ret != MV_OK:
            print(f"[错误] 打开设备失败！错误码: {n_ret:#x}")
            print("可能原因:")
            print("  1. 设备已被其他程序占用（如MVS客户端）")
            print("  2. 权限不足，请尝试以管理员身份运行")
            self.cam.MV_CC_DestroyHandle()
            return False
        print("  设备打开成功")
        
        # 1.5 设置像素格式为Bayer RG8
        n_ret = self.cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_BayerRG8)
        if n_ret == MV_OK:
            print("  像素格式设置为: Bayer RG8")
        else:
            print(f"  [警告] 设置像素格式失败: {n_ret:#x}，将使用当前格式")
        
        # 1.6 应用相机参数
        print("\n[信息] 应用初始相机参数...")
        self._apply_all_camera_params()
        
        return True
    
    def _apply_all_camera_params(self):
        """应用所有相机参数到硬件"""
        if not self.cam:
            return
        
        try:
            # 设置触发模式为连续采集
            n_ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            if n_ret != MV_OK:
                print(f"  [警告] 设置触发模式失败: {n_ret:#x}")
            
            # 设置帧率
            n_ret = self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", self.cam_params['AcquisitionFrameRate'])
            
            # 设置曝光模式
            n_ret = self.cam.MV_CC_SetEnumValue("ExposureAuto", self.cam_params['ExposureAuto'])
            if self.cam_params['ExposureAuto'] == 0:  # 手动模式
                n_ret = self.cam.MV_CC_SetFloatValue("ExposureTime", self.cam_params['ExposureTime'])
                if n_ret == MV_OK:
                    print(f"  手动曝光时间: {self.cam_params['ExposureTime']:.0f} us")
            
            # 设置增益模式
            n_ret = self.cam.MV_CC_SetEnumValue("GainAuto", self.cam_params['GainAuto'])
            if self.cam_params['GainAuto'] == 0:  # 手动模式
                n_ret = self.cam.MV_CC_SetFloatValue("Gain", self.cam_params['Gain'])
                if n_ret == MV_OK:
                    print(f"  手动增益: {self.cam_params['Gain']:.1f} dB")
            
            # 设置Gamma
            n_ret = self.cam.MV_CC_SetFloatValue("Gamma", self.cam_params['Gamma'])
            if n_ret == MV_OK:
                print(f"  Gamma值: {self.cam_params['Gamma']:.1f}")
            
            # 设置白平衡
            n_ret = self.cam.MV_CC_SetEnumValue("BalanceWhiteAuto", self.cam_params['BalanceWhiteAuto'])
            
        except Exception as e:
            print(f"  应用相机参数时出错: {e}")
    
    def create_control_panel(self):
        """创建参数控制面板"""
        cv2.namedWindow('Control Panel', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Control Panel', 500, 700)
        
        print("\n[信息] 创建控制面板...")
        print("  相机参数和视觉参数均可实时调整")
        print("  调整后参数立即生效")
        
        # ===== 相机硬件参数调节条 =====
        cv2.createTrackbar('ExpAuto Mode', 'Control Panel', self.cam_params['ExposureAuto'], 2, lambda x: None)
        # 曝光时间滑动条 (对数尺度: 0-100对应约1000-100000微秒)
        exp_time_slider = int(np.log10(self.cam_params['ExposureTime']/1000) * 33.3)
        exp_time_slider = max(0, min(100, exp_time_slider))
        cv2.createTrackbar('ExpTime(us)', 'Control Panel', exp_time_slider, 100, lambda x: None)
        
        cv2.createTrackbar('GainAuto Mode', 'Control Panel', self.cam_params['GainAuto'], 2, lambda x: None)
        cv2.createTrackbar('Gain(dB)', 'Control Panel', int(self.cam_params['Gain']), 24, lambda x: None)
        cv2.createTrackbar('Gamma(x10)', 'Control Panel', int(self.cam_params['Gamma']*10), 40, lambda x: None)
        cv2.createTrackbar('FrameRate', 'Control Panel', int(self.cam_params['AcquisitionFrameRate']), 120, lambda x: None)
        cv2.createTrackbar('WB Auto', 'Control Panel', self.cam_params['BalanceWhiteAuto'], 2, lambda x: None)
        
        # ===== 视觉算法参数调节条 =====
        cv2.createTrackbar('H1 Low', 'Control Panel', self.hsv_lower1[0], 179, lambda x: None)
        cv2.createTrackbar('H1 High', 'Control Panel', self.hsv_upper1[0], 179, lambda x: None)
        cv2.createTrackbar('H2 Low', 'Control Panel', self.hsv_lower2[0], 179, lambda x: None)
        cv2.createTrackbar('H2 High', 'Control Panel', self.hsv_upper2[0], 179, lambda x: None)
        cv2.createTrackbar('S Low', 'Control Panel', self.hsv_lower1[1], 255, lambda x: None)
        cv2.createTrackbar('S High', 'Control Panel', self.hsv_upper1[1], 255, lambda x: None)
        cv2.createTrackbar('V Low', 'Control Panel', self.hsv_lower1[2], 255, lambda x: None)
        cv2.createTrackbar('V High', 'Control Panel', self.hsv_upper1[2], 255, lambda x: None)
        cv2.createTrackbar('Morph Kernel', 'Control Panel', self.morph_kernel_size, 15, lambda x: None)
        
        # 控制面板显示图像
        self.control_panel_img = np.zeros((700, 500, 3), dtype=np.uint8)
        
        print("  控制面板创建完成")
    
    def _read_trackbar_params(self):
        """从控制面板读取所有参数"""
        # 读取相机参数
        new_exp_auto = cv2.getTrackbarPos('ExpAuto Mode', 'Control Panel')
        exp_slider = cv2.getTrackbarPos('ExpTime(us)', 'Control Panel')
        new_exp_time = 1000.0 * (10 ** (exp_slider / 33.3))  # 对数映射
        
        new_gain_auto = cv2.getTrackbarPos('GainAuto Mode', 'Control Panel')
        new_gain = float(cv2.getTrackbarPos('Gain(dB)', 'Control Panel'))
        new_gamma = cv2.getTrackbarPos('Gamma(x10)', 'Control Panel') / 10.0
        new_framerate = float(cv2.getTrackbarPos('FrameRate', 'Control Panel'))
        new_wb_auto = cv2.getTrackbarPos('WB Auto', 'Control Panel')
        
        # 读取视觉参数
        new_h1_low = cv2.getTrackbarPos('H1 Low', 'Control Panel')
        new_h1_high = cv2.getTrackbarPos('H1 High', 'Control Panel')
        new_h2_low = cv2.getTrackbarPos('H2 Low', 'Control Panel')
        new_h2_high = cv2.getTrackbarPos('H2 High', 'Control Panel')
        new_s_low = cv2.getTrackbarPos('S Low', 'Control Panel')
        new_s_high = cv2.getTrackbarPos('S High', 'Control Panel')
        new_v_low = cv2.getTrackbarPos('V Low', 'Control Panel')
        new_v_high = cv2.getTrackbarPos('V High', 'Control Panel')
        new_kernel = cv2.getTrackbarPos('Morph Kernel', 'Control Panel')
        
        # 确保核大小为奇数
        new_kernel = max(1, new_kernel if new_kernel % 2 == 1 else new_kernel + 1)
        
        return {
            'cam_params': {
                'ExposureAuto': new_exp_auto,
                'ExposureTime': new_exp_time,
                'GainAuto': new_gain_auto,
                'Gain': new_gain,
                'Gamma': new_gamma,
                'AcquisitionFrameRate': new_framerate,
                'BalanceWhiteAuto': new_wb_auto,
            },
            'hsv_params': {
                'lower1': np.array([new_h1_low, new_s_low, new_v_low]),
                'upper1': np.array([new_h1_high, new_s_high, new_v_high]),
                'lower2': np.array([new_h2_low, new_s_low, new_v_low]),
                'upper2': np.array([new_h2_high, new_s_high, new_v_high]),
            },
            'morph_kernel': new_kernel
        }
    
    def _params_changed(self, new_params):
        """检查参数是否发生变化"""
        # 检查相机参数
        for key, value in new_params['cam_params'].items():
            if key == 'ExposureTime':
                if abs(value - self.cam_params[key]) > 100:
                    return True
            elif key == 'Gain':
                if abs(value - self.cam_params[key]) > 0.1:
                    return True
            elif key == 'Gamma':
                if abs(value - self.cam_params[key]) > 0.05:
                    return True
            elif value != self.cam_params[key]:
                return True
        
        # 检查视觉参数
        if (not np.array_equal(new_params['hsv_params']['lower1'], self.hsv_lower1) or
            not np.array_equal(new_params['hsv_params']['upper1'], self.hsv_upper1) or
            not np.array_equal(new_params['hsv_params']['lower2'], self.hsv_lower2) or
            not np.array_equal(new_params['hsv_params']['upper2'], self.hsv_upper2) or
            new_params['morph_kernel'] != self.morph_kernel_size):
            return True
        
        return False
    
    def update_params_from_panel(self):
        """从控制面板更新参数"""
        new_params = self._read_trackbar_params()
        
        if self._params_changed(new_params):
            # 更新相机参数
            self.cam_params.update(new_params['cam_params'])
            self._apply_all_camera_params()
            
            # 更新视觉参数
            self.hsv_lower1 = new_params['hsv_params']['lower1']
            self.hsv_upper1 = new_params['hsv_params']['upper1']
            self.hsv_lower2 = new_params['hsv_params']['lower2']
            self.hsv_upper2 = new_params['hsv_params']['upper2']
            self.morph_kernel_size = new_params['morph_kernel']
            
            return True
        
        return False
    
    def update_control_panel_display(self, coords=None):
        """更新控制面板显示"""
        self.control_panel_img[:] = 0  # 清空画布
        
        # 相机参数显示
        y_offset = 30
        cam_params_text = [
            "=== 相机参数 ===",
            f"曝光模式: {self.cam_params['ExposureAuto']} (0=手动,1=单次,2=连续)",
            f"曝光时间: {self.cam_params['ExposureTime']:.0f} us",
            f"增益模式: {self.cam_params['GainAuto']} (0=手动,1=单次,2=连续)",
            f"增益值: {self.cam_params['Gain']:.1f} dB",
            f"Gamma值: {self.cam_params['Gamma']:.1f}",
            f"帧率: {self.cam_params['AcquisitionFrameRate']:.0f} fps",
            f"白平衡: {self.cam_params['BalanceWhiteAuto']} (0=关,1=单次,2=连续)",
            "",
            "=== 视觉参数 ===",
            f"H1范围: [{self.hsv_lower1[0]}, {self.hsv_upper1[0]}]",
            f"H2范围: [{self.hsv_lower2[0]}, {self.hsv_upper2[0]}]",
            f"S范围: [{self.hsv_lower1[1]}, {self.hsv_upper1[1]}]",
            f"V范围: [{self.hsv_lower1[2]}, {self.hsv_upper1[2]}]",
            f"形态学核: {self.morph_kernel_size}",
        ]
        
        # 添加坐标信息（如果检测到目标）
        if coords is not None:
            cam_params_text.append("")
            cam_params_text.append("=== 目标坐标 ===")
            cam_params_text.append(f"相机坐标系:")
            cam_params_text.append(f"  X: {coords['camera'][0]:.3f} m")
            cam_params_text.append(f"  Y: {coords['camera'][1]:.3f} m")
            cam_params_text.append(f"  Z: {coords['camera'][2]:.3f} m")
            cam_params_text.append(f"机械臂坐标系:")
            cam_params_text.append(f"  X: {coords['robot'][0]:.3f} m")
            cam_params_text.append(f"  Y: {coords['robot'][1]:.3f} m")
            cam_params_text.append(f"  Z: {coords['robot'][2]:.3f} m")
        
        # 绘制文本
        for i, text in enumerate(cam_params_text):
            color = (255, 255, 255)  # 白色
            if "===" in text:
                color = (0, 255, 255)  # 黄色标题
            elif "坐标" in text:
                color = (0, 255, 0)  # 绿色坐标
            
            cv2.putText(self.control_panel_img, text, (10, y_offset + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # 添加使用说明
        instructions = [
            "",
            "=== 使用说明 ===",
            "1. 先调整相机参数获得清晰图像",
            "2. 再调整视觉参数提取红色目标",
            "3. 按 'S' 保存快照",
            "4. 按 'R' 重置参数",
            "5. 按 'Q' 退出程序",
            "",
            "提示: 观察右侧掩膜视图",
            "目标区域应为白色，背景为黑色"
        ]
        
        for i, text in enumerate(instructions):
            cv2.putText(self.control_panel_img, text, (10, 500 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow('Control Panel', self.control_panel_img)
    
    def process_frame(self, bgr_frame):
        """处理单帧图像：红色目标检测与三维坐标计算"""
        # 转换为HSV颜色空间
        hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        
        # 创建红色掩膜（两个范围）
        mask1 = cv2.inRange(hsv_frame, self.hsv_lower1, self.hsv_upper1)
        mask2 = cv2.inRange(hsv_frame, self.hsv_lower2, self.hsv_upper2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # 形态学处理（去噪）
        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)  # 闭运算：填充空洞
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)   # 开运算：去除噪声
        
        # 查找轮廓
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 准备结果图像
        result_frame = bgr_frame.copy()
        coord_info = None
        
        if contours:
            # 找到最大轮廓（假设是目标）
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # 面积过滤（忽略小噪声）
            if area > 100:
                # 获取外接矩形
                x, y, w, h = cv2.boundingRect(largest_contour)
                center_x, center_y = x + w//2, y + h//2
                
                # 在结果图像上绘制
                cv2.drawContours(result_frame, [largest_contour], -1, (0, 255, 0), 2)  # 绿色轮廓
                cv2.rectangle(result_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # 蓝色外接框
                cv2.circle(result_frame, (center_x, center_y), 8, (0, 0, 255), -1)  # 红色中心点
                
                # 计算三维坐标（仅当宽度有效时）
                if w > 10:
                    # 从相机内参矩阵获取参数
                    fx = self.camera_matrix[0, 0]
                    fy = self.camera_matrix[1, 1]
                    cx = self.camera_matrix[0, 2]
                    cy = self.camera_matrix[1, 2]
                    
                    # 基于相似三角形原理估算深度
                    Z = (self.known_width * fx) / w
                    
                    # 计算相机坐标系下的X, Y坐标
                    X = (center_x - cx) * Z / fx
                    Y = (center_y - cy) * Z / fy
                    
                    # 转换到机械臂坐标系
                    point_camera = np.array([X, Y, Z, 1.0])
                    point_robot = self.hand_eye_matrix @ point_camera.T
                    
                    # 保存坐标信息
                    coord_info = {
                        'pixel': (center_x, center_y),
                        'camera': (X, Y, Z),
                        'robot': point_robot[:3].tolist(),
                        'width_pixel': w,
                        'height_pixel': h,
                        'area_pixel': area
                    }
                    
                    # 在图像上显示坐标
                    coord_text = f"Robot: ({point_robot[0]:.3f}, {point_robot[1]:.3f}, {point_robot[2]:.3f}) m"
                    cv2.putText(result_frame, coord_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # 显示像素信息
                    pixel_text = f"Pixel: ({center_x}, {center_y}), Size: {w}x{h}"
                    cv2.putText(result_frame, pixel_text, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return result_frame, red_mask, coord_info
    
    def _grab_single_frame(self):
        """获取单帧图像（Bayer格式转换）"""
        if not self.cam:
            return None
        
        # 准备帧信息结构体
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(ctypes.byref(stFrameInfo), 0, ctypes.sizeof(stFrameInfo))
        
        # 创建缓冲区（足够大以容纳最大分辨率图像）
        data_buf_size = 3072 * 2048 * 2  # 宽*高*2（考虑16位情况）
        pData = (ctypes.c_ubyte * data_buf_size)()
        
        # 获取一帧图像（100ms超时）
        nRet = self.cam.MV_CC_GetOneFrameTimeout(pData, data_buf_size, stFrameInfo, 100)
        if nRet != MV_OK:
            return None
        
        # Bayer RG8 转 BGR
        image_array = np.frombuffer(pData, dtype=np.uint8, count=stFrameInfo.nFrameLen)
        
        # 检查像素格式
        if stFrameInfo.enPixelType == PixelType_Gvsp_BayerRG8:
            # 重塑为2D Bayer图像
            bayer_2d = image_array.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
            # 去马赛克，转换为BGR
            bgr_frame = cv2.cvtColor(bayer_2d, cv2.COLOR_BayerRG2BGR)
            return bgr_frame
        else:
            print(f"[警告] 非预期的像素格式: {stFrameInfo.enPixelType:#x}")
            return None
    
    def _reset_to_default_params(self):
        """重置所有参数为默认值"""
        print("\n[信息] 重置所有参数为默认值")
        
        # 重置相机参数
        self.cam_params = {
            'ExposureAuto': 2,
            'ExposureTime': 20000.0,
            'GainAuto': 2,
            'Gain': 10.0,
            'Gamma': 2.2,
            'AcquisitionFrameRate': 30.0,
            'BalanceWhiteAuto': 2,
        }
        
        # 重置视觉参数
        self.hsv_lower1 = np.array([0, 100, 100])
        self.hsv_upper1 = np.array([10, 255, 255])
        self.hsv_lower2 = np.array([160, 100, 100])
        self.hsv_upper2 = np.array([180, 255, 255])
        self.morph_kernel_size = 5
        
        # 更新控制面板滑动条
        cv2.setTrackbarPos('ExpAuto Mode', 'Control Panel', 2)
        cv2.setTrackbarPos('ExpTime(us)', 'Control Panel', 30)
        cv2.setTrackbarPos('GainAuto Mode', 'Control Panel', 2)
        cv2.setTrackbarPos('Gain(dB)', 'Control Panel', 10)
        cv2.setTrackbarPos('Gamma(x10)', 'Control Panel', 22)
        cv2.setTrackbarPos('FrameRate', 'Control Panel', 30)
        cv2.setTrackbarPos('WB Auto', 'Control Panel', 2)
        
        cv2.setTrackbarPos('H1 Low', 'Control Panel', 0)
        cv2.setTrackbarPos('H1 High', 'Control Panel', 10)
        cv2.setTrackbarPos('H2 Low', 'Control Panel', 160)
        cv2.setTrackbarPos('H2 High', 'Control Panel', 180)
        cv2.setTrackbarPos('S Low', 'Control Panel', 100)
        cv2.setTrackbarPos('S High', 'Control Panel', 255)
        cv2.setTrackbarPos('V Low', 'Control Panel', 100)
        cv2.setTrackbarPos('V High', 'Control Panel', 255)
        cv2.setTrackbarPos('Morph Kernel', 'Control Panel', 5)
        
        # 应用相机参数
        self._apply_all_camera_params()
    
    def run(self):
        """主运行循环"""
        print("\n" + "="*50)
        print("海康工业相机视觉引导系统")
        print("="*50)
        
        # 1. 初始化相机
        if not self.setup_camera():
            print("[错误] 相机初始化失败，程序退出")
            return
        
        # 2. 开始采集
        n_ret = self.cam.MV_CC_StartGrabbing()
        if n_ret != MV_OK:
            print(f"[错误] 开始采集失败: {n_ret:#x}")
            self.cleanup()
            return
        
        self.is_grabbing = True
        print("\n[成功] 开始连续采集")
        
        # 3. 创建控制面板和显示窗口
        self.create_control_panel()
        cv2.namedWindow('Camera View - 按Q退出', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Mask View - 调试用', cv2.WINDOW_NORMAL)
        
        cv2.resizeWindow('Camera View - 按Q退出', 800, 600)
        cv2.resizeWindow('Mask View - 调试用', 800, 600)
        
        print("\n[提示] 调试流程:")
        print("  1. 先调整相机参数（曝光、增益）获得清晰图像")
        print("  2. 再调整视觉参数（HSV范围）精确提取红色目标")
        print("  3. 观察右侧掩膜视图，目标应为白色，背景为黑色")
        print("\n[快捷键] Q:退出 | S:保存快照 | R:重置参数")
        
        # 4. 主循环
        fps_counter = 0
        last_fps_time = time.time()
        last_coords = None
        
        try:
            while self.is_grabbing:
                # 4.1 更新参数
                params_updated = self.update_params_from_panel()
                
                # 4.2 获取并处理图像
                bgr_frame = self._grab_single_frame()
                if bgr_frame is None:
                    continue
                
                # 4.3 目标检测与坐标计算
                result_frame, red_mask, coords = self.process_frame(bgr_frame)
                
                # 4.4 更新FPS显示
                fps_counter += 1
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    fps = fps_counter / (current_time - last_fps_time)
                    fps_counter = 0
                    last_fps_time = current_time
                    
                    fps_text = f"FPS: {fps:.1f}"
                    cv2.putText(result_frame, fps_text, (bgr_frame.shape[1]-150, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 4.5 显示图像
                cv2.imshow('Camera View - Enter Q to quit', result_frame)
                cv2.imshow('Mask View - The avaible image', red_mask)
                
                # 4.6 更新控制面板显示
                self.update_control_panel_display(coords if coords else last_coords)
                if coords:
                    last_coords = coords
                
                # 4.7 键盘控制
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q或ESC
                    print("\n[信息] 用户请求退出")
                    break
                elif key == ord('s'):
                    # 保存快照
                    timestamp = int(time.time())
                    cv2.imwrite(f'camera_snapshot_{timestamp}.jpg', result_frame)
                    cv2.imwrite(f'mask_snapshot_{timestamp}.jpg', red_mask)
                    print(f"[信息] 快照已保存: camera_snapshot_{timestamp}.jpg")
                elif key == ord('r'):
                    # 重置参数
                    self._reset_to_default_params()
        
        except KeyboardInterrupt:
            print("\n[信息] 程序被用户中断")
        except Exception as e:
            print(f"\n[错误] 程序运行异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("\n[信息] 正在清理资源...")
        
        if self.cam:
            if self.is_grabbing:
                self.cam.MV_CC_StopGrabbing()
                print("  采集已停止")
                self.is_grabbing = False
            
            self.cam.MV_CC_CloseDevice()
            print("  设备已关闭")
            
            self.cam.MV_CC_DestroyHandle()
            print("  句柄已销毁")
        
        cv2.destroyAllWindows()
        print("[信息] 资源清理完成，程序退出")


def main():
    """主函数"""
    print("="*60)
    print("海康工业相机完整视觉引导系统")
    print("="*60)
    print("系统要求:")
    print("  1. 海康机器人工业相机 (如MV-CU060-10UC)")
    print("  2. 已安装MVS软件和Python SDK")
    print("  3. Python环境: OpenCV, NumPy")
    print("="*60)
    
    # 创建处理器实例
    processor = RealTimeCameraProcessor()
    
    # 重要提示
    print("\n[重要提示] 当前使用示例标定参数!")
    print("  1. camera_matrix: 需要替换为您的相机标定结果")
    print("  2. hand_eye_matrix: 需要替换为您的手眼标定结果")
    print("  3. known_width: 请根据实际目标尺寸修改")
    print("\n标定前计算出的坐标仅供参考，不能用于精确抓取!")
    
    input("\n按Enter键开始运行，按Ctrl+C终止...")
    
    # 运行主程序
    processor.run()


if __name__ == "__main__":
    # 检查必要的库
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        print(f"[错误] 缺少必要的Python库: {e}")
        print("请安装: pip install opencv-python numpy")
        sys.exit(1)
    
    # 检查海康SDK
    try:
        from MvCameraControl_class import *
    except ImportError as e:
        print(f"[错误] 无法导入海康SDK: {e}")
        print("请确保:")
        print("  1. 已安装海康MVS软件")
        print("  2. MvImport目录在Python路径中")
        print("  3. 或尝试: sys.path.append(r'C:\\Program Files (x86)\\MVS\\Development\\Samples\\Python')")
        sys.exit(1)
    
    # 运行主程序
    main()