import ctypes
import numpy as np
import cv2
import time
from MvCameraControl_class import *

# -------------------- 参数设置 --------------------
# 机械臂参考点（单位mm）
ARM_X = 9 * 25
ARM_Z = -27 * 25

# 相机参数
CAMERA_FOCAL_LENGTH = 6.0  # mm, 可根据标定或说明书设置
PIXEL_SIZE = 0.0024        # mm/pixel
BALL_ACTUAL_DIAMETER = 120.0  # mm
CAMERA_RES_WIDTH = 3072
CAMERA_RES_HEIGHT = 2048

# -------------------- 辅助函数 --------------------
def analyze_ball_frame(image, mask, contours, frame_width, frame_height):
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
    depth_mm = (BALL_ACTUAL_DIAMETER * CAMERA_FOCAL_LENGTH) / (ball_pixel_diameter * PIXEL_SIZE)
    
    # 3. 计算物理平面坐标（以图像中心为原点，单位：mm）
    cx_physical = (cx_pixel - frame_width/2) * depth_mm * PIXEL_SIZE / CAMERA_FOCAL_LENGTH
    cy_physical = (cy_pixel - frame_height/2) * depth_mm * PIXEL_SIZE / CAMERA_FOCAL_LENGTH
    
    # 计算相对机械臂坐标
    relative_x = depth_mm - ARM_Z
    relative_y = -(cx_physical - ARM_X)
    
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

# -------------------- 主函数 --------------------
if __name__ == "__main__":
    # 初始化相机
    cam = MvCamera()
    device_list = MV_CC_DEVICE_INFO_LIST()
    n_ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
    if n_ret != MV_OK or device_list.nDeviceNum == 0:
        raise RuntimeError("未找到相机设备！")

    device_info = device_list.pDeviceInfo[0].contents
    n_ret = cam.MV_CC_CreateHandle(device_info)
    if n_ret != MV_OK:
        raise RuntimeError(f"创建设备句柄失败: {n_ret:#x}")

    n_ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if n_ret != MV_OK:
        raise RuntimeError(f"打开设备失败: {n_ret:#x}")

    # -------------------- 相机模式设置 --------------------
    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", False)

    # 自动曝光 & 自动增益
    cam.MV_CC_SetEnumValue("ExposureAuto", 2)
    cam.MV_CC_SetFloatValue("ExposureTimeUpper", 100000.0)
    cam.MV_CC_SetEnumValue("GainAuto", 2)
    cam.MV_CC_SetFloatValue("GainUpperLimit", 18.0)

    # Gamma
    cam.MV_CC_SetFloatValue("Gamma", 1)

    # 像素格式
    cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_BayerRG8)

    # 自动白平衡
    cam.MV_CC_SetEnumValue("BalanceWhiteAuto", 2)

    # 锐化
    cam.MV_CC_SetBoolValue("SharpnessEnable", True)

    # 开始采集
    cam.MV_CC_StartGrabbing()
    
    # 初始化帧信息结构体
    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    ctypes.memset(ctypes.byref(stFrameInfo), 0, ctypes.sizeof(stFrameInfo))
    data_buf_size = 3072 * 2048 * 2
    pData = (ctypes.c_ubyte * data_buf_size)()
    
    # 创建显示窗口
    cv2.namedWindow("Camera Stream", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("Red Mask", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    
    print("[INFO] 按空格键开始分析红色小球，按 q 或 ESC 退出")
    
    relative_x = None
    relative_y = None
    frame_captured = False
    
    # 主循环
    while not frame_captured:
        # 获取一帧图像
        nRet = cam.MV_CC_GetOneFrameTimeout(pData, data_buf_size, stFrameInfo, 1000)
        if nRet != MV_OK:
            print(f"[ERROR] 获取帧失败: {nRet:#x}")
            continue
        
        # 转换图像数据
        image_array = np.frombuffer(pData, dtype=np.uint8, count=stFrameInfo.nFrameLen)
        
        if stFrameInfo.enPixelType == PixelType_Gvsp_BayerRG8:
            bayer_2d = image_array.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
            output_image = cv2.cvtColor(bayer_2d, cv2.COLOR_BAYER_RG2RGB)
        else:
            print("[ERROR] 不支持的像素格式")
            continue
        
        # 红色检测
        hsv_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 60, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 60, 50])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2) 

        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        cv2.imshow("Camera Stream", output_image)
        cv2.imshow("Red Mask", red_mask)
        
        # 检测按键
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32:  # 空格键
            if contours:
                relative_x, relative_y = analyze_ball_frame(
                    output_image.copy(),
                    red_mask.copy(),
                    contours,
                    stFrameInfo.nWidth,
                    stFrameInfo.nHeight
                )
                frame_captured = True
            else:
                print("[WARNING] 当前帧未检测到红色小球，请重新尝试")
        
        elif key == ord('q') or key == 27:  # 'q' 或 ESC
            print("[INFO] 用户退出")
            break
    
    # 清理资源
    cv2.destroyAllWindows()
    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()

    if relative_x is not None and relative_y is not None:
        print(f"[RESULT] 最终结果: relative_x = {relative_x:.2f}, relative_y = {relative_y:.2f}")
    else:
        print("[RESULT] 未获取到有效结果")
        
    