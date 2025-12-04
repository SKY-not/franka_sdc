import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading

PARAMS_FILE = "camera_params.npz"

class CameraViewerGUI:
    def __init__(self, master):
        self.master = master
        master.title("Camera Figure GUI")

        # ----------------- 状态变量 -----------------
        self.cap = None
        self.running = False
        self.frame = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.roi = None
        self.map1 = None
        self.map2 = None

        # ----------------- 控件 -----------------
        ctrl_frame = ttk.Frame(master)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        ttk.Label(ctrl_frame, text="Camera ID:").grid(row=0, column=0)
        self.cam_id_var = tk.IntVar(value=0)
        self.cam_id_spin = ttk.Spinbox(ctrl_frame, from_=0, to=10, width=5, textvariable=self.cam_id_var)
        self.cam_id_spin.grid(row=0, column=1)

        self.open_btn = ttk.Button(ctrl_frame, text="Open Camera", command=self.open_camera)
        self.open_btn.grid(row=0, column=2, padx=4)

        self.close_btn = ttk.Button(ctrl_frame, text="Close Camera", command=self.close_camera, state=tk.DISABLED)
        self.close_btn.grid(row=0, column=3, padx=4)

        # ----------------- 视频显示 -----------------
        self.display_width = 800
        self.display_height = 600
        # 使用 Canvas 替代 Label，防止闪烁
        self.video_canvas = tk.Canvas(master, width=self.display_width, height=self.display_height, bg="black")
        self.video_canvas.pack(padx=6, pady=6)
        self.canvas_img = None  # 存储 Canvas image

        # ----------------- 加载标定数据 -----------------
        try:
            data = np.load(PARAMS_FILE)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            print("Loaded calibration data")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load calibration data: {e}")

    # ----------------- 打开摄像头 -----------------
    def open_camera(self):
        cam_id = int(self.cam_id_var.get())
        self.cap = cv2.VideoCapture(cam_id)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot open camera {cam_id}")
            return

        # 读取一帧，初始化 undistort map
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Cannot read from camera")
            return
        h, w = frame.shape[:2]
        new_cam_mtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
        self.roi = roi
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, new_cam_mtx, (w, h), cv2.CV_16SC2
        )

        self.running = True
        self.open_btn.config(state=tk.DISABLED)
        self.close_btn.config(state=tk.NORMAL)

        # 启动线程更新视频
        threading.Thread(target=self.update_loop, daemon=True).start()

    # ----------------- 关闭摄像头 -----------------
    def close_camera(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.open_btn.config(state=tk.NORMAL)
        self.close_btn.config(state=tk.DISABLED)

    # ----------------- 更新视频 -----------------
    def update_loop(self):
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue

            # 畸变矫正 + ROI 裁剪
            undistorted = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
            x, y, w, h = self.roi
            cropped = undistorted[y:y+h, x:x+w]

            # resize 到固定显示尺寸
            img = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.display_width, self.display_height))
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(img)

            # Canvas 更新图像，防止闪烁
            if self.canvas_img is None:
                self.canvas_img = self.video_canvas.create_image(0, 0, anchor="nw", image=imgtk)
            else:
                self.video_canvas.itemconfig(self.canvas_img, image=imgtk)

            # 保持引用，防止被垃圾回收
            self.video_canvas.imgtk = imgtk

            cv2.waitKey(10)

# ----------------- 主程序 -----------------
if __name__ == '__main__':
    root = tk.Tk()
    app = CameraViewerGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.close_camera(), root.destroy()))
    root.mainloop()