import cv2
import time
import math
import torch
import threading
from queue import Queue
from ultralytics import YOLO

# ================= 显卡检测与强制开启 =================
if torch.cuda.is_available():
    compute_device = 0
    print(f"✅ 成功检测到显卡: {torch.cuda.get_device_name(0)}")
else:
    compute_device = 'cpu'
    print("❌ 警告：未检测到可用的 GPU，使用 CPU 运行...")
# ======================================================

# ================= 配置区 =================
camera_id = 0
desired_width = 2560
desired_height = 1440
infer_size = 640
preview_width = 1280
preview_height = 720


# ==========================================

# ================= 异步摄像头读取类 =================
class VideoCaptureAsync:
    def __init__(self, src, width, height):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if self.fps < 1: self.fps = 30

        self.q = Queue(maxsize=2)  # 只保留最新的 2 帧，防止堆积导致延迟
        self.stopped = False

    def start(self):
        # 启动一个后台线程专门负责读取画面
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                return

            # 如果队列满了，踢掉最老的一帧，保证永远处理最新的画面
            if self.q.full():
                self.q.get()
            self.q.put(frame)

    def read(self):
        if self.q.empty():
            return False, None
        return True, self.q.get()

    def stop(self):
        self.stopped = True
        self.cap.release()


# ====================================================

# 加载模型
model = YOLO("yolo11n-pose.pt")

print("正在初始化异步摄像头流...")
stream = VideoCaptureAsync(camera_id, desired_width, desired_height).start()

# 给摄像头一点时间热身
time.sleep(2.0)

if stream.stopped:
    print(f"无法打开摄像头 {camera_id}，请检查设备连接！")
    exit()

print(f"实际采集分辨率: {stream.actual_width} x {stream.actual_height} @ {stream.fps} FPS")


def process_frame(frame, model):
    # 核心优化1：增加 half=True 开启 FP16 半精度运算，提升 GPU 吞吐量，降低显存占用
    use_fp16 = True if compute_device != 'cpu' else False
    results = model.track(frame, persist=True, classes=[0], imgsz=infer_size, verbose=False, device=compute_device, half=use_fp16)

    # 增加安全检查：确保有检测结果，并且分配了 track ID
    if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # 确保关键点也检测到了
        if results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy.cpu().numpy()
            confs = results[0].keypoints.conf.cpu().numpy()
            box_confs = results[0].boxes.conf.cpu().numpy()

            for box, track_id, kp, kp_conf, b_conf in zip(boxes, track_ids, keypoints, confs, box_confs):

                # === 修复问题2：过滤低置信度的错误检测（如椅子） ===
                if b_conf < 0.55:
                    continue
                # 必须能清晰看到头部，否则不判定（防止把椅背当成人）
                if kp_conf[0] < 0.4:
                    continue

                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                box_w = x2 - x1
                box_h = y2 - y1

                # === 提取核心关键点 ===
                head_x, head_y = kp[0][0], kp[0][1]

                # 肩膀
                shoulder_c_x = (kp[5][0] + kp[6][0]) / 2
                shoulder_c_y = (kp[5][1] + kp[6][1]) / 2
                # 计算肩宽 (防止除零)
                shoulder_width = math.hypot(kp[5][0] - kp[6][0], kp[5][1] - kp[6][1])
                shoulder_width = max(shoulder_width, 10.0)

                # 胯部 (如果下半身完全被遮挡/没检测到胯部，则无法做精准判定)
                hip_l_conf, hip_r_conf = kp_conf[11], kp_conf[12]
                if hip_l_conf < 0.2 and hip_r_conf < 0.2:
                    continue  # 看不到胯部不强行判定

                hip_c_x = (kp[11][0] + kp[12][0]) / 2
                hip_c_y = (kp[11][1] + kp[12][1]) / 2

                # 脚踝
                ankle_l_conf, ankle_r_conf = kp_conf[15], kp_conf[16]
                ankle_visible = ankle_l_conf > 0.3 or ankle_r_conf > 0.3
                if ankle_visible:
                    if ankle_l_conf > 0.3 and ankle_r_conf > 0.3:
                        ankle_c_x = (kp[15][0] + kp[16][0]) / 2
                        ankle_c_y = (kp[15][1] + kp[16][1]) / 2
                    else:
                        ankle_c_x = kp[15][0] if ankle_l_conf > 0.3 else kp[16][0]
                        ankle_c_y = kp[15][1] if ankle_l_conf > 0.3 else kp[16][1]

                # === 核心几何学计算 ===
                # 计算“脊柱向量” (头到胯部的距离) -> 彻底解决手臂伸开的干扰！
                spine_dx = head_x - hip_c_x
                spine_dy = head_y - hip_c_y
                spine_length = math.hypot(spine_dx, spine_dy)

                is_fall = False

                # === 规则 1：横向摔倒 (侧卧) ===
                if abs(spine_dx) > abs(spine_dy) * 1.2:
                    if ankle_visible:
                        leg_dx = hip_c_x - ankle_c_x
                        leg_dy = hip_c_y - ankle_c_y
                        if abs(leg_dx) > abs(leg_dy) * 0.8 or abs(leg_dy) < shoulder_width:
                            is_fall = True
                    else:
                        is_fall = True

                # === 规则 2：倒挂 / 严重前倾扑倒 (头到了胯的下方) ===
                elif head_y > hip_c_y:
                    if ankle_visible:
                        if (ankle_c_y - hip_c_y) < shoulder_width * 1.5:
                            is_fall = True
                    else:
                        if head_y > hip_c_y + shoulder_width * 0.5:
                            is_fall = True

                # === 规则 3：修复问题3（高空俯视/屋檐视角的垂直摔倒） ===
                else:
                    box_aspect = box_h / box_w if box_w > 0 else 1
                    if spine_length > shoulder_width * 2.5 and box_aspect < 2.0:
                        is_fall = True
                    if ankle_visible:
                        total_dy = ankle_c_y - head_y
                        if total_dy < shoulder_width * 1.8 and abs(spine_dx) > shoulder_width * 0.8:
                            is_fall = True

                # === 画面可视化 ===
                color = (0, 0, 255) if is_fall else (0, 255, 0)
                text = f"ID:{track_id} FALLEN" if is_fall else f"ID:{track_id} Normal"

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # 画出核心骨架点
                cv2.circle(frame, (int(head_x), int(head_y)), 5, (255, 0, 0), -1)
                cv2.circle(frame, (int(shoulder_c_x), int(shoulder_c_y)), 5, (255, 0, 255), -1)
                cv2.circle(frame, (int(hip_c_x), int(hip_c_y)), 5, (0, 255, 255), -1)
                if ankle_visible:
                    cv2.circle(frame, (int(ankle_c_x), int(ankle_c_y)), 5, (255, 255, 0), -1)

    # 【关键修复】无论有没有检测到人，都必须返回 frame，否则主循环接收到的就是 None
    return frame


frame_count = 0
avg_fps = 0.0  # 用于平滑显示的帧率，防止数字剧烈跳动
while True:
    # 改为从异步队列中读取最新帧
    success, raw_frame = stream.read()
    if not success:
        time.sleep(0.01)  # 队列为空时稍等一下
        continue

    # 核心优化2：在送入推理前，提前将获取到的全自动超高分辨率原图(2560)按比例缩小
    # 好处：
    # 1. 极大降低了 YOLO 模型前处理阶段的 CPU 缩放开销
    # 2. 减少了后面绘制几十个关键点和文本时的 CPU 遍历像素耗时！
    # 3. 使用高效的 INTER_LINEAR 插值算法替代原来的 INTER_AREA
    rh, rw = raw_frame.shape[:2]
    scale = min(preview_width / rw, preview_height / rh)
    if scale < 1.0:
        frame = cv2.resize(raw_frame, (int(rw * scale), int(rh * scale)), interpolation=cv2.INTER_LINEAR)
    else:
        frame = raw_frame

    frame_count += 1
    start_time = time.perf_counter()

    out_frame = process_frame(frame, model)

    process_time = time.perf_counter() - start_time
    current_fps = 1 / process_time if process_time > 0 else 0
    avg_fps = current_fps if avg_fps == 0 else avg_fps * 0.8 + current_fps * 0.2

    print(f"\r[运行中] 帧: {frame_count} | 耗时: {process_time:.4f}s | FPS: {avg_fps:.1f}", end="")

    info_text = f"FPS: {avg_fps:.1f} | Cam: {stream.actual_width}x{stream.actual_height} | Infer: {infer_size}"
    cv2.putText(out_frame, info_text, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

    # 取消原有的末尾性能杀手——极其昂贵的高清后处理双重 Resize
    cv2.imshow("Webcam Fall Detection", out_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.stop()
cv2.destroyAllWindows()