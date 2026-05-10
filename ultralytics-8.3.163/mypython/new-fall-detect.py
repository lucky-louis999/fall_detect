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
    use_fp16 = True
    print(f"✅ 成功检测到显卡: {torch.cuda.get_device_name(0)}")
else:
    compute_device = 'cpu'
    use_fp16 = False
    print("❌ 警告：未检测到可用的 GPU，使用 CPU 运行...")
# ======================================================

# ================= 配置区 =================
camera_id = 0
desired_width = 2560
desired_height = 1440
infer_size = 640
preview_width = 1280
preview_height = 720

# ----------------- 🎯 时序状态机与阈值配置 (痛点解决方案) -----------------
FALL_FRAMES_THRESHOLD = 8      # 异常姿态需要连续存在多少帧，才正式报警 (防瞬间弯腰误报)
DROP_VELOCITY_THRESHOLD = 5.0  # 头部瞬间下坠速度 (像素/帧)。达到此速度说明是突发性下落

# 全局字典缓存，用于根据 track_id 维护每个人的时序历史特征
# 结构: track_history[track_id] = {'head_y_list': [], 'fall_status_count': 0, 'is_confirmed_fall': False}
track_history = {}


# ================= 异步摄像头读取类 (含防卡死安全修改) =================
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

        self.q = Queue(maxsize=1)  # 只需要锁死最新的1帧即可
        self.stopped = False

    def start(self):
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

            # 安全的队列存取模式：解决满队列造成的线程锁死隐患
            try:
                self.q.put_nowait(frame)
            except:
                try:
                    self.q.get_nowait()
                    self.q.put_nowait(frame)
                except:
                    pass

    def read(self):
        if self.q.empty():
            return False, None
        return True, self.q.get()

    def stop(self):
        self.stopped = True
        self.cap.release()

#加载模型
print("正在加载 YOLO 模型...")
model = YOLO("yolo11n-pose.pt")

print("正在初始化异步摄像头流...")
stream = VideoCaptureAsync(camera_id, desired_width, desired_height).start()

time.sleep(2.0)
if stream.stopped:
    print(f"无法打开摄像头 {camera_id}，请检查设备连接！")
    exit()

print(f"实际采集分辨率: {stream.actual_width} x {stream.actual_height} @ {stream.fps} FPS")


def get_skeleton_bbox(kp, kp_conf, conf_thresh=0.3):
    """
    痛点2 & 4解决方案: 获取由关键点撑起的骨架包围盒 (Skeleton BBox)
    不受手部挥舞、持有长形物体的外部 YOLO BBox 膨胀干扰。
    """
    valid_x = [kp[i][0] for i in range(len(kp)) if kp_conf[i] > conf_thresh]
    valid_y = [kp[i][1] for i in range(len(kp)) if kp_conf[i] > conf_thresh]
    if len(valid_x) < 2:
        return 0, 0, 0, 0
    return min(valid_x), min(valid_y), max(valid_x), max(valid_y)


def process_frame(frame, model):
    global track_history
    current_ids = []

    # 进行推理，集成半精度加速
    results = model.track(frame, persist=True, classes=[0], imgsz=infer_size, verbose=False, device=compute_device, half=use_fp16)

    if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        if results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy.cpu().numpy()
            confs = results[0].keypoints.conf.cpu().numpy()
            box_confs = results[0].boxes.conf.cpu().numpy()

            for box, track_id, kp, kp_conf, b_conf in zip(boxes, track_ids, keypoints, confs, box_confs):
                current_ids.append(track_id)
                
                # ------ 时序状态获取初始化 ------
                if track_id not in track_history:
                    track_history[track_id] = {
                        'head_y_list': [],
                        'fall_status_count': 0,
                        'is_confirmed_fall': False
                    }
                history = track_history[track_id]

                # 基础置信度过滤
                if b_conf < 0.55 or kp_conf[0] < 0.4:
                    continue

                # 提取基准特征
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                head_x, head_y = kp[0][0], kp[0][1]

                # 计算纯骨架占比
                sk_x1, sk_y1, sk_x2, sk_y2 = get_skeleton_bbox(kp, kp_conf)
                sk_w = sk_x2 - sk_x1
                sk_h = sk_y2 - sk_y1
                
                # ------ 核心优化: 头部降落速度记录 (痛点 1 解决) ------
                history['head_y_list'].append(head_y)
                if len(history['head_y_list']) > 15:
                    history['head_y_list'].pop(0)

                head_drop_velocity = 0
                if len(history['head_y_list']) >= 5:
                    # 用最近一帧的Y坐标，减去 5 帧前的Y坐标得出瞬时速度 (Y轴向下变大，正值代表下坠)
                    head_drop_velocity = history['head_y_list'][-1] - history['head_y_list'][-5]

                # 计算关节尺寸
                shoulder_c_x = (kp[5][0] + kp[6][0]) / 2
                shoulder_c_y = (kp[5][1] + kp[6][1]) / 2
                shoulder_width = math.hypot(kp[5][0] - kp[6][0], kp[5][1] - kp[6][1])
                shoulder_width = max(shoulder_width, 10.0)

                hip_l_conf, hip_r_conf = kp_conf[11], kp_conf[12]
                hip_visible = hip_l_conf > 0.2 and hip_r_conf > 0.2
                hip_c_x = (kp[11][0] + kp[12][0]) / 2
                hip_c_y = (kp[11][1] + kp[12][1]) / 2

                ankle_l_conf, ankle_r_conf = kp_conf[15], kp_conf[16]
                ankle_visible = ankle_l_conf > 0.3 or ankle_r_conf > 0.3
                if ankle_visible:
                    # 融合左右脚踝
                    if ankle_l_conf > 0.3 and ankle_r_conf > 0.3:
                        ankle_c_x = (kp[15][0] + kp[16][0]) / 2
                        ankle_c_y = (kp[15][1] + kp[16][1]) / 2
                    else:
                        ankle_c_x = kp[15][0] if ankle_l_conf > 0.3 else kp[16][0]
                        ankle_c_y = kp[15][1] if ankle_l_conf > 0.3 else kp[16][1]
                
                # 脊柱长度计算
                spine_dx = head_x - hip_c_x
                spine_dy = head_y - hip_c_y
                spine_length = math.hypot(spine_dx, spine_dy)

                is_abnormal_posture = False

                # ------ 改进版逻辑判别：严格依靠组合条件 ------
                # 前提：痛点3解决，人坐在桌前由于遮挡丢失下半身，绝不妄下断语
                if hip_visible:
                    # 规则 1：横向平躺 / 侧摔
                    if abs(spine_dx) > abs(spine_dy) * 1.2:
                        if ankle_visible:
                            leg_dx = hip_c_x - ankle_c_x
                            leg_dy = hip_c_y - ankle_c_y
                            # 腿平了或者双腿蜷缩
                            if abs(leg_dx) > abs(leg_dy) * 0.8 or abs(leg_dy) < shoulder_width:
                                is_abnormal_posture = True
                        else:
                            # 脚踝被桌子挡住，且姿态偏了。这时必须结合"下坠速度"或之前的摔倒记忆，杜绝弯腰误报
                            if head_drop_velocity > DROP_VELOCITY_THRESHOLD or history['is_confirmed_fall']:
                                is_abnormal_posture = True

                    # 规则 2：翻倒在地/前倾倒地
                    elif head_y > hip_c_y:
                        # 头低于胯部，同样需要结合速度，因为弯腰捡起东西时也会这样
                        if head_drop_velocity > DROP_VELOCITY_THRESHOLD or history['is_confirmed_fall']:
                            is_abnormal_posture = True

                    # 规则 3：透视缩短类跌倒 (垂直朝镜头扑) (痛点2)
                    else:
                        sk_aspect = sk_w / sk_h if sk_h > 0 else 1
                        # 骨架长宽比例失调(挤成一个饼)，且脊椎短缩
                        if sk_aspect > 1.2 and spine_length < shoulder_width * 2.0:
                            if head_drop_velocity > DROP_VELOCITY_THRESHOLD or history['is_confirmed_fall']:
                                is_abnormal_posture = True

                # ------ 状态机衰减机制 (解决偶尔帧抖动与持续性问题) ------
                if is_abnormal_posture:
                    history['fall_status_count'] += 1
                else:
                    # 姿态正常了，将累积危险值逐渐扣减至0，允许重新判定
                    history['fall_status_count'] = max(0, history['fall_status_count'] - 1)
                    if history['fall_status_count'] == 0:
                        history['is_confirmed_fall'] = False
                
                # 只有异常姿态连贯保持了指定帧数，才最终发送不可逆转的报警信号
                if history['fall_status_count'] >= FALL_FRAMES_THRESHOLD:
                    history['is_confirmed_fall'] = True

                # ------ 绘制画面 ------
                is_fall = history['is_confirmed_fall']
                color = (0, 0, 255) if is_fall else (0, 255, 0)

                # 添加警告指示器 (Warning 代表正在积累异常帧数，还没达到警戒线)
                if is_fall:
                    status_txt = "FALLEN (Confirmed)"
                elif history['fall_status_count'] > 0:
                    status_txt = f"Warning... ({history['fall_status_count']})"
                else:
                    status_txt = "Normal"

                text = f"ID:{track_id} {status_txt}"

                # 绘制 YOLO 边界框
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                # 绘制基于关节生成的实际骨架包围框(橙色细线)，直观对比
                if sk_w > 0 and sk_h > 0:
                    cv2.rectangle(frame, (int(sk_x1), int(sk_y1)), (int(sk_x2), int(sk_y2)), (0, 165, 255), 1)

                cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # 绘制关键点
                cv2.circle(frame, (int(head_x), int(head_y)), 5, (255, 0, 0), -1)
                cv2.circle(frame, (int(shoulder_c_x), int(shoulder_c_y)), 5, (255, 0, 255), -1)
                cv2.circle(frame, (int(hip_c_x), int(hip_c_y)), 5, (0, 255, 255), -1)
                if ankle_visible:
                    cv2.circle(frame, (int(ankle_c_x), int(ankle_c_y)), 5, (255, 255, 0), -1)

    # 清理事后离开画面的人员特征，防止爆内存
    keys_to_remove = [k for k in track_history.keys() if k not in current_ids]
    for k in keys_to_remove:
        del track_history[k]

    return frame

frame_count = 0
avg_fps = 0.0

while True:
    success, raw_frame = stream.read()
    if not success:
        time.sleep(0.01)
        continue

    # 前处理降分辨率提升流畅度
    rh, rw = raw_frame.shape[:2]
    scale = min(preview_width / rw, preview_height / rh)
    if scale < 1.0:
        frame = cv2.resize(raw_frame, (int(rw * scale), int(rh * scale)), interpolation=cv2.INTER_LINEAR)
    else:
        frame = raw_frame.copy()

    frame_count += 1
    start_time = time.perf_counter()

    out_frame = process_frame(frame, model)

    process_time = time.perf_counter() - start_time
    current_fps = 1 / process_time if process_time > 0 else 0
    avg_fps = current_fps if avg_fps == 0 else avg_fps * 0.8 + current_fps * 0.2

    print(f"\r[运行中] 帧: {frame_count} | 耗时: {process_time:.4f}s | FPS: {avg_fps:.1f}", end="")

    info_text = f"FPS: {avg_fps:.1f} | Cam: {stream.actual_width}x{stream.actual_height} | Infer: {infer_size}"
    cv2.putText(out_frame, info_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow("Webcam Fall Detection (New Engine)", out_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.stop()
cv2.destroyAllWindows()