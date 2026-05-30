import cv2
import time
import math
import torch
import threading
from queue import Queue
from ultralytics import YOLO
import urllib.request
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

# ================= 人脸识别配置区 (OpenCV DNN) =================
YUNET_PATH = os.path.join(os.path.dirname(__file__), "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(os.path.dirname(__file__), "face_recognition_sface_2021dec.onnx")

# 初始化人脸检测器 (YuNet) 和特征识别器 (SFace)
face_detector = None
face_recognizer = None
if os.path.exists(YUNET_PATH) and os.path.exists(SFACE_PATH):
    # 这里设置的尺寸是初始值，后续在 process_frame 中会针对抠出的图片动态覆盖
    face_detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320), score_threshold=0.8, nms_threshold=0.3)
    face_recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")

# 已知人脸特征库字典: { "张三": [<feature_vector1>, <feature_vector2>] }
known_faces_features = {}
FACES_DIR = os.path.join(os.path.dirname(__file__), "faces")
os.makedirs(FACES_DIR, exist_ok=True)

# 预加载中文字体库（只需在启动时加载一次，避免帧率下降）
try:
    CHINESE_FONT = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8")
except:
    CHINESE_FONT = ImageFont.load_default()

def load_known_faces():
    global known_faces_features
    if not face_detector or not face_recognizer: return
    print(f"正在加载 {FACES_DIR} 目录下的已知人脸照片...")
    for file_name in os.listdir(FACES_DIR):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(FACES_DIR, file_name)
            # 修复 OpenCV 在 Windows 下无法读取含有中文路径或中文文件名的 Bug
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None: 
                print(f"  - ⚠️ [加载失败]: 无法读取文件 {file_name}")
                continue
            
            # 1. 如果图片过大（如手机原图 4K、8K），会导致 YuNet 漏检或报错，需要等比例缩小
            h, w = img.shape[:2]
            max_size = 1024
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
                h, w = img.shape[:2]
                
            # 2. 动态自适应调整 YuNet 输入大小
            face_detector.setInputSize((w, h))
            
            # 3. 检测人脸 (适当降低注册时的置信度阈值)
            face_detector.setScoreThreshold(0.6)
            ret, faces = face_detector.detect(img)
            face_detector.setScoreThreshold(0.8) # 恢复默认值供实时视频流使用
            
            if faces is not None and len(faces) > 0:
                # 取最大的一张脸（防止背景里有其他人）
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                face_box = faces[0][:-1] 
                aligned_face = face_recognizer.alignCrop(img, face_box)
                feature = face_recognizer.feature(aligned_face)
                
                # 允许多张照片注册给同一个人：如 "张三_正面.jpg"、"张三_侧面.jpg" 都会统一到 "张三" 名下
                raw_name = os.path.splitext(file_name)[0]
                base_name = raw_name.split('_')[0]
                
                if base_name not in known_faces_features:
                    known_faces_features[base_name] = []
                known_faces_features[base_name].append(feature)
                
                print(f"  - ✔️ [注册成功]: 归入档案 '{base_name}' (来源文件: {file_name})")
    
    total_persons = len(known_faces_features)
    total_features = sum(len(feats) for feats in known_faces_features.values())
    print(f"共加载 {total_persons} 个人员，总计 {total_features} 张底底特征。")

load_known_faces()
# ===============================================================

# ================= 配置区 =================
camera_id = 1
desired_width = 2560
desired_height = 1440
infer_size = 640
preview_width = 1280
preview_height = 720

# ================= 视频录制配置 =================
# 是否录制保存处理后的视频 (支持自主选择，True为保存，False为不开启)
SAVE_VIDEO = False
# 自己选择视频存储路径（绝对路径最稳妥，扩展名以 .mp4 结尾）
VIDEO_SAVE_PATH = "d:/deeplearning/fall_detect_record.mp4"
# ================================================

# ----------------- 🎯 时序状态机与阈值配置 (痛点解决方案) -----------------
FALL_FRAMES_THRESHOLD = 4      # 【改小】从 8 降为 4，减少常规响应时间
DROP_VELOCITY_THRESHOLD = 3.0  # 【改小】对下落的感知更敏锐
SEVERE_DROP_VELOCITY = 10.0    # 【新增】猛烈下坠速度，用于触发“秒判”

# 全局字典缓存，用于根据 track_id 维护每个人的时序历史特征
# 结构: track_history[track_id] = {'head_y_list': [], 'fall_status_count': 0, 'is_confirmed_fall': False}
track_history = {}


# ================= 异步摄像头读取类 (含防卡死安全修改) =================
class VideoCaptureAsync:
    def __init__(self, src, width, height):
        # 极简模式：不再进行繁琐的出图自检保护判断
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

import os

# 加载模型：使用绝对路径以确保能够找到 ultralytics-8.3.163 根目录下的模型文件
print("正在加载 YOLO 模型...")
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "yolo11n-pose.pt")
model = YOLO(model_path)

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
                        'is_confirmed_fall': False,
                        'name': 'Unknown',
                        'face_score': 0.0,
                        'face_checked': False,
                        'face_check_count': 0,
                        'confirmed_frames': 0
                    }
                history = track_history[track_id]

                # 双阈值置信度过滤：新人用严格门道(0.55)，确认后放宽(0.30)
                # 海报/雕像过不了 0.55 拿不到身份，真人站直入画即可确认
                torso_conf = (kp_conf[5] + kp_conf[6] + kp_conf[11] + kp_conf[12]) / 4.0
                is_trusted = history['confirmed_frames'] >= 5

                if torso_conf < 0.3:
                    history['confirmed_frames'] = max(0, history['confirmed_frames'] - 1)
                    continue

                if is_trusted:
                    if b_conf < 0.30:  # 老熟人：宽松门槛
                        history['confirmed_frames'] = max(0, history['confirmed_frames'] - 1)
                        continue
                else:
                    if b_conf < 0.55:  # 新人：严格门槛
                        continue
                    history['confirmed_frames'] += 1  # 通过严格检验，累积信任度

                # 提取基准特征
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                head_x, head_y = kp[0][0], kp[0][1]

                # ================= 嵌入式友好的人脸识别模块 (时序缓存机制) =================
                # 只有未识别成功的人才做检测，且不需要每帧都测（添加跳帧机制以恢复极速 FPS）
                # 利用 head_y_list 的长度来作为该对象的生命周期帧数计时器
                if face_detector and face_recognizer and not history['face_checked'] and history['face_check_count'] < 15:
                    # 避免在主线程中每帧都做人脸计算，每 3 帧做 1 次检测
                    should_detect_face = (len(history['head_y_list']) % 3 == 0)
                    
                    if should_detect_face:
                        history['face_check_count'] += 1
                        # 从原图中粗略抠出包含头部的区域送给轻量级 YuNet 检测，极大节省开销
                        crop_y1 = max(0, int(y1) - 20)
                        crop_y2 = min(frame.shape[0], int(y1 + (y2 - y1) * 0.4)) # 取人体框的上40%
                        crop_x1 = max(0, int(x1) - 20)
                        crop_x2 = min(frame.shape[1], int(x2) + 20)
                        
                        if crop_y2 > crop_y1 and crop_x2 > crop_x1:
                            person_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2] # 去掉 .copy() 节约内存申请时间
                            
                            # 动态设置 detector 输入尺寸
                            ch, cw = person_crop.shape[:2]
                            face_detector.setInputSize((cw, ch))
                            
                            # 提高抓取质量：拒绝模糊侧脸，只处理由于清晰度高(>0.7)而被检测到的人脸进行特征对齐
                            face_detector.setScoreThreshold(0.7)
                            _, faces = face_detector.detect(person_crop)
                            
                            if faces is not None and len(faces) > 0:
                                # 取检测到的最大脸
                                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                                face_box = faces[0][:-1]
                                aligned_face = face_recognizer.alignCrop(person_crop, face_box)
                                feature = face_recognizer.feature(aligned_face)
                                
                                # C++ 部署时同理进行前向比对
                                best_match_name = "Unknown"
                                best_score = 0.0
                                for known_name, feat_list in known_faces_features.items():
                                    for known_feat in feat_list:
                                        score = face_recognizer.match(feature, known_feat, cv2.FaceRecognizerSF_FR_COSINE)
                                        # 恢复为高标准 0.36~0.38 严堵乱认现象
                                        if score > 0.363 and score > best_score:
                                            best_score = score
                                            best_match_name = known_name
                                
                                if best_match_name != "Unknown":
                                    print(f"\n✨ [人脸识别成功]: ID {track_id} -> {best_match_name} (置信度: {best_score:.2f})")
                                    history['name'] = best_match_name
                                    history['face_score'] = best_score
                                    history['face_checked'] = True # 永久拉入白名单缓存，后续不再重新计算
                                else:
                                    print(f"\r👀 发现人脸，但未匹配库中照片。(最高相似度: {best_score:.2f} < 0.30)", end="")

                # =========================================================================

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
                shoulder_width = max(math.hypot(kp[5][0] - kp[6][0], kp[5][1] - kp[6][1]), 10.0)

                hip_l_conf, hip_r_conf = kp_conf[11], kp_conf[12]
                hip_visible = hip_l_conf > 0.2 and hip_r_conf > 0.2
                hip_c_x = (kp[11][0] + kp[12][0]) / 2
                hip_c_y = (kp[11][1] + kp[12][1]) / 2

                ankle_l_conf, ankle_r_conf = kp_conf[15], kp_conf[16]
                ankle_visible = ankle_l_conf > 0.3 or ankle_r_conf > 0.3
                if ankle_visible:
                    if ankle_l_conf > 0.3 and ankle_r_conf > 0.3:
                        ankle_c_x = (kp[15][0] + kp[16][0]) / 2
                        ankle_c_y = (kp[15][1] + kp[16][1]) / 2
                    else:
                        ankle_c_x = kp[15][0] if ankle_l_conf > 0.3 else kp[16][0]
                        ankle_c_y = kp[15][1] if ankle_l_conf > 0.3 else kp[16][1]
                
                spine_dx = head_x - hip_c_x
                spine_dy = head_y - hip_c_y
                spine_length = math.hypot(spine_dx, spine_dy)

                is_abnormal_posture = False

                # 新规则：膝盖弯曲角度判断 — 蹲下时膝盖深弯，摔倒时腿基本是直的
                is_squatting = False
                if hip_visible:
                    for knee_idx, hip_idx, ankle_idx in [(13, 11, 15), (14, 12, 16)]:
                        if kp_conf[knee_idx] > 0.3 and kp_conf[hip_idx] > 0.2 and kp_conf[ankle_idx] > 0.3:
                            thigh_x = kp[hip_idx][0] - kp[knee_idx][0]
                            thigh_y = kp[hip_idx][1] - kp[knee_idx][1]
                            shin_x = kp[ankle_idx][0] - kp[knee_idx][0]
                            shin_y = kp[ankle_idx][1] - kp[knee_idx][1]
                            thigh_len = math.hypot(thigh_x, thigh_y)
                            shin_len = math.hypot(shin_x, shin_y)
                            if thigh_len > 5 and shin_len > 5:
                                dot = thigh_x * shin_x + thigh_y * shin_y
                                cos_angle = max(-1.0, min(1.0, dot / (thigh_len * shin_len)))
                                knee_angle = math.degrees(math.acos(cos_angle))
                                if knee_angle < 100:  # 侧身：膝盖深弯 → 主动下蹲
                                    is_squatting = True
                                    break
                   

                if hip_visible and not is_squatting:
                    # 规则 1：横向卧倒
                    if abs(spine_dx) > abs(spine_dy) * 1.2:
                        if ankle_visible:
                            leg_dx = hip_c_x - ankle_c_x
                            leg_dy = hip_c_y - ankle_c_y
                            if abs(leg_dx) > abs(leg_dy) * 0.8 or abs(leg_dy) < shoulder_width:
                                is_abnormal_posture = True
                        else:
                            if head_drop_velocity > DROP_VELOCITY_THRESHOLD or history['is_confirmed_fall']:
                                is_abnormal_posture = True

                    # 规则 2：前倾扑倒 (头部低于胯部)
                    elif head_y > hip_c_y:
                        if head_drop_velocity > DROP_VELOCITY_THRESHOLD or history['is_confirmed_fall']:
                            is_abnormal_posture = True

                    # 规则 3：极度透视缩短（需脊柱有明显横向倾斜，排除蹲下/系鞋带等竖直压缩姿态）
                    else:
                        sk_aspect = sk_w / sk_h if sk_h > 0 else 1
                        if sk_aspect > 1.2 and spine_length < shoulder_width * 2.0:
                            # 脊柱水平分量必须显著，蹲下时脊柱竖直(abs(spine_dx)小)，摔倒时才是斜/横的
                            if abs(spine_dx) > abs(spine_dy) * 0.5:
                                if head_drop_velocity > DROP_VELOCITY_THRESHOLD or history['is_confirmed_fall']:
                                    is_abnormal_posture = True
                                
                        # 👑 新增：规则 4 垂直滑倒/瘫倒防漏判 (上下半身折叠 + 坠落速度)
                        # 特征：身体没有横过来，头也没倒挂，但是”屁股坐地上了”，导致整体高度被严重压缩
                        elif head_drop_velocity > DROP_VELOCITY_THRESHOLD or history['is_confirmed_fall']:
                            # 脊柱必须有明显倾斜，排除蹲下/系鞋带等竖直压缩姿态
                            if abs(spine_dx) > abs(spine_dy) * 0.5:
                                # 取全身上下极端视点（有脚踝找脚踝，没脚踝找骨架底端）
                                bottom_y = ankle_c_y if ankle_visible else sk_y2
                                total_height = bottom_y - head_y

                                # 正常人直立时，整体身高大约是肩宽的 3.5 到 4.5 倍
                                # 滑倒跌坐在地时，双膝或者双腿折叠，总高度通常会被压扁到肩宽的 2.7 倍以内
                                if total_height > 0 and total_height < shoulder_width * 2.7:
                                    is_abnormal_posture = True

                # ------ 状态机衰减机制 (解决偶尔帧抖动与持续性问题) ------
                if is_abnormal_posture:
                    # 【核心改进代码】：如果伴随巨大的下落速度，视作“极其危险”，一步加快报警进度
                    if head_drop_velocity > SEVERE_DROP_VELOCITY:
                        history['fall_status_count'] += 2
                    else:
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

                # 将 ID 替换为姓名，如果识别成功的话
                if history['name'] != "Unknown":
                    display_name = f"{history['name']}({history['face_score']:.2f})"
                else:
                    display_name = f"ID:{track_id}"
                
                text = f"{display_name} {status_txt}"

                # 绘制 YOLO 边界框
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                # 绘制基于关节生成的实际骨架包围框(橙色细线)，直观对比
                if sk_w > 0 and sk_h > 0:
                    cv2.rectangle(frame, (int(sk_x1), int(sk_y1)), (int(sk_x2), int(sk_y2)), (0, 165, 255), 1)

                # OpenCV 无法绘制中文的兼容处理
                # 判断是否有非 ascii 字符（中文）
                if max([ord(c) for c in text]) > 127:
                    # 使用 PIL 绘制中文字符
                    # 避免使用耗时的 copy 或多次转换，借用共享内存
                    cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(cv2_img)
                    draw = ImageDraw.Draw(pil_img)
                    
                    # 字体颜色转换为RGB并绘制
                    pil_color = (color[2], color[1], color[0])
                    draw.text((int(x1), int(y1) - 25), text, pil_color, font=CHINESE_FONT)
                    
                    # 转回 OpenCV 格式覆盖原图
                    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                else:
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
video_writer = None

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

    if SAVE_VIDEO:
        if video_writer is None:
            # 动态获取实际画面大小并初始化编解码器 (Windows推荐 mp4v 编码存 mp4)
            vid_h, vid_w = out_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            save_fps = stream.fps if stream.fps > 0 else 30
            video_writer = cv2.VideoWriter(VIDEO_SAVE_PATH, fourcc, save_fps, (vid_w, vid_h))
        video_writer.write(out_frame)

    cv2.imshow("Webcam Fall Detection (New Engine)", out_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if video_writer is not None:
    video_writer.release()
    print(f"\n录像已安全保存完毕，路径: {VIDEO_SAVE_PATH}")

stream.stop()
cv2.destroyAllWindows()