import cv2
import time
import math
import torch
import os
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

# ================= 🚀 核心测试配置区 =================
INPUT_PATH = "D:/google下载/摔倒参考 _ 50种摔倒方式.mp4"  # 【修改这里】填写你想测试的视频文件 (.mp4, .avi) 或图片文件 (.jpg, .png) 的绝对或相对路径
SAVE_RESULT = False            # 【修改这里】是否保存处理后的结果：True 为保存，False 为仅预览不保存
OUTPUT_DIR = "output_results"  # 保存结果的文件夹名称

infer_size = 640
preview_width = 1280
preview_height = 720

# ----------------- 🎯 时序状态机与阈值配置 -----------------
FALL_FRAMES_THRESHOLD = 8      # 异常姿态需要连续存在多少帧，才正式报警 (防瞬间弯腰误报)
DROP_VELOCITY_THRESHOLD = 5.0  # 头部瞬间下坠速度 (像素/帧)

track_history = {}


def get_skeleton_bbox(kp, kp_conf, conf_thresh=0.3):
    """ 获取由关键点撑起的纯净骨架框 """
    valid_x = [kp[i][0] for i in range(len(kp)) if kp_conf[i] > conf_thresh]
    valid_y = [kp[i][1] for i in range(len(kp)) if kp_conf[i] > conf_thresh]
    if len(valid_x) < 2:
        return 0, 0, 0, 0
    return min(valid_x), min(valid_y), max(valid_x), max(valid_y)


def process_frame(frame, model, is_video=True):
    global track_history
    current_ids = []

    # 图片测试不使用 track，而使用原生预测；视频测试继续使用 track
    if is_video:
        results = model.track(frame, persist=True, classes=[0], imgsz=infer_size, verbose=False, device=compute_device, half=use_fp16)
    else:
        results = model(frame, classes=[0], imgsz=infer_size, verbose=False, device=compute_device, half=use_fp16)

    # 安全检查
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        # 视频有追踪ID，图片没有追踪ID时分配默认ID
        if getattr(results[0].boxes, 'id', None) is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        else:
            track_ids = [i for i in range(len(boxes))]

        if results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy.cpu().numpy()
            confs = results[0].keypoints.conf.cpu().numpy()
            box_confs = results[0].boxes.conf.cpu().numpy()

            for box, track_id, kp, kp_conf, b_conf in zip(boxes, track_ids, keypoints, confs, box_confs):
                current_ids.append(track_id)
                
                if track_id not in track_history:
                    track_history[track_id] = {'head_y_list': [], 'fall_status_count': 0, 'is_confirmed_fall': False}
                history = track_history[track_id]

                if b_conf < 0.55 or kp_conf[0] < 0.4:
                    continue

                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                head_x, head_y = kp[0][0], kp[0][1]

                sk_x1, sk_y1, sk_x2, sk_y2 = get_skeleton_bbox(kp, kp_conf)
                sk_w = sk_x2 - sk_x1
                sk_h = sk_y2 - sk_y1
                
                history['head_y_list'].append(head_y)
                if len(history['head_y_list']) > 15:
                    history['head_y_list'].pop(0)

                head_drop_velocity = 0
                if is_video and len(history['head_y_list']) >= 5:
                    head_drop_velocity = history['head_y_list'][-1] - history['head_y_list'][-5]

                shoulder_c_x = (kp[5][0] + kp[6][0]) / 2
                shoulder_c_y = (kp[5][1] + kp[6][1]) / 2
                shoulder_width = max(math.hypot(kp[5][0] - kp[6][0], kp[5][1] - kp[6][1]), 10.0)

                hip_visible = kp_conf[11] > 0.2 and kp_conf[12] > 0.2
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

                if hip_visible:
                    # 规则 1：横向卧倒
                    if abs(spine_dx) > abs(spine_dy) * 1.2:
                        if ankle_visible:
                            leg_dx = hip_c_x - ankle_c_x
                            leg_dy = hip_c_y - ankle_c_y
                            if abs(leg_dx) > abs(leg_dy) * 0.8 or abs(leg_dy) < shoulder_width:
                                is_abnormal_posture = True
                        else:
                            if not is_video or head_drop_velocity > DROP_VELOCITY_THRESHOLD or history['is_confirmed_fall']:
                                is_abnormal_posture = True

                    # 规则 2：前倾扑倒 (头部低于胯部)
                    elif head_y > hip_c_y:
                        if not is_video or head_drop_velocity > DROP_VELOCITY_THRESHOLD or history['is_confirmed_fall']:
                            is_abnormal_posture = True

                    # 规则 3：极度透视缩短
                    else:
                        sk_aspect = sk_w / sk_h if sk_h > 0 else 1
                        if sk_aspect > 1.2 and spine_length < shoulder_width * 2.0:
                            if not is_video or head_drop_velocity > DROP_VELOCITY_THRESHOLD or history['is_confirmed_fall']:
                                is_abnormal_posture = True

                # 若是单张图片测试，抛弃状态机延迟，只要姿态异常即刻报警
                if not is_video:
                    if is_abnormal_posture:
                        history['is_confirmed_fall'] = True
                else: # 视频状态机
                    if is_abnormal_posture:
                        history['fall_status_count'] += 1
                    else:
                        history['fall_status_count'] = max(0, history['fall_status_count'] - 1)
                        if history['fall_status_count'] == 0:
                            history['is_confirmed_fall'] = False
                    
                    if history['fall_status_count'] >= FALL_FRAMES_THRESHOLD:
                        history['is_confirmed_fall'] = True

                # -- 绘制 --
                is_fall = history['is_confirmed_fall']
                color = (0, 0, 255) if is_fall else (0, 255, 0)
                status_txt = "FALLEN" if is_fall else (f"Warning({history['fall_status_count']})" if history['fall_status_count'] > 0 else "Normal")
                text = f"ID:{track_id} {status_txt}"

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                if sk_w > 0 and sk_h > 0:
                    cv2.rectangle(frame, (int(sk_x1), int(sk_y1)), (int(sk_x2), int(sk_y2)), (0, 165, 255), 1)
                cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # 画点
                cv2.circle(frame, (int(head_x), int(head_y)), 5, (255, 0, 0), -1)
                cv2.circle(frame, (int(shoulder_c_x), int(shoulder_c_y)), 5, (255, 0, 255), -1)
                cv2.circle(frame, (int(hip_c_x), int(hip_c_y)), 5, (0, 255, 255), -1)
                if ankle_visible:
                    cv2.circle(frame, (int(ankle_c_x), int(ankle_c_y)), 5, (255, 255, 0), -1)

    # 视频清理冗余内存
    if is_video:
        keys_to_remove = [k for k in track_history.keys() if k not in current_ids]
        for k in keys_to_remove:
            del track_history[k]

    return frame

# ================= 主控制流 =================
if __name__ == "__main__":
    print("正在加载 YOLO 模型...")
    model = YOLO("yolo11n-pose.pt")

    if not os.path.exists(INPUT_PATH):
        print(f"❌ 找不到输入文件: {INPUT_PATH}。请修改文件顶部的 INPUT_PATH 变量！")
        exit()

    is_video_mode = INPUT_PATH.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    
    if SAVE_RESULT and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📂 创建输出文件夹: {OUTPUT_DIR}/")

    # ---------------- 模式 1：处理图片 ----------------
    if not is_video_mode:
        print(f"🖼️ 正在处理单张图片: {INPUT_PATH}")
        frame = cv2.imread(INPUT_PATH)
        if frame is None:
            print("图片读取失败！")
            exit()
            
        out_frame = process_frame(frame.copy(), model, is_video=False)
        
        # 降分辨率预览
        rh, rw = out_frame.shape[:2]
        scale = min(preview_width / rw, preview_height / rh)
        if scale < 1.0:
            preview = cv2.resize(out_frame, (int(rw * scale), int(rh * scale)))
        else:
            preview = out_frame
            
        cv2.imshow("Offline Fall Detection - Image", preview)
        print("💡 图片处理完成！按键盘任意键退出...")
        
        if SAVE_RESULT:
            out_file = os.path.join(OUTPUT_DIR, f"result_{os.path.basename(INPUT_PATH)}")
            cv2.imwrite(out_file, out_frame)
            print(f"💾 结果已保存至: {out_file}")
            
        cv2.waitKey(0)

    # ---------------- 模式 2：处理视频 ----------------
    else:
        print(f"🎥 正在处理视频文件: {INPUT_PATH}")
        cap = cv2.VideoCapture(INPUT_PATH)
        
        # 准备 VideoWriter
        if SAVE_RESULT:
            out_file = os.path.join(OUTPUT_DIR, f"result_{os.path.basename(INPUT_PATH)}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
            print(f"💾 结果将会被保存至: {out_file}")

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            print(f"\r[处理中] 进度: {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)", end="")
            
            # 使用原图处理保证结果的精度，但在展示时缩放
            out_frame = process_frame(frame, model, is_video=True)
            
            if SAVE_RESULT:
                writer.write(out_frame)
                
            # 缩放至屏幕大小以便预览
            rh, rw = out_frame.shape[:2]
            scale = min(preview_width / rw, preview_height / rh)
            if scale < 1.0:
                preview = cv2.resize(out_frame, (int(rw * scale), int(rh * scale)))
            else:
                preview = out_frame
                
            cv2.putText(preview, f"Frame: {frame_count}/{total_frames}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            cv2.imshow("Offline Fall Detection - Video", preview)
            
            # 这里的1毫米等待不仅是为了刷新画面，如果觉得播放太快，可以把1改成20或更大的毫秒数
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⚠️ 用户手动中断了处理。")
                break
                
        cap.release()
        if SAVE_RESULT:
            writer.release()
        print("\n✅ 视频处理彻底完成！")

cv2.destroyAllWindows()