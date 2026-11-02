import streamlit as st
import hashlib
import pandas as pd
import base64
from io import BytesIO
from PIL import Image, ImageDraw
import time
import os
import json
import numpy as np
import colorsys
import onnxruntime as ort
from zhipuai import ZhipuAI

# ===== 导入 cv2 和 webrtc =====
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None
    st.warning("opencv-python未安装，部分高级功能将受限。请运行: pip install opencv-python")

try:
    import av
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    st.warning("streamlit-webrtc未安装，摄像头实时功能将无法使用。请运行: pip install streamlit-webrtc")

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="舌象智能诊断系统",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- 自定义CSS样式 ----------
def inject_custom_css():
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #FDF8F0 0%, #F9EFE0 100%); }
    .main .block-container { background-color: rgba(255, 248, 240, 0.85); border-radius: 20px; padding: 2rem 2rem; box-shadow: 0 8px 32px rgba(0,0,0,0.05); }
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #5E3A2C !important; font-weight: 600 !important; letter-spacing: -0.3px; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #EADBC6 0%, #DCC9AC 100%); border-right: 1px solid #C9AD7A; }
    [data-testid="stSidebar"] * { color: #3A241C !important; }
    [data-testid="stSidebar"] .stButton button { background-color: #A97C50 !important; color: white !important; border: none; border-radius: 30px; transition: all 0.3s; }
    [data-testid="stSidebar"] .stButton button:hover { background-color: #8B5E3C !important; transform: translateY(-2px); }
    .stButton button, .st-emotion-cache-1v0mbdj button { background: linear-gradient(90deg, #B8865B 0%, #9B6A42 100%); color: white !important; border: none; border-radius: 40px; padding: 0.6rem 1.5rem; font-weight: 500; box-shadow: 0 2px 6px rgba(0,0,0,0.1); transition: all 0.3s; }
    .stButton button:hover, .st-emotion-cache-1v0mbdj button:hover { background: linear-gradient(90deg, #9B6A42 0%, #7B4A2E 100%); transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
    [data-testid="stMetricValue"] { background: #FFFFFFCC; padding: 0.2rem 0.6rem; border-radius: 20px; display: inline-block; color: #5E3A2C; font-weight: 600; }
    .stAlert, .stInfo, .stSuccess, .stWarning, .stError { border-radius: 16px; border-left: 5px solid #B8865B; }
    .diagnosis-card { background: #FFFFFFDD; border-radius: 28px; padding: 1.2rem; margin: 0.8rem 0; box-shadow: 0 8px 20px rgba(0,0,0,0.03); border: 1px solid #F0E2D2; backdrop-filter: blur(2px); }
    .metric-container { background: #FCF6EF; border-radius: 24px; padding: 0.8rem; text-align: center; border: 1px solid #E9DBCB; }
    hr { margin: 1rem 0; border-color: #E2D0BA; }
    .stBalloons { filter: hue-rotate(15deg); }
    </style>
    """, unsafe_allow_html=True)

# ===== 图像质量检测函数 =====
def check_image_quality(image):
    issues = []
    try:
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array, img_array, img_array], axis=2)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        gray = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])
        brightness = np.mean(gray)
        if brightness < 80: issues.append("⚠️ 光线太暗，请确保在充足自然光下拍摄")
        elif brightness > 230: issues.append("⚠️ 光线太亮/过曝，请避免强光直射")
        if CV2_AVAILABLE:
            try:
                gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
                laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
                blur_score = np.var(laplacian)
                if blur_score < 50: issues.append("⚠️ 画面模糊，请保持手机稳定并对焦舌体")
                elif blur_score < 100: issues.append("⚠️ 画面略模糊，建议拍摄更清晰的照片")
            except Exception: pass
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        sat = np.where(max_rgb > 0, (max_rgb - min_rgb) / max_rgb, 0)
        if np.mean(sat) > 0.65: issues.append("⚠️ 颜色饱和度异常，请关闭美颜/滤镜功能")
        gray_flat = gray.flatten()
        sorted_gray = np.sort(gray_flat)
        lower_bound = sorted_gray[int(len(sorted_gray) * 0.3)]
        upper_bound = sorted_gray[int(len(sorted_gray) * 0.7)]
        tongue_mask = (gray >= lower_bound) & (gray <= upper_bound)
        tongue_ratio = np.sum(tongue_mask) / gray.size
        if tongue_ratio < 0.15: issues.append("⚠️ 舌体占比过小，请靠近拍摄")
        elif tongue_ratio > 0.75: issues.append("⚠️ 画面过满，请适当拉远")
        high_threshold = np.percentile(gray, 98)
        if np.sum(gray > high_threshold) / gray.size > 0.05: issues.append("⚠️ 检测到反光区域，请避开光源直射")
    except Exception:
        return True, []
    return len(issues) == 0, issues

# ===== 色彩校正功能 =====
def advanced_color_correction(image):
    if not CV2_AVAILABLE: return image
    try:
        img_array = np.array(image).astype(np.float32)
        r_avg, g_avg, b_avg = img_array[:, :, 0].mean(), img_array[:, :, 1].mean(), img_array[:, :, 2].mean()
        gray_avg = (r_avg + g_avg + b_avg) / 3
        r_gain = np.clip(gray_avg / (r_avg + 1e-6), 0.85, 1.15)
        g_gain = np.clip(gray_avg / (g_avg + 1e-6), 0.85, 1.15)
        b_gain = np.clip(gray_avg / (b_avg + 1e-6), 0.85, 1.15)
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * r_gain, 0, 255)
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] * g_gain, 0, 255)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * b_gain, 0, 255)
        return Image.fromarray(img_array.astype(np.uint8))
    except Exception: return image

# ===== 舌苔舌质分离功能 =====
def separate_tongue_coating(image):
    if not CV2_AVAILABLE: return image, image, 0.5, "无法分离"
    try:
        img_array = np.array(image.convert("RGB"))
        H, W = img_array.shape[:2]
        if H < 5 or W < 5: return image, image, 0.5, "尺寸异常"
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)
        _, mask_s = cv2.threshold(s_ch, 30, 255, cv2.THRESH_BINARY)
        _, mask_v = cv2.threshold(v_ch, 50, 255, cv2.THRESH_BINARY)
        tongue_mask = cv2.bitwise_and(mask_s, mask_v)
        k = np.ones((5, 5), np.uint8)
        tongue_mask = cv2.morphologyEx(tongue_mask, cv2.MORPH_CLOSE, k)
        tongue_mask = cv2.morphologyEx(tongue_mask, cv2.MORPH_OPEN, k)
        if np.sum(tongue_mask > 0) < 0.2 * H * W: tongue_mask = np.full((H, W), 255, dtype=np.uint8)
        v_tongue = v_ch.copy()
        v_tongue[tongue_mask == 0] = 0
        _, coating_bin = cv2.threshold(v_tongue, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        coating_mask = cv2.bitwise_and(coating_bin, tongue_mask)
        body_mask = cv2.bitwise_and(cv2.bitwise_not(coating_bin), tongue_mask)
        k2 = np.ones((5, 5), np.uint8)
        coating_mask = cv2.morphologyEx(coating_mask, cv2.MORPH_OPEN, k2)
        coating_mask = cv2.morphologyEx(coating_mask, cv2.MORPH_CLOSE, k2)
        body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, k2)
        body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, k2)
        def overlay(color_rgb, mask):
            base = img_array.astype(np.float32)
            color_layer = np.zeros_like(base)
            color_layer[:, :] = np.array(color_rgb, dtype=np.float32)
            alpha = (mask.astype(np.float32) / 255.0) * 0.55
            alpha = np.expand_dims(alpha, axis=2)
            out = base * (1 - alpha) + color_layer * alpha
            out[mask == 0] = 0
            return Image.fromarray(out.astype(np.uint8))
        coating_img = overlay([0, 200, 0], coating_mask)
        body_img = overlay([220, 40, 40], body_mask)
        total = int(np.sum(tongue_mask > 0))
        coat_px = int(np.sum(coating_mask > 0))
        coating_ratio = float(min(0.95, max(0.05, coat_px / total if total > 0 else 0.5)))
        if coat_px > 100:
            c_rgb = img_array[coating_mask > 0]
            mr, mg, mb = c_rgb[:, 0].mean(), c_rgb[:, 1].mean(), c_rgb[:, 2].mean()
            if mr > mg + 20 and mr > mb + 20: coating_color = "偏黄"
            elif mg > mr + 10 and mg > mb + 10: coating_color = "偏白"
            elif mb > mg + 15 and mb > mr + 15: coating_color = "偏灰"
            else: coating_color = "薄白"
        else: coating_color = "较少或无"
        if np.sum(body_mask > 0) > 100:
            b_rgb = img_array[body_mask > 0]
            br, bg, bb = b_rgb[:, 0].mean(), b_rgb[:, 1].mean(), b_rgb[:, 2].mean()
            if br > bg * 1.2 and br > bb * 1.2: body_color = "偏红"
            elif br < bg * 0.85 and br < bb * 0.85: body_color = "偏淡/偏白"
            else: body_color = "淡红"
        else: body_color = "无法判断"
        return coating_img, body_img, coating_ratio, f"舌苔：{coating_color}，覆盖比例 {coating_ratio*100:.1f}%；舌质：{body_color}"
    except Exception as e: return image, image, 0.5, f"分离失败: {str(e)}"

# ---------- 辨证提示和智能建议映射 ----------
DIAGNOSIS_MAP = {
    "Mirror-Approximated": {"诊断": "镜面舌（舌面光滑如镜）", "辩证提示": "胃阴枯竭，胃气大伤。舌面光滑如镜，舌苔完全脱落。", "智能建议": "1. 立即就医\n2. 营养支持：高蛋白、高维生素\n3. 中药调理：沙参麦冬汤"},
    "Thin-White": {"诊断": "薄白苔（正常舌象）", "辩证提示": "气血调和，胃气充足。", "智能建议": "1. 维持现状\n2. 饮食调理：均衡饮食\n3. 适度运动\n4. 充足睡眠"},
    "White-Greasy": {"诊断": "白腻苔（湿浊内蕴）", "辩证提示": "湿浊内停，痰饮积聚。多见于消化不良。", "智能建议": "1. 健脾祛湿：茯苓、白术\n2. 减少油腻\n3. 平胃散、二陈汤\n4. 按摩足三里"},
    "Yellow-Greasy": {"诊断": "黄腻苔（湿热内蕴）", "辩证提示": "湿热蕴结，痰热互结。", "智能建议": "1. 清热利湿：黄芩、黄连\n2. 饮食清淡\n3. 及时就医\n4. 多饮水"},
    "Grey-Black": {"诊断": "灰黑苔（热极或寒极）", "辩证提示": "病情危重，或热极或寒极。", "智能建议": "1. 紧急就医\n2. 全面检查\n3. 专科治疗"}
}

# ===== 高精度舌象识别算法 =====
def extract_tongue_color_features(image, coating_ratio=None, coating_description=None):
    try:
        img_array = np.array(image)
        h_img, w_img = img_array.shape[:2]
        crop_size = max(10, min(h_img, w_img) // 3)
        cy, cx = h_img // 2, w_img // 2
        center = img_array[cy - crop_size // 2: cy + crop_size // 2, cx - crop_size // 2: cx + crop_size // 2]
        if center.size == 0: return "Thin-White", 0.75, {"错误": "空"}
        r, g, b = np.mean(center[:, :, 0]), np.mean(center[:, :, 1]), np.mean(center[:, :, 2])
        if np.isnan(r) or np.isnan(g) or np.isnan(b): return "Thin-White", 0.75, {"错误": "NaN"}
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        hue, sat, val = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
        uniformity = (np.std(center[:, :, 0]) + np.std(center[:, :, 1]) + np.std(center[:, :, 2])) / 3
        yellow_ratio = (r - b) / (g + 1)
        confidence = 0.85
        
        if coating_ratio is not None and coating_ratio < 0.08:
            tongue_type, confidence = "Mirror-Approximated", 0.94
        elif coating_ratio is not None and coating_ratio > 0.35:
            if yellow_ratio > 10 and sat > 0.45: tongue_type, confidence = "Yellow-Greasy", 0.91
            elif sat > 0.25 and brightness < 180: tongue_type, confidence = "White-Greasy", 0.89
            elif brightness > 210 and sat < 0.15: tongue_type, confidence = "Mirror-Approximated", 0.92
            elif brightness < 90: tongue_type, confidence = "Grey-Black", 0.90
            else: tongue_type, confidence = "Thin-White", 0.80
        elif yellow_ratio > 12 and sat > 0.48: tongue_type, confidence = "Yellow-Greasy", 0.89
        elif sat > 0.28 and 110 < brightness < 170: tongue_type, confidence = "White-Greasy", 0.87
        else: tongue_type, confidence = "Thin-White", 0.82
        
        feature_info = {"亮度": round(brightness, 2), "饱和度": round(sat, 2), "色调": round(hue, 2), "均匀度": round(uniformity, 2), "黄色指数": round(yellow_ratio, 2)}
        if coating_ratio is not None:
            feature_info["舌苔覆盖比例"] = round(coating_ratio * 100, 1)
            feature_info["舌苔分析"] = coating_description or "无"
        return tongue_type, round(max(0.75, min(0.98, confidence)), 2), feature_info
    except Exception: return "Thin-White", 0.75, {}

# ==============================================
# ========== 【ONNX 推理模块】 ==========
# ==============================================
IMG_SIZE = 640

@st.cache_resource
def load_onnx_model():
    onnx_path = "yolov10n.onnx"
    if not os.path.exists(onnx_path): raise FileNotFoundError("找不到 yolov10n.onnx")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return session, session.get_inputs()[0].name

# ===== 实时视频处理器 =====
class TongueVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.conf_threshold = 0.35
        self.session, self.input_name = load_onnx_model()
        self.latest_frame = None # 存储原始画面

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.latest_frame = img.copy() # 存下最干净的图用于分析
        
        h0, w0 = img.shape[:2]
        scale = min(IMG_SIZE / w0, IMG_SIZE / h0)
        new_w, new_h = int(w0 * scale), int(h0 * scale)
        resized = cv2.resize(img, (new_w, new_h))
        pad_img = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
        pad_img[:new_h, :new_w] = resized
        blob = pad_img[:, :, ::-1].astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)

        try:
            outputs = self.session.run(None, {self.input_name: blob})
            pred = np.squeeze(outputs[0])
            if pred.ndim == 2 and pred.shape[0] < pred.shape[1]: pred = pred.T
            
            if pred.ndim == 2 and pred.shape[1] >= 5:
                boxes_xywh = pred[:, :4]
                scores = np.nan_to_num(pred[:, 4:], nan=0.0, posinf=0.0, neginf=0.0)
                conf = np.clip(np.max(scores, axis=1), 0.0, 1.0)
                valid_indices = np.where(conf > self.conf_threshold)[0]
                
                if len(valid_indices) > 0:
                    best_idx = np.argmax(conf[valid_indices])
                    bx, by, bw_, bh_ = boxes_xywh[valid_indices[best_idx]]
                    x1_orig, y1_orig = int((bx - bw_ / 2) / scale), int((by - bh_ / 2) / scale)
                    x2_orig, y2_orig = int((bx + bw_ / 2) / scale), int((by + bh_ / 2) / scale)
                    box_w, box_h = x2_orig - x1_orig, y2_orig - y1_orig
                    
                    # 实时绿框：宽度 35%，高度 25%，下移 18%
                    new_w_box, new_h_box = int(box_w * 0.35), int(box_h * 0.25)
                    center_x, center_y = x1_orig + box_w // 2, y1_orig + box_h // 2 + int(box_h * 0.18)
                    x1_new, y1_new = max(0, center_x - new_w_box // 2), max(0, center_y - new_h_box // 2)
                    x2_new, y2_new = min(w0, center_x + new_w_box // 2), min(h0, center_y + new_h_box // 2)

                    cv2.rectangle(img, (x1_new, y1_new), (x2_new, y2_new), (0, 255, 0), 2)
                    label = f"Target Area"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(img, (x1_new, max(y1_new - 25, 0)), (x1_new + tw, max(y1_new, 25)), (0, 255, 0), -1)
                    cv2.putText(img, label, (x1_new, max(y1_new - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    try:
                        preview = img[y1_new:y2_new, x1_new:x2_new]
                        if preview.size > 0:
                            img[10:110, 10:110] = cv2.resize(preview, (100, 100))
                            cv2.rectangle(img, (10, 10), (110, 110), (0, 255, 0), 2)
                    except: pass
        except Exception as e: print(f"[实时异常]: {e}")
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def onnx_detect(image_path, conf_thres=0.1, iou_thres=0.45):
    session, input_name = load_onnx_model()
    try:
        img = np.array(Image.open(image_path).convert('RGB'))[:, :, ::-1].copy()
    except Exception: return []
    h0, w0 = img.shape[:2]
    scale = min(IMG_SIZE / w0, IMG_SIZE / h0)
    new_w, new_h = int(w0 * scale), int(h0 * scale)
    resized = cv2.resize(img, (new_w, new_h))
    pad_img = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
    pad_img[:new_h, :new_w] = resized
    blob = pad_img[:, :, ::-1].astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)
    blob = np.expand_dims(blob, axis=0)
    outputs = session.run(None, {input_name: blob})
    pred = np.squeeze(outputs[0])
    if pred.ndim == 2 and pred.shape[0] < pred.shape[1]: pred = pred.T
    if pred.ndim != 2 or pred.shape[1] < 5: return []
    boxes_xywh, scores = pred[:, :4], pred[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    conf = np.clip(np.max(scores, axis=1), 0.0, 1.0)
    mask = conf > conf_thres
    boxes_xywh, conf, class_ids = boxes_xywh[mask], conf[mask], class_ids[mask]
    det_list = []
    if len(boxes_xywh) > 0:
        xyxy = np.zeros_like(boxes_xywh)
        xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
        xyxy /= scale
        indices = cv2.dnn.NMSBoxes(xyxy.tolist(), conf.tolist(), conf_thres, iou_thres)
        if len(indices) > 0:
            for i in np.array(indices).flatten():
                det_list.append((int(class_ids[i]), float(conf[i]), xyxy[i].tolist()))
    return det_list

def analyze_tongue(image_path, enable_color_correction=True, enable_coating_separation=True):
    try:
        img = Image.open(image_path).convert("RGB")
        original_img = img.copy()
        if enable_color_correction and CV2_AVAILABLE:
            try:
                corrected = advanced_color_correction(img)
                if np.mean(np.abs(np.array(original_img).astype(np.float32) - np.array(corrected).astype(np.float32))) < 60:
                    img = corrected
            except Exception: pass

        det_list = onnx_detect(image_path)
        best_conf = 0.0
        for _, conf, _ in det_list:
            if conf > best_conf: best_conf = conf
        best_conf = float(np.clip(best_conf, 0.0, 1.0))

        # ==== 核心优化：动态聚焦截取（照片框保持原大小，不变） ====
        bw, bh = img.size
        cx1, cy1 = max(0, bw//2 - bw//3), max(0, bh//2 - bh//3)
        cx2, cy2 = min(bw, bw//2 + bw//3), min(bh, bh//2 + bh//3)
        
        if best_conf > 0.1 and len(det_list) > 0:
            best_idx = np.argmax([d[1] for d in det_list])
            xyxy = det_list[best_idx][2]
            x1_orig, y1_orig, x2_orig, y2_orig = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            
            box_w, box_h = x2_orig - x1_orig, y2_orig - y1_orig
            
            # 照片显示框：宽度 40%，高度 30%，下移 15%
            new_w_box = int(box_w * 0.4)
            new_h_box = int(box_h * 0.3)
            center_x = x1_orig + box_w // 2
            center_y = y1_orig + box_h // 2 + int(box_h * 0.15)
            
            cx1 = max(0, center_x - new_w_box // 2)
            cy1 = max(0, center_y - new_h_box // 2)
            cx2 = min(bw, center_x + new_w_box // 2)
            cy2 = min(bh, center_y + new_h_box // 2)
            
            if cx2 <= cx1: cx2 = cx1 + 20
            if cy2 <= cy1: cy2 = cy1 + 20

        cropped_img = img.crop((cx1, cy1, cx2, cy2))
        # ==== 动态聚焦截取结束 ====

        coating_img_base64, body_img_base64 = None, None
        coating_ratio, coating_description = None, None
        if enable_coating_separation and CV2_AVAILABLE:
            coating_img, body_img, coating_ratio, coating_description = separate_tongue_coating(cropped_img)
            buffered = BytesIO(); coating_img.save(buffered, format="PNG")
            coating_img_base64 = base64.b64encode(buffered.getvalue()).decode()
            buffered = BytesIO(); body_img.save(buffered, format="PNG")
            body_img_base64 = base64.b64encode(buffered.getvalue()).decode()

        tongue_type, confidence, feature_info = extract_tongue_color_features(cropped_img, coating_ratio, coating_description)

        draw_img = img.copy()
        ImageDraw.Draw(draw_img).rectangle([cx1, cy1, cx2, cy2], outline="#FF2200", width=6)
        buffered = BytesIO(); draw_img.save(buffered, format="PNG")
        boxed_image_base64 = base64.b64encode(buffered.getvalue()).decode()
        buffered = BytesIO(); cropped_img.save(buffered, format="PNG")
        tongue_region_base64 = base64.b64encode(buffered.getvalue()).decode()

        diagnosis_info = DIAGNOSIS_MAP.get(tongue_type, {"诊断": "未知", "辩证提示": "无法判断", "智能建议": "请咨询医生"})
        result_dict = {
            "舌象类型": tongue_type, "置信度": confidence, "诊断": diagnosis_info["诊断"],
            "辩证提示": diagnosis_info["辩证提示"], "智能建议": diagnosis_info["智能建议"],
            "boxed_image_base64": boxed_image_base64, "tongue_region_base64": tongue_region_base64,
            "分析时间": time.strftime("%Y-%m-%d %H:%M:%S"), "特征信息": feature_info, "人体检测置信度": best_conf,
            "检测详情": "无检测" if not det_list else f"命中{len(det_list)}个目标"
        }
        if coating_img_base64:
            result_dict.update({"coating_img_base64": coating_img_base64, "body_img_base64": body_img_base64, "舌苔分离分析": coating_description})
        return result_dict
    except Exception as e:
        st.error(f"分析出错: {e}")
        return None

def analyze_video(video_path, sample_interval=10, max_frames=40, **kwargs):
    if not CV2_AVAILABLE: return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    session, input_name = load_onnx_model()
    progress = st.progress(0.0, text="正在逐帧检测...")
    frame_idx, best_conf, best_frame, detected_count, checked = 0, -1.0, None, 0, 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % sample_interval == 0:
            checked += 1
            h0, w0 = frame.shape[:2]
            scale = min(IMG_SIZE / w0, IMG_SIZE / h0)
            blob = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
            blob[:int(h0*scale), :int(w0*scale)] = cv2.resize(frame, (int(w0*scale), int(h0*scale)))
            blob = blob[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
            pred = np.squeeze(session.run(None, {input_name: blob})[0])
            if pred.ndim == 2 and pred.shape[0] < pred.shape[1]: pred = pred.T
            if pred.ndim == 2 and pred.shape[1] >= 5:
                conf = np.clip(np.max(pred[:, 4:], axis=1), 0.0, 1.0)
                if len(conf) > 0:
                    max_c = float(np.max(conf))
                    if max_c > 0.1:
                        detected_count += 1
                        if max_c > best_conf: best_conf, best_frame = max_c, frame.copy()
            if checked >= max_frames or total_frames <= 0: break
            progress.progress(min(1.0, frame_idx / total_frames), text=f"已检测 {checked} 帧")
        frame_idx += 1
    cap.release(); progress.progress(1.0, text="帧检测完成")
    if best_frame is None:
        st.error("❌ 视频中未检测到清晰的舌体。"); return None
    best_path = f"video_best_{int(time.time())}.png"
    cv2.imwrite(best_path, best_frame)
    with st.spinner("正在对最佳帧进行舌象分析..."):
        result = analyze_tongue(best_path, **kwargs)
    if result: result.update({"视频检测信息": f"抽样 {checked} 帧 / 命中 {detected_count} 帧", "来源": "视频逐帧诊断"})
    os.remove(best_path)
    return result

def save_history(history):
    with open("tongue_history.json", "w", encoding="utf-8") as f: json.dump(history, f, ensure_ascii=False, indent=2)

def load_history():
    if os.path.exists("tongue_history.json"):
        with open("tongue_history.json", "r", encoding="utf-8") as f: return json.load(f)
    return []

def make_hashes(password): return hashlib.sha256(str.encode(password)).hexdigest()
def check_hashes(password, hashed_text): return make_hashes(password) == hashed_text

def init_auth():
    if 'users' not in st.session_state:
        st.session_state['users'] = {"doctor": make_hashes("123456"), "student": make_hashes("654321")}
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['username'] = ""

def login_page():
    st.markdown("<h1 style='text-align: center;'>🔐 舌象诊断 · 专属登录</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #7A4A2E; margin-bottom: 2rem;'>欢迎使用智能中医舌象分析系统</p>",
        unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        with st.container():
            st.markdown("<div class='diagnosis-card'>", unsafe_allow_html=True)
            
            choice = st.radio("选择操作", ["登录", "注册"], horizontal=True, label_visibility="collapsed")
            st.markdown("<br>", unsafe_allow_html=True)

            if choice == "登录":
                username = st.text_input("用户名", key="login_username", placeholder="请输入用户名")
                password = st.text_input("密码", type="password", key="login_password", placeholder="请输入密码")
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("登 录", use_container_width=True):
                    if username in st.session_state['users'] and check_hashes(password, st.session_state['users'][username]):
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = username
                        st.rerun()
                    else:
                        st.error("用户名或密码错误！")
            else:
                new_user = st.text_input("新用户名", key="reg_username", placeholder="请输入新用户名")
                new_password = st.text_input("新密码", type="password", key="reg_password", placeholder="请设置密码")
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("注 册", use_container_width=True):
                    if new_user in st.session_state['users']:
                        st.warning("用户已存在！")
                    elif new_user and new_password:
                        st.session_state['users'][new_user] = make_hashes(new_password)
                        st.success("注册成功！请切换到登录页")
                    else:
                        st.error("用户名和密码不能为空")
            
            st.markdown("</div>", unsafe_allow_html=True)

# ---------- 舌象分析页面 ----------
def analysis_page():
    st.markdown("<h1>👅 智能舌象诊断</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6B4C3C;'>上传舌部照片或视频，或开启摄像头实时检测，系统将基于YOLO模型精准定位舌体区域并给出中医辨证建议。</p>", unsafe_allow_html=True)
    st.markdown("---")

    if "history" not in st.session_state: st.session_state.history = load_history()
    if "current_result" not in st.session_state: st.session_state.current_result = None
    if "upload_passed_quality" not in st.session_state: st.session_state.upload_passed_quality = False
    if "current_temp_path" not in st.session_state: st.session_state.current_temp_path = None
    if "captured_frame_path" not in st.session_state: st.session_state.captured_frame_path = None

    # ====== 患者基本信息 ======
    st.markdown("<div class='diagnosis-card'>", unsafe_allow_html=True)
    st.subheader("👤 患者基本信息")
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        gender = st.selectbox("性别", ["男", "女"], index=0)
        age = st.number_input("年龄（岁）", min_value=1, max_value=120, value=30, step=1)
    with info_col2:
        height = st.number_input("身高（cm）", min_value=50, max_value=250, value=170, step=1)
        weight = st.number_input("体重（kg）", min_value=10, max_value=300, value=65, step=1)

    bmi = round(weight / ((height / 100) ** 2), 1) if height > 0 else 0.0
    bmi_level = "偏瘦" if bmi < 18.5 else "正常" if 18.5 <= bmi < 24 else "超重" if 24 <= bmi < 28 else "肥胖"
    st.success(f"**自动计算BMI指数：{bmi} | 体型：{bmi_level}**")
    st.markdown("</div>", unsafe_allow_html=True)

    # ====== 上传与设置 ======
    st.markdown('<div class="diagnosis-card">', unsafe_allow_html=True)
    st.subheader("📤 获取舌象素材")
    with st.expander("📷 拍摄指南（点击收起）", expanded=True):
        st.markdown("""
        <div style="background:#FCF6EF; padding:1rem; border-radius:16px;">
        <p>✅ 在<strong>自然光</strong>下拍摄，避免强光直射或过暗</p>
        <p>✅ 舌头<strong>自然平伸</strong>，不要过度用力</p>
        <p>✅ 对焦舌体，确保图像<strong>清晰不模糊</strong></p>
        <p>✅ 关闭相机<strong>美颜、滤镜、HDR</strong>等功能</p>
        </div>
        """, unsafe_allow_html=True)
    with st.expander("⚙️ 高级分析设置（可选）"):
        col_set1, col_set2 = st.columns(2)
        with col_set1: enable_color_correction = st.checkbox("🎨 自动色彩校正（白平衡）", value=True)
        with col_set2: enable_coating_separation = st.checkbox("🔬 舌苔/舌质分离分析", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

    upload_mode = st.radio(
        "📥 选择输入方式",
        ["📷 照片诊断", "🎬 视频文件诊断", "📹 电脑摄像头实时诊断"],
        horizontal=True, label_visibility="collapsed"
    )

    uploaded_file, uploaded_video, video_path = None, None, None

    # --- 模式 1：照片诊断 ---
    if upload_mode == "📷 照片诊断":
        uploaded_file = st.file_uploader("点击或拖拽上传舌象照片", type=["jpg", "jpeg", "png"])

    # --- 模式 2：视频文件诊断 ---
    elif upload_mode == "🎬 视频文件诊断":
        uploaded_video = st.file_uploader("上传舌象短视频（建议 3~15 秒）", type=["mp4", "avi", "mov", "mkv"])
        st.caption("💡 系统将自动抽样关键帧，用 YOLOv10n 定位舌体，并对最佳帧做舌象辨证分析。")

    # --- 模式 3：电脑摄像头实时诊断 ---
    else:
        st.info("👇 请点击 'START' 允许浏览器调用摄像头。画面中会实时显示 YOLO 识别框。")
        if not WEBRTC_AVAILABLE:
            st.error("❌ 摄像头功能需要安装 streamlit-webrtc 库。请运行: `pip install streamlit-webrtc`")
        else:
            ctx = webrtc_streamer(
                key="tongue-realtime-video",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=TongueVideoProcessor,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False},
            )
            st.markdown("---")
            
            if st.button("📸 定格画面并进行深度辨证分析", type="primary", use_container_width=True):
                # 核心修复：直接从处理器实例拿干净的原图
                if ctx.video_processor is not None:
                    latest_img = getattr(ctx.video_processor, "latest_frame", None)
                    if latest_img is not None:
                        temp_cap_path = f"temp_cap_{int(time.time())}.png"
                        cv2.imwrite(temp_cap_path, latest_img)
                        st.session_state.captured_frame_path = temp_cap_path
                        ctx.video_processor.latest_frame = None # 清空缓存
                        st.rerun()
                    else:
                        st.warning("还没收到画面数据，请稍等 1-2 秒再点，或检查摄像头是否在播放。")
                else:
                    st.warning("摄像头未启动或已被关闭，请先点击 START。")

            # 展示定格结果
            if st.session_state.captured_frame_path and os.path.exists(st.session_state.captured_frame_path):
                st.success("✅ 成功定格画面！请点击下方按钮进行深度分析。")
                img = Image.open(st.session_state.captured_frame_path)
                st.image(img, caption="待深度分析的画面", width=400)
                video_path = st.session_state.captured_frame_path

    # ====== 照片分析执行 ======
    if uploaded_file is not None:
        temp_path = f"temp_{int(time.time())}_{uploaded_file.name}"
        if st.session_state.current_temp_path != temp_path:
            st.session_state.upload_passed_quality = False
            st.session_state.current_temp_path = temp_path
            with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())
            img = Image.open(temp_path).convert("RGB")
            is_qualified, issues = check_image_quality(img)
            if is_qualified:
                st.session_state.upload_passed_quality = True
                st.success("✅ 图像质量合格，可以开始分析！")
            else:
                st.session_state.upload_passed_quality = False
                st.error("❌ 图像质量不符合要求，请重新拍摄/上传")
                for issue in issues: st.warning(issue)
                if os.path.exists(temp_path): os.remove(temp_path)
                st.session_state.current_temp_path = None

        if st.session_state.upload_passed_quality and st.session_state.current_temp_path:
            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.subheader("🔍 检测定位结果")
                boxed_placeholder = st.empty()
                conf_caption_placeholder = st.empty()
            with col2:
                st.subheader("✂️ 舌体区域特写")
                tongue_placeholder = st.empty()

            if st.button("✨ 开始分析", type="primary", use_container_width=True):
                with st.spinner("正在分析舌象..."):
                    result = analyze_tongue(st.session_state.current_temp_path, enable_color_correction, enable_coating_separation)
                    if result:
                        st.session_state.current_result = result
                        result.update({"时间": result["分析时间"], "性别": gender, "年龄": age, "BMI": bmi, "体型": bmi_level, "来源": "照片诊断"})
                        st.session_state.history.append(result)
                        save_history(st.session_state.history)
                        st.success("✅ 分析完成！")
                        st.balloons()

                        boxed_placeholder.image(f"data:image/png;base64,{result['boxed_image_base64']}", width=400)
                        tongue_placeholder.image(f"data:image/png;base64,{result['tongue_region_base64']}", width=400)
                        conf_caption_placeholder.caption(f"Conf: {result.get('人体检测置信度', 0.0):.2f}")

                        if 'coating_img_base64' in result:
                            st.markdown("---")
                            st.subheader("🔬 舌苔/舌质分离分析")
                            col_coat1, col_coat2 = st.columns(2)
                            with col_coat1:
                                st.markdown("**🟢 舌苔区域**")
                                st.image(f"data:image/png;base64,{result['coating_img_base64']}", width=300)
                            with col_coat2:
                                st.markdown("**🔴 舌质区域**")
                                st.image(f"data:image/png;base64,{result['body_img_base64']}", width=300)
                            if '舌苔分离分析' in result:
                                st.info(f"📊 {result['舌苔分离分析']}")
                                st.error(f"📌 最终判定：{result['舌象类型']} (置信度 {result['置信度'] * 100:.1f}%)")

                        # AI 解读
                        st.markdown("---")
                        st.markdown("<h3>🤖 AI 舌象智能解读</h3>", unsafe_allow_html=True)
                        with st.spinner("大模型深度解读中..."):
                            try:
                                api_key = st.secrets.get("ZHIPU_API_KEY", "e7477314b09b4be087232c42440cfe29.RKgXonaZScPmwLEm")
                                if api_key != "YOUR_KEY_HERE":
                                    client = ZhipuAI(api_key=api_key)
                                    b64 = base64.b64encode(uploaded_file.getvalue()).decode()
                                    prompt_text = f"初步算法识别为：{result['舌象类型']}。请扮演专业中医师，结合图片给出详细解读。按特征描述、辨证分析、调理建议三段输出。"
                                    resp = client.chat.completions.create(
                                        model="glm-4v-flash",
                                        messages=[{"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]
                                    )
                                    st.markdown(resp.choices[0].message.content)
                            except Exception as e: st.error(f"AI 解读失败: {e}")

                        col_dial, col_adv = st.columns(2)
                        with col_dial:
                            st.markdown(f"<div style='background:#FFF3E0;padding:15px;border-radius:8px;'><h4 style='color:#E65100;'>🔴 辨证提示</h4><p>{result['辩证提示'].replace(chr(10), '<br>')}</p></div>", unsafe_allow_html=True)
                        with col_adv:
                            st.markdown(f"<div style='background:#E3F2FD;padding:15px;border-radius:8px;'><h4 style='color:#1565C0;'>💡 智能建议</h4><p>{result['智能建议'].replace(chr(10), '<br>')}</p></div>", unsafe_allow_html=True)

    # ====== 视频/摄像头分析执行 ======
    if uploaded_video is not None:
        video_path = f"temp_video_{int(time.time())}_{uploaded_video.name}"
        with open(video_path, "wb") as f: f.write(uploaded_video.getbuffer())

    if video_path and os.path.exists(video_path):
        st.markdown("---")
        st.subheader("🎬 画面分析")
        if uploaded_video is not None: st.video(video_path)

        col_v1, col_v2 = st.columns(2)
        with col_v1: sample_interval = st.slider("帧抽样间隔", 1, 30, 10)
        with col_v2: max_frames = st.slider("最多检测帧数", 10, 100, 40)

        if st.button("🎬 开始深度诊断", type="primary", use_container_width=True):
            if uploaded_video is None and video_path.endswith(".png"):
                with st.spinner("正在分析抓拍画面..."):
                    result = analyze_tongue(video_path, enable_color_correction, enable_coating_separation)
                    if result: result["来源"] = "摄像头实时抓拍"
            else:
                result = analyze_video(video_path, sample_interval, max_frames, enable_color_correction=enable_color_correction, enable_coating_separation=enable_coating_separation)
                
            if result:
                st.session_state.current_result = result
                result.update({"时间": result["分析时间"], "性别": gender, "年龄": age, "BMI": bmi, "体型": bmi_level})
                st.session_state.history.append(result)
                save_history(st.session_state.history)
                st.success("✅ 诊断完成！")
                st.balloons()

                st.subheader("🔍 最佳帧检测定位结果")
                st.image(f"data:image/png;base64,{result['boxed_image_base64']}", width=400)
                st.caption(f"Conf: {result.get('人体检测置信度', 0):.2f}")
                if "视频检测信息" in result: st.info(f"🎬 {result['视频检测信息']}")

                st.subheader("✂️ 舌体区域特写")
                st.image(f"data:image/png;base64,{result['tongue_region_base64']}", width=400)

                if 'coating_img_base64' in result:
                    st.markdown("---")
                    st.subheader("🔬 舌苔/舌质分离分析")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**🟢 舌苔区域**")
                        st.image(f"data:image/png;base64,{result['coating_img_base64']}", width=300)
                    with c2:
                        st.markdown("**🔴 舌质区域**")
                        st.image(f"data:image/png;base64,{result['body_img_base64']}", width=300)
                    if '舌苔分离分析' in result:
                        st.info(f"📊 {result['舌苔分离分析']}")
                        st.error(f"📌 最终判定：{result['舌象类型']} (置信度 {result['置信度'] * 100:.1f}%)")

                st.markdown("---")
                st.subheader("📊 辨证与建议")
                c_dial, c_adv = st.columns(2)
                with c_dial:
                    st.markdown(f"<div style='background:#FFF3E0;padding:15px;border-radius:8px;'><h4 style='color:#E65100;'>🔴 辨证提示</h4><p>{result['辩证提示'].replace(chr(10), '<br>')}</p></div>", unsafe_allow_html=True)
                with c_adv:
                    st.markdown(f"<div style='background:#E3F2FD;padding:15px;border-radius:8px;'><h4 style='color:#1565C0;'>💡 智能建议</h4><p>{result['智能建议'].replace(chr(10), '<br>')}</p></div>", unsafe_allow_html=True)

        if uploaded_video is not None:
            try: os.remove(video_path)
            except Exception: pass

    # =========================================================
    # ====== 底部常驻：AI 医生智能对话窗口 ======
    # =========================================================
    st.markdown("---")
    st.subheader("💬 AI 医生智能问答")
    st.caption("你可以在这里继续追问关于诊断结果、中医养生等问题。")
    if "chat_history" not in st.session_state: st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("向 AI 医生提问（例如：白腻苔平时吃什么好？）"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    api_key = st.secrets.get("ZHIPU_API_KEY", "e7477314b09b4be087232c42440cfe29.RKgXonaZScPmwLEm")
                    if api_key != "YOUR_KEY_HERE":
                        client = ZhipuAI(api_key=api_key)
                        if uploaded_file is not None:
                            b64 = base64.b64encode(uploaded_file.getvalue()).decode()
                            resp = client.chat.completions.create(model="glm-4v-flash", messages=[{"role": "user", "content": [{"type": "text", "text": f"中医舌象图。提问：{prompt}。请简短专业地回答。"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}])
                        else:
                            resp = client.chat.completions.create(model="glm-4-flash", messages=[{"role": "user", "content": prompt}])
                        ans = resp.choices[0].message.content
                        st.markdown(ans)
                        st.session_state.chat_history.append({"role": "assistant", "content": ans})
                except Exception as e: st.error(f"AI 失败: {e}")

# ---------- 历史记录页面 ----------
def history_page():
    st.markdown("<h1>📋 诊断历史档案</h1>", unsafe_allow_html=True)
    st.markdown("---")
    history = load_history()
    if len(history) == 0:
        st.info("✨ 暂无历史分析记录。")
        return
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总记录数", len(history))
    with col2:
        tongue_types = [h.get("舌象类型", "未知") for h in history]
        st.metric("最常见舌象", max(set(tongue_types), key=tongue_types.count) if tongue_types else "无")
    with col3:
        avg_confidence = sum([h.get("置信度", 0) for h in history]) / len(history)
        st.metric("平均置信度", f"{avg_confidence*100:.1f}%")
    
    st.subheader("📜 历史记录列表")
    table_data = [{"序号": idx + 1, "时间": r.get("时间", ""), "来源": r.get("来源", "照片诊断"), "性别": r.get("性别", ""), "年龄": f"{r.get('年龄', '')}岁", "BMI": r.get("BMI", ""), "舌象类型": r.get("舌象类型", ""), "诊断": r.get("诊断", ""), "置信度": f"{r.get('置信度', 0)*100:.1f}%"} for idx, r in enumerate(history)]
    st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)
    
    st.subheader("🔍 回顾详细记录")
    options = [f"{i+1}. {record['时间']} - {record.get('性别','')}{record.get('年龄','')}岁 - {record['舌象类型']}" for i, record in enumerate(history)]
    if options:
        selected_option = st.selectbox("选择要查看的记录", options)
        if selected_option:
            idx = int(selected_option.split(".")[0]) - 1
            r = history[idx]
            c1, c2 = st.columns(2)
            with c1:
                st.info(f"性别：{r.get('性别')} | 年龄：{r.get('年龄')}岁")
                st.info(f"BMI：{r.get('BMI')} ({r.get('体型')})")
                st.info(f"来源：{r.get('来源', '照片诊断')}")
                st.info(f"舌象：{r.get('舌象类型')} | 诊断：{r.get('诊断')}")
                if '舌苔分离分析' in r: st.info(f"细节：{r.get('舌苔分离分析')}")
            with c2:
                if "tongue_region_base64" in r:
                    st.image(Image.open(BytesIO(base64.b64decode(r["tongue_region_base64"]))), width=250)
                if 'coating_img_base64' in r:
                    col_show1, col_show2 = st.columns(2)
                    with col_show1: st.image(Image.open(BytesIO(base64.b64decode(r["coating_img_base64"]))), caption="舌苔(绿)", width=120)
                    with col_show2: st.image(Image.open(BytesIO(base64.b64decode(r["body_img_base64"]))), caption="舌质(红)", width=120)
            with st.expander("🧠 辩证提示"): st.write(r.get("辩证提示"))
            with st.expander("💡 智能建议"): st.write(r.get("智能建议"))
            with st.expander("🔬 详细特征信息"): st.json(r.get("特征信息", {}))

# ---------- 主应用 ----------
def main():
    inject_custom_css()
    init_auth()
    if not st.session_state['logged_in']:
        login_page()
        return
    with st.sidebar:
        st.markdown(f"<h2 style='color:#5E3A2C;'>👤 {st.session_state['username']}</h2>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        page = st.radio("📱 导航菜单", ["👅 舌象分析", "📋 历史记录"], label_visibility="collapsed")
        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("🚪 退出登录", use_container_width=True):
            st.session_state['logged_in'] = False
            st.session_state['username'] = ""
            st.rerun()
    if page == "👅 舌象分析":
        analysis_page()
    else:
        history_page()

if __name__ == '__main__':
    main()
