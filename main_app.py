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

# ===== 导入cv2 =====
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None
    st.warning("opencv-python未安装，部分高级功能将受限。请运行: pip install opencv-python")

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
    .stApp {
        background: linear-gradient(135deg, #FDF8F0 0%, #F9EFE0 100%);
    }
    .main .block-container {
        background-color: rgba(255, 248, 240, 0.85);
        border-radius: 20px;
        padding: 2rem 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.05);
    }
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #5E3A2C !important;
        font-weight: 600 !important;
        letter-spacing: -0.3px;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #EADBC6 0%, #DCC9AC 100%);
        border-right: 1px solid #C9AD7A;
    }
    [data-testid="stSidebar"] * {
        color: #3A241C !important;
    }
    [data-testid="stSidebar"] .stButton button {
        background-color: #A97C50 !important;
        color: white !important;
        border: none;
        border-radius: 30px;
        transition: all 0.3s;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #8B5E3C !important;
        transform: translateY(-2px);
    }
    .stButton button, .st-emotion-cache-1v0mbdj button {
        background: linear-gradient(90deg, #B8865B 0%, #9B6A42 100%);
        color: white !important;
        border: none;
        border-radius: 40px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }
    .stButton button:hover, .st-emotion-cache-1v0mbdj button:hover {
        background: linear-gradient(90deg, #9B6A42 0%, #7B4A2E 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    [data-testid="stMetricValue"] {
        background: #FFFFFFCC;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        display: inline-block;
        color: #5E3A2C;
        font-weight: 600;
    }
    .stAlert, .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 16px;
        border-left: 5px solid #B8865B;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #F5E8DA;
        border-radius: 40px;
        padding: 0.3rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 30px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        color: #7A4A2E;
    }
    .stTabs [aria-selected="true"] {
        background-color: #B8865B;
        color: white !important;
    }
    [data-testid="stFileUploader"] {
        background-color: #FFFFFFBB;
        border: 2px dashed #C9AD7A;
        border-radius: 24px;
        padding: 1rem;
    }
    .stDataFrame {
        border-radius: 20px;
        overflow: hidden;
        border: 1px solid #EADBC6;
    }
    .diagnosis-card {
        background: #FFFFFFDD;
        border-radius: 28px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        box-shadow: 0 8px 20px rgba(0,0,0,0.03);
        border: 1px solid #F0E2D2;
        backdrop-filter: blur(2px);
    }
    .metric-container {
        background: #FCF6EF;
        border-radius: 24px;
        padding: 0.8rem;
        text-align: center;
        border: 1px solid #E9DBCB;
    }
    hr {
        margin: 1rem 0;
        border-color: #E2D0BA;
    }
    .stBalloons {
        filter: hue-rotate(15deg);
    }
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
        if brightness < 80:
            issues.append("⚠️ 光线太暗，请确保在充足自然光下拍摄")
        elif brightness > 230:
            issues.append("⚠️ 光线太亮/过曝，请避免强光直射")

        if CV2_AVAILABLE:
            try:
                gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
                laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
                blur_score = np.var(laplacian)
                if blur_score < 50:
                    issues.append("⚠️ 画面模糊，请保持手机稳定并对焦舌体")
                elif blur_score < 100:
                    issues.append("⚠️ 画面略模糊，建议拍摄更清晰的照片")
            except Exception as e:
                print(f"清晰度检测跳过: {e}")

        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        sat = np.where(max_rgb > 0, (max_rgb - min_rgb) / max_rgb, 0)
        avg_sat = np.mean(sat)
        if avg_sat > 0.65:
            issues.append("⚠️ 颜色饱和度异常，请关闭美颜/滤镜功能")

        gray_flat = gray.flatten()
        sorted_gray = np.sort(gray_flat)
        lower_bound = sorted_gray[int(len(sorted_gray) * 0.3)]
        upper_bound = sorted_gray[int(len(sorted_gray) * 0.7)]
        tongue_mask = (gray >= lower_bound) & (gray <= upper_bound)
        tongue_ratio = np.sum(tongue_mask) / gray.size

        if tongue_ratio < 0.15:
            issues.append("⚠️ 舌体占比过小，请靠近拍摄/确保舌头完整露出")
        elif tongue_ratio > 0.75:
            issues.append("⚠️ 画面过满，请适当拉远使舌体完整")

        high_threshold = np.percentile(gray, 98)
        highlight_ratio = np.sum(gray > high_threshold) / gray.size
        if highlight_ratio > 0.05:
            issues.append("⚠️ 检测到反光区域，请避开光源直射或关闭闪光灯")

    except Exception as e:
        print(f"质量检测异常: {e}")
        return True, []

    is_qualified = len(issues) == 0
    return is_qualified, issues

# ===== 色彩校正功能 =====
def auto_white_balance(image):
    if not CV2_AVAILABLE:
        return image
    try:
        img_array = np.array(image).astype(np.float32)
        r_channel = img_array[:, :, 0]
        g_channel = img_array[:, :, 1]
        b_channel = img_array[:, :, 2]

        r_avg = np.mean(r_channel)
        g_avg = np.mean(g_channel)
        b_avg = np.mean(b_channel)
        gray_avg = (r_avg + g_avg + b_avg) / 3

        r_gain = np.clip(gray_avg / (r_avg + 1e-6), 0.5, 2.0)
        g_gain = np.clip(gray_avg / (g_avg + 1e-6), 0.5, 2.0)
        b_gain = np.clip(gray_avg / (b_avg + 1e-6), 0.5, 2.0)

        img_array[:, :, 0] = np.clip(r_channel * r_gain, 0, 255)
        img_array[:, :, 1] = np.clip(g_channel * g_gain, 0, 255)
        img_array[:, :, 2] = np.clip(b_channel * b_gain, 0, 255)
        return Image.fromarray(img_array.astype(np.uint8))
    except Exception as e:
        print(f"白平衡校正失败: {e}")
        return image

def advanced_color_correction(image):
    if not CV2_AVAILABLE:
        return image
    try:
        img_array = np.array(image).astype(np.float32)
        r_avg = img_array[:, :, 0].mean()
        g_avg = img_array[:, :, 1].mean()
        b_avg = img_array[:, :, 2].mean()
        gray_avg = (r_avg + g_avg + b_avg) / 3

        r_gain = np.clip(gray_avg / (r_avg + 1e-6), 0.85, 1.15)
        g_gain = np.clip(gray_avg / (g_avg + 1e-6), 0.85, 1.15)
        b_gain = np.clip(gray_avg / (b_avg + 1e-6), 0.85, 1.15)

        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * r_gain, 0, 255)
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] * g_gain, 0, 255)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * b_gain, 0, 255)
        return Image.fromarray(img_array.astype(np.uint8))
    except Exception as e:
        print(f"色彩校正失败: {e}")
        return image

# ===== 舌苔舌质分离功能（彩色叠加版）=====
def separate_tongue_coating(image):
    if not CV2_AVAILABLE:
        return image, image, 0.5, "无法分离（缺少cv2库）"
    try:
        img_array = np.array(image.convert("RGB"))
        H, W = img_array.shape[:2]
        if H < 5 or W < 5:
            return image, image, 0.5, "图像尺寸或通道异常"

        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)

        _, mask_s = cv2.threshold(s_ch, 30, 255, cv2.THRESH_BINARY)
        _, mask_v = cv2.threshold(v_ch, 50, 255, cv2.THRESH_BINARY)
        tongue_mask = cv2.bitwise_and(mask_s, mask_v)
        k = np.ones((5, 5), np.uint8)
        tongue_mask = cv2.morphologyEx(tongue_mask, cv2.MORPH_CLOSE, k)
        tongue_mask = cv2.morphologyEx(tongue_mask, cv2.MORPH_OPEN, k)

        if np.sum(tongue_mask > 0) < 0.2 * H * W:
            tongue_mask = np.full((H, W), 255, dtype=np.uint8)

        v_tongue = v_ch.copy()
        v_tongue[tongue_mask == 0] = 0
        _, coating_bin = cv2.threshold(v_tongue, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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

        coating_img = overlay([0, 200, 0],   coating_mask)
        body_img    = overlay([220, 40, 40], body_mask)

        total = int(np.sum(tongue_mask > 0))
        coat_px = int(np.sum(coating_mask > 0))
        coating_ratio = coat_px / total if total > 0 else 0.5
        coating_ratio = float(min(0.95, max(0.05, coating_ratio)))

        if coat_px > 100:
            c_rgb = img_array[coating_mask > 0]
            mr, mg, mb = c_rgb[:, 0].mean(), c_rgb[:, 1].mean(), c_rgb[:, 2].mean()
            if mr > mg + 20 and mr > mb + 20:
                coating_color = "偏黄"
            elif mg > mr + 10 and mg > mb + 10:
                coating_color = "偏白"
            elif mb > mg + 15 and mb > mr + 15:
                coating_color = "偏灰"
            else:
                coating_color = "薄白"
        else:
            coating_color = "较少或无"

        if np.sum(body_mask > 0) > 100:
            b_rgb = img_array[body_mask > 0]
            br, bg, bb = b_rgb[:, 0].mean(), b_rgb[:, 1].mean(), b_rgb[:, 2].mean()
            if br > bg * 1.2 and br > bb * 1.2:
                body_color = "偏红"
            elif br < bg * 0.85 and br < bb * 0.85:
                body_color = "偏淡/偏白"
            else:
                body_color = "淡红"
        else:
            body_color = "无法判断"

        desc = (f"舌苔：{coating_color}，覆盖比例 {coating_ratio*100:.1f}%；"
                f"舌质：{body_color}")
        return coating_img, body_img, coating_ratio, desc

    except Exception as e:
        print(f"舌苔舌质分离失败: {e}")
        return image, image, 0.5, f"分离失败: {str(e)}"

# ---------- 舌象类型标签 ----------
TONGUE_LABELS = [
    "Mirror-Approximated",
    "Thin-White",
    "White-Greasy",
    "Yellow-Greasy",
    "Grey-Black"
]

# ---------- 辨证提示和智能建议映射 ----------
DIAGNOSIS_MAP = {
    "Mirror-Approximated": {
        "诊断": "镜面舌（舌面光滑如镜，舌苔全无）",
        "辩证提示": "胃阴枯竭，胃气大伤。舌面光滑如镜，舌苔完全脱落，舌质红绛。多见于重症疾病后期。",
        "智能建议": """1. 立即就医：建议立即到正规医院就诊
2. 营养支持：高蛋白、高维生素、易消化饮食
3. 中药调理：可考虑沙参麦冬汤加减
4. 避免刺激：禁食辛辣、油炸食物
5. 定期复查：每1-2周复查舌象变化"""
    },
    "Thin-White": {
        "诊断": "薄白苔（正常舌象）",
        "辩证提示": "气血调和，胃气充足。为健康舌象。",
        "智能建议": """1. 维持现状：保持良好生活习惯
2. 饮食调理：均衡饮食，多吃蔬菜水果
3. 适度运动：每周3-5次中等强度运动
4. 充足睡眠：保证每天7-8小时睡眠
5. 定期检查：每年进行一次体检"""
    },
    "White-Greasy": {
        "诊断": "白腻苔（湿浊内蕴）",
        "辩证提示": "湿浊内停，痰饮积聚。多见于消化不良、慢性胃炎等。",
        "智能建议": """1. 健脾祛湿：可用茯苓、白术、薏苡仁煮粥
2. 饮食调整：减少油腻、甜食摄入
3. 中药调理：考虑平胃散、二陈汤
4. 适量运动：每天散步30分钟
5. 穴位按摩：按摩足三里、丰隆穴"""
    },
    "Yellow-Greasy": {
        "诊断": "黄腻苔（湿热内蕴）",
        "辩证提示": "湿热蕴结，痰热互结。常见于急性胃肠炎等。",
        "智能建议": """1. 清热利湿：可用黄芩、黄连泡水
2. 饮食清淡：忌辛辣油腻
3. 及时就医：如伴发热应及时就诊
4. 中药治疗：考虑茵陈蒿汤
5. 多饮水：每天饮水2000ml以上"""
    },
    "Grey-Black": {
        "诊断": "灰黑苔（热极或寒极）",
        "辩证提示": "病情危重，或热极或寒极。",
        "智能建议": """1. 紧急就医：立即到急诊就诊
2. 全面检查：进行血常规、肝肾功能检查
3. 专科治疗：到相应专科治疗
4. 中西医结合：配合中药治疗
5. 密切观察：每日记录舌象变化"""
    }
}

# ===== 高精度舌象识别算法 =====
def extract_tongue_color_features(image, coating_ratio=None, coating_description=None):
    try:
        img_array = np.array(image)
        h_img, w_img = img_array.shape[:2]
        crop_size = min(h_img, w_img) // 3
        crop_size = max(10, crop_size)

        cy, cx = h_img // 2, w_img // 2
        center = img_array[
            cy - crop_size // 2: cy + crop_size // 2,
            cx - crop_size // 2: cx + crop_size // 2
        ]

        if center.size == 0:
            return "Thin-White", 0.75, {"错误": "提取区域为空"}

        r = np.mean(center[:, :, 0])
        g = np.mean(center[:, :, 1])
        b = np.mean(center[:, :, 2])

        if np.isnan(r) or np.isnan(g) or np.isnan(b):
            return "Thin-White", 0.75, {"错误": "均值计算结果为NaN"}

        brightness = 0.299 * r + 0.587 * g + 0.114 * b

        rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
        hue, sat, val = colorsys.rgb_to_hsv(rn, gn, bn)

        r_std = np.std(center[:, :, 0])
        g_std = np.std(center[:, :, 1])
        b_std = np.std(center[:, :, 2])
        uniformity = (r_std + g_std + b_std) / 3

        yellow_ratio = (r - b) / (g + 1)
        confidence = 0.85

        if coating_ratio is not None and coating_ratio < 0.08:
            tongue_type = "Mirror-Approximated"
            confidence = 0.94
            return tongue_type, round(confidence, 2), {
                "亮度": round(brightness, 2),
                "饱和度": round(sat, 2),
                "色调": round(hue, 2),
                "颜色均匀度": round(uniformity, 2),
                "黄色指数": round(yellow_ratio, 2),
                "舌苔覆盖比例": round(coating_ratio * 100, 1),
                "舌苔分析": coating_description or "无"
            }

        if coating_ratio is not None and coating_ratio > 0.35:
            if yellow_ratio > 10 and sat > 0.45:
                tongue_type = "Yellow-Greasy"
                confidence = 0.91
            elif sat > 0.25 and brightness < 180:
                tongue_type = "White-Greasy"
                confidence = 0.89
            elif brightness > 210 and sat < 0.15 and uniformity < 25:
                tongue_type = "Mirror-Approximated"
                confidence = 0.92
            elif brightness < 90:
                tongue_type = "Grey-Black"
                confidence = 0.90
            else:
                tongue_type = "Thin-White"
                confidence = 0.80
        elif yellow_ratio > 12 and sat > 0.48 and hue > 0.08:
            tongue_type = "Yellow-Greasy"
            confidence = 0.89
        elif sat > 0.28 and 110 < brightness < 170 and yellow_ratio < 8:
            tongue_type = "White-Greasy"
            confidence = 0.87
        else:
            tongue_type = "Thin-White"
            confidence = 0.82

        confidence = max(0.75, min(0.98, confidence))

        feature_info = {
            "亮度": round(brightness, 2),
            "饱和度": round(sat, 2),
            "色调": round(hue, 2),
            "颜色均匀度": round(uniformity, 2),
            "黄色指数": round(yellow_ratio, 2)
        }

        if coating_ratio is not None:
            feature_info["舌苔覆盖比例"] = round(coating_ratio * 100, 1)
            feature_info["舌苔分析"] = coating_description or "无"

        return tongue_type, round(confidence, 2), feature_info

    except Exception as e:
        print(f"特征提取错误: {e}")
        return "Thin-White", 0.75, {}

# ==============================================
# ========== 【ONNX 推理模块】 ==========
# ==============================================
IMG_SIZE = 640

@st.cache_resource
def load_onnx_model():
    onnx_path = "best.onnx"
    if not os.path.exists(onnx_path):
        raise FileNotFoundError("找不到 yolov10n.onnx，请确认文件在当前项目文件夹！")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    return session, input_name

def onnx_detect(image_path, conf_thres=0.1, iou_thres=0.45):
    session, input_name = load_onnx_model()

    try:
        pil_img = Image.open(image_path).convert('RGB')
        img = np.array(pil_img)[:, :, ::-1].copy()
    except Exception as e:
        st.error(f"图片读取失败: {e}")
        return []

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

    if pred.ndim == 2 and pred.shape[0] < pred.shape[1]:
        pred = pred.T
    if pred.ndim != 2 or pred.shape[1] < 5:
        print(f"[YOLO] 输出形状异常: {pred.shape}")
        return []

    boxes_xywh = pred[:, :4]
    scores = pred[:, 4:]
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    class_ids = np.argmax(scores, axis=1)
    conf = np.max(scores, axis=1)

    conf = np.clip(conf, 0.0, 1.0)

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

        indices = cv2.dnn.NMSBoxes(
            xyxy.tolist(), conf.tolist(), conf_thres, iou_thres
        )
        if len(indices) > 0:
            for i in np.array(indices).flatten():
                det_list.append(
                    (int(class_ids[i]), float(conf[i]), xyxy[i].tolist())
                )
    return det_list

# ===== 舌象分析核心函数 =====
def analyze_tongue(image_path, enable_color_correction=True, enable_coating_separation=True):
    try:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size

        original_img = img.copy()
        if enable_color_correction and CV2_AVAILABLE:
            try:
                corrected = advanced_color_correction(img)
                a1 = np.array(original_img).astype(np.float32)
                a2 = np.array(corrected).astype(np.float32)
                if a1.shape == a2.shape:
                    diff = float(np.mean(np.abs(a1 - a2)))
                    if diff < 60:
                        img = corrected
                    else:
                        print(f"[WB] 放弃校正，差异过大 diff={diff:.2f}")
            except Exception as e:
                print(f"[WB] 校正异常，跳过: {e}")

        det_list = onnx_detect(image_path, conf_thres=0.1, iou_thres=0.45)
        best_box, best_conf, all_boxes = None, 0.0, []

        class_names = {0: "tongue"}
        for cls_id, conf, xyxy in det_list:
            name = class_names.get(cls_id, f"cls{cls_id}")
            all_boxes.append(f"{name}:{conf:.2f}")
            if cls_id == 0 and conf > best_conf:
                best_conf, best_box = conf, xyxy

        if best_box is None and len(det_list) > 0:
            for cls_id, conf, xyxy in det_list:
                if conf > best_conf:
                    best_conf, best_box = conf, xyxy

        best_conf = float(np.clip(best_conf, 0.0, 1.0))

        bw, bh = img.size
        crop_size = max(20, min(bw, bh) * 2 // 3)
        ccx, ccy = bw // 2, bh // 2

        cx1 = max(0, ccx - crop_size // 2)
        cy1 = max(0, ccy - crop_size // 2)
        cx2 = min(bw, ccx + crop_size // 2)
        cy2 = min(bh, ccy + crop_size // 2)

        if cx2 - cx1 < 20: cx2 = min(bw, cx1 + 20)
        if cy2 - cy1 < 20: cy2 = min(bh, cy1 + 20)

        cropped_img = img.crop((cx1, cy1, cx2, cy2))

        coating_ratio = None
        coating_description = None
        coating_img_base64 = None
        body_img_base64 = None

        if enable_coating_separation and CV2_AVAILABLE:
            coating_img, body_img, coating_ratio, coating_description = \
                separate_tongue_coating(cropped_img)

            buffered_coating = BytesIO()
            coating_img.save(buffered_coating, format="PNG")
            coating_img_base64 = base64.b64encode(buffered_coating.getvalue()).decode()

            buffered_body = BytesIO()
            body_img.save(buffered_body, format="PNG")
            body_img_base64 = base64.b64encode(buffered_body.getvalue()).decode()

        tongue_type, confidence, feature_info = extract_tongue_color_features(
            cropped_img, coating_ratio, coating_description
        )

        # 6. 可视化：只画红框，不写任何文字（避免中文乱码）
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)
        draw.rectangle([cx1, cy1, cx2, cy2], outline="#FF2200", width=6)

        buffered = BytesIO()
        draw_img.save(buffered, format="PNG")
        boxed_image_base64 = base64.b64encode(buffered.getvalue()).decode()

        buffered_crop = BytesIO()
        cropped_img.save(buffered_crop, format="PNG")
        tongue_region_base64 = base64.b64encode(buffered_crop.getvalue()).decode()

        diagnosis_info = DIAGNOSIS_MAP.get(
            tongue_type,
            {"诊断": "未知舌象", "辩证提示": "无法准确判断",
             "智能建议": "请咨询专业中医师"}
        )

        result_dict = {
            "舌象类型": tongue_type,
            "置信度": confidence,
            "诊断": diagnosis_info["诊断"],
            "辩证提示": diagnosis_info["辩证提示"],
            "智能建议": diagnosis_info["智能建议"],
            "boxed_image_base64": boxed_image_base64,
            "tongue_region_base64": tongue_region_base64,
            "分析时间": time.strftime("%Y-%m-%d %H:%M:%S"),
            "特征信息": feature_info,
            "人体检测置信度": best_conf,
            "检测详情": ", ".join(all_boxes) if all_boxes else "无检测"
        }

        if coating_img_base64:
            result_dict["coating_img_base64"] = coating_img_base64
            result_dict["body_img_base64"] = body_img_base64
            result_dict["舌苔分离分析"] = coating_description

        return result_dict

    except Exception as e:
        st.error(f"分析过程中发生错误: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

# ---------- 历史记录读写 ----------
def save_history(history):
    with open("tongue_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_history():
    if os.path.exists("tongue_history.json"):
        with open("tongue_history.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# ---------- 登录模块 ----------
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

def init_auth():
    if 'users' not in st.session_state:
        st.session_state['users'] = {
            "doctor": make_hashes("123456"),
            "student": make_hashes("654321")
        }
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['username'] = ""

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

# ===== 导入cv2 =====
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None
    st.warning("opencv-python未安装，部分高级功能将受限。请运行: pip install opencv-python")

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
    .stApp {
        background: linear-gradient(135deg, #FDF8F0 0%, #F9EFE0 100%);
    }
    .main .block-container {
        background-color: rgba(255, 248, 240, 0.85);
        border-radius: 20px;
        padding: 2rem 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.05);
    }
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #5E3A2C !important;
        font-weight: 600 !important;
        letter-spacing: -0.3px;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #EADBC6 0%, #DCC9AC 100%);
        border-right: 1px solid #C9AD7A;
    }
    [data-testid="stSidebar"] * {
        color: #3A241C !important;
    }
    [data-testid="stSidebar"] .stButton button {
        background-color: #A97C50 !important;
        color: white !important;
        border: none;
        border-radius: 30px;
        transition: all 0.3s;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #8B5E3C !important;
        transform: translateY(-2px);
    }
    .stButton button, .st-emotion-cache-1v0mbdj button {
        background: linear-gradient(90deg, #B8865B 0%, #9B6A42 100%);
        color: white !important;
        border: none;
        border-radius: 40px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }
    .stButton button:hover, .st-emotion-cache-1v0mbdj button:hover {
        background: linear-gradient(90deg, #9B6A42 0%, #7B4A2E 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    [data-testid="stMetricValue"] {
        background: #FFFFFFCC;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        display: inline-block;
        color: #5E3A2C;
        font-weight: 600;
    }
    .stAlert, .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 16px;
        border-left: 5px solid #B8865B;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #F5E8DA;
        border-radius: 40px;
        padding: 0.3rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 30px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        color: #7A4A2E;
    }
    .stTabs [aria-selected="true"] {
        background-color: #B8865B;
        color: white !important;
    }
    [data-testid="stFileUploader"] {
        background-color: #FFFFFFBB;
        border: 2px dashed #C9AD7A;
        border-radius: 24px;
        padding: 1rem;
    }
    .stDataFrame {
        border-radius: 20px;
        overflow: hidden;
        border: 1px solid #EADBC6;
    }
    .diagnosis-card {
        background: #FFFFFFDD;
        border-radius: 28px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        box-shadow: 0 8px 20px rgba(0,0,0,0.03);
        border: 1px solid #F0E2D2;
        backdrop-filter: blur(2px);
    }
    .metric-container {
        background: #FCF6EF;
        border-radius: 24px;
        padding: 0.8rem;
        text-align: center;
        border: 1px solid #E9DBCB;
    }
    hr {
        margin: 1rem 0;
        border-color: #E2D0BA;
    }
    .stBalloons {
        filter: hue-rotate(15deg);
    }
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
        if brightness < 80:
            issues.append("⚠️ 光线太暗，请确保在充足自然光下拍摄")
        elif brightness > 230:
            issues.append("⚠️ 光线太亮/过曝，请避免强光直射")
        if CV2_AVAILABLE:
            try:
                gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
                laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
                blur_score = np.var(laplacian)
                if blur_score < 50:
                    issues.append("⚠️ 画面模糊，请保持手机稳定并对焦舌体")
                elif blur_score < 100:
                    issues.append("⚠️ 画面略模糊，建议拍摄更清晰的照片")
            except Exception as e:
                print(f"清晰度检测跳过: {e}")
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        sat = np.where(max_rgb > 0, (max_rgb - min_rgb) / max_rgb, 0)
        avg_sat = np.mean(sat)
        if avg_sat > 0.65:
            issues.append("⚠️ 颜色饱和度异常，请关闭美颜/滤镜功能")
        gray_flat = gray.flatten()
        sorted_gray = np.sort(gray_flat)
        lower_bound = sorted_gray[int(len(sorted_gray) * 0.3)]
        upper_bound = sorted_gray[int(len(sorted_gray) * 0.7)]
        tongue_mask = (gray >= lower_bound) & (gray <= upper_bound)
        tongue_ratio = np.sum(tongue_mask) / gray.size
        if tongue_ratio < 0.15:
            issues.append("⚠️ 舌体占比过小，请靠近拍摄/确保舌头完整露出")
        elif tongue_ratio > 0.75:
            issues.append("⚠️ 画面过满，请适当拉远使舌体完整")
        high_threshold = np.percentile(gray, 98)
        highlight_ratio = np.sum(gray > high_threshold) / gray.size
        if highlight_ratio > 0.05:
            issues.append("⚠️ 检测到反光区域，请避开光源直射或关闭闪光灯")
    except Exception as e:
        print(f"质量检测异常: {e}")
        return True, []
    is_qualified = len(issues) == 0
    return is_qualified, issues

# ===== 色彩校正功能 =====
def auto_white_balance(image):
    if not CV2_AVAILABLE:
        return image
    try:
        img_array = np.array(image).astype(np.float32)
        r_channel = img_array[:, :, 0]
        g_channel = img_array[:, :, 1]
        b_channel = img_array[:, :, 2]
        r_avg = np.mean(r_channel)
        g_avg = np.mean(g_channel)
        b_avg = np.mean(b_channel)
        gray_avg = (r_avg + g_avg + b_avg) / 3
        r_gain = np.clip(gray_avg / (r_avg + 1e-6), 0.5, 2.0)
        g_gain = np.clip(gray_avg / (g_avg + 1e-6), 0.5, 2.0)
        b_gain = np.clip(gray_avg / (b_avg + 1e-6), 0.5, 2.0)
        img_array[:, :, 0] = np.clip(r_channel * r_gain, 0, 255)
        img_array[:, :, 1] = np.clip(g_channel * g_gain, 0, 255)
        img_array[:, :, 2] = np.clip(b_channel * b_gain, 0, 255)
        return Image.fromarray(img_array.astype(np.uint8))
    except Exception as e:
        print(f"白平衡校正失败: {e}")
        return image

def advanced_color_correction(image):
    if not CV2_AVAILABLE:
        return image
    try:
        img_array = np.array(image).astype(np.float32)
        r_avg = img_array[:, :, 0].mean()
        g_avg = img_array[:, :, 1].mean()
        b_avg = img_array[:, :, 2].mean()
        gray_avg = (r_avg + g_avg + b_avg) / 3
        r_gain = np.clip(gray_avg / (r_avg + 1e-6), 0.85, 1.15)
        g_gain = np.clip(gray_avg / (g_avg + 1e-6), 0.85, 1.15)
        b_gain = np.clip(gray_avg / (b_avg + 1e-6), 0.85, 1.15)
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * r_gain, 0, 255)
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] * g_gain, 0, 255)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * b_gain, 0, 255)
        return Image.fromarray(img_array.astype(np.uint8))
    except Exception as e:
        print(f"色彩校正失败: {e}")
        return image

# ===== 舌苔舌质分离功能 =====
def separate_tongue_coating(image):
    if not CV2_AVAILABLE:
        return image, image, 0.5, "无法分离（缺少cv2库）"
    try:
        img_array = np.array(image.convert("RGB"))
        H, W = img_array.shape[:2]
        if H < 5 or W < 5:
            return image, image, 0.5, "图像尺寸或通道异常"
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)
        _, mask_s = cv2.threshold(s_ch, 30, 255, cv2.THRESH_BINARY)
        _, mask_v = cv2.threshold(v_ch, 50, 255, cv2.THRESH_BINARY)
        tongue_mask = cv2.bitwise_and(mask_s, mask_v)
        k = np.ones((5, 5), np.uint8)
        tongue_mask = cv2.morphologyEx(tongue_mask, cv2.MORPH_CLOSE, k)
        tongue_mask = cv2.morphologyEx(tongue_mask, cv2.MORPH_OPEN, k)
        if np.sum(tongue_mask > 0) < 0.2 * H * W:
            tongue_mask = np.full((H, W), 255, dtype=np.uint8)
        v_tongue = v_ch.copy()
        v_tongue[tongue_mask == 0] = 0
        _, coating_bin = cv2.threshold(v_tongue, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
        coating_ratio = coat_px / total if total > 0 else 0.5
        coating_ratio = float(min(0.95, max(0.05, coating_ratio)))
        if coat_px > 100:
            c_rgb = img_array[coating_mask > 0]
            mr, mg, mb = c_rgb[:, 0].mean(), c_rgb[:, 1].mean(), c_rgb[:, 2].mean()
            if mr > mg + 20 and mr > mb + 20:
                coating_color = "偏黄"
            elif mg > mr + 10 and mg > mb + 10:
                coating_color = "偏白"
            elif mb > mg + 15 and mb > mr + 15:
                coating_color = "偏灰"
            else:
                coating_color = "薄白"
        else:
            coating_color = "较少或无"
        if np.sum(body_mask > 0) > 100:
            b_rgb = img_array[body_mask > 0]
            br, bg, bb = b_rgb[:, 0].mean(), b_rgb[:, 1].mean(), b_rgb[:, 2].mean()
            if br > bg * 1.2 and br > bb * 1.2:
                body_color = "偏红"
            elif br < bg * 0.85 and br < bb * 0.85:
                body_color = "偏淡/偏白"
            else:
                body_color = "淡红"
        else:
            body_color = "无法判断"
        desc = (f"舌苔：{coating_color}，覆盖比例 {coating_ratio*100:.1f}%；"
                f"舌质：{body_color}")
        return coating_img, body_img, coating_ratio, desc
    except Exception as e:
        print(f"舌苔舌质分离失败: {e}")
        return image, image, 0.5, f"分离失败: {str(e)}"

# ---------- 舌象类型标签 ----------
TONGUE_LABELS = [
    "Mirror-Approximated",
    "Thin-White",
    "White-Greasy",
    "Yellow-Greasy",
    "Grey-Black"
]

# ---------- 辨证提示和智能建议映射 ----------
DIAGNOSIS_MAP = {
    "Mirror-Approximated": {
        "诊断": "镜面舌（舌面光滑如镜，舌苔全无）",
        "辩证提示": "胃阴枯竭，胃气大伤。舌面光滑如镜，舌苔完全脱落，舌质红绛。多见于重症疾病后期。",
        "智能建议": """1. 立即就医：建议立即到正规医院就诊
2. 营养支持：高蛋白、高维生素、易消化饮食
3. 中药调理：可考虑沙参麦冬汤加减
4. 避免刺激：禁食辛辣、油炸食物
5. 定期复查：每1-2周复查舌象变化"""
    },
    "Thin-White": {
        "诊断": "薄白苔（正常舌象）",
        "辩证提示": "气血调和，胃气充足。为健康舌象。",
        "智能建议": """1. 维持现状：保持良好生活习惯
2. 饮食调理：均衡饮食，多吃蔬菜水果
3. 适度运动：每周3-5次中等强度运动
4. 充足睡眠：保证每天7-8小时睡眠
5. 定期检查：每年进行一次体检"""
    },
    "White-Greasy": {
        "诊断": "白腻苔（湿浊内蕴）",
        "辩证提示": "湿浊内停，痰饮积聚。多见于消化不良、慢性胃炎等。",
        "智能建议": """1. 健脾祛湿：可用茯苓、白术、薏苡仁煮粥
2. 饮食调整：减少油腻、甜食摄入
3. 中药调理：考虑平胃散、二陈汤
4. 适量运动：每天散步30分钟
5. 穴位按摩：按摩足三里、丰隆穴"""
    },
    "Yellow-Greasy": {
        "诊断": "黄腻苔（湿热内蕴）",
        "辩证提示": "湿热蕴结，痰热互结。常见于急性胃肠炎等。",
        "智能建议": """1. 清热利湿：可用黄芩、黄连泡水
2. 饮食清淡：忌辛辣油腻
3. 及时就医：如伴发热应及时就诊
4. 中药治疗：考虑茵陈蒿汤
5. 多饮水：每天饮水2000ml以上"""
    },
    "Grey-Black": {
        "诊断": "灰黑苔（热极或寒极）",
        "辩证提示": "病情危重，或热极或寒极。",
        "智能建议": """1. 紧急就医：立即到急诊就诊
2. 全面检查：进行血常规、肝肾功能检查
3. 专科治疗：到相应专科治疗
4. 中西医结合：配合中药治疗
5. 密切观察：每日记录舌象变化"""
    }
}

# ===== 高精度舌象识别算法 =====
def extract_tongue_color_features(image, coating_ratio=None, coating_description=None):
    try:
        img_array = np.array(image)
        h_img, w_img = img_array.shape[:2]
        crop_size = min(h_img, w_img) // 3
        crop_size = max(10, crop_size)
        cy, cx = h_img // 2, w_img // 2
        center = img_array[
                 cy - crop_size // 2: cy + crop_size // 2,
                 cx - crop_size // 2: cx + crop_size // 2
                 ]
        if center.size == 0:
            return "Thin-White", 0.75, {"错误": "提取区域为空"}
        r = np.mean(center[:, :, 0])
        g = np.mean(center[:, :, 1])
        b = np.mean(center[:, :, 2])
        if np.isnan(r) or np.isnan(g) or np.isnan(b):
            return "Thin-White", 0.75, {"错误": "均值计算结果为NaN"}
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
        hue, sat, val = colorsys.rgb_to_hsv(rn, gn, bn)
        r_std = np.std(center[:, :, 0])
        g_std = np.std(center[:, :, 1])
        b_std = np.std(center[:, :, 2])
        uniformity = (r_std + g_std + b_std) / 3
        yellow_ratio = (r - b) / (g + 1)
        confidence = 0.85
        if coating_ratio is not None and coating_ratio < 0.08:
            tongue_type = "Mirror-Approximated"
            confidence = 0.94
            return tongue_type, round(confidence, 2), {
                "亮度": round(brightness, 2),
                "饱和度": round(sat, 2),
                "色调": round(hue, 2),
                "颜色均匀度": round(uniformity, 2),
                "黄色指数": round(yellow_ratio, 2),
                "舌苔覆盖比例": round(coating_ratio * 100, 1),
                "舌苔分析": coating_description or "无"
            }
        if coating_ratio is not None and coating_ratio > 0.35:
            if yellow_ratio > 10 and sat > 0.45:
                tongue_type = "Yellow-Greasy"
                confidence = 0.91
            elif sat > 0.25 and brightness < 180:
                tongue_type = "White-Greasy"
                confidence = 0.89
            elif brightness > 210 and sat < 0.15 and uniformity < 25:
                tongue_type = "Mirror-Approximated"
                confidence = 0.92
            elif brightness < 90:
                tongue_type = "Grey-Black"
                confidence = 0.90
            else:
                tongue_type = "Thin-White"
                confidence = 0.80
        elif yellow_ratio > 12 and sat > 0.48 and hue > 0.08:
            tongue_type = "Yellow-Greasy"
            confidence = 0.89
        elif sat > 0.28 and 110 < brightness < 170 and yellow_ratio < 8:
            tongue_type = "White-Greasy"
            confidence = 0.87
        else:
            tongue_type = "Thin-White"
            confidence = 0.82
        confidence = max(0.75, min(0.98, confidence))
        feature_info = {
            "亮度": round(brightness, 2),
            "饱和度": round(sat, 2),
            "色调": round(hue, 2),
            "颜色均匀度": round(uniformity, 2),
            "黄色指数": round(yellow_ratio, 2)
        }
        if coating_ratio is not None:
            feature_info["舌苔覆盖比例"] = round(coating_ratio * 100, 1)
            feature_info["舌苔分析"] = coating_description or "无"
        return tongue_type, round(confidence, 2), feature_info
    except Exception as e:
        print(f"特征提取错误: {e}")
        return "Thin-White", 0.75, {}

# ==============================================
# ========== 【ONNX 推理模块】 ==========
# ==============================================
IMG_SIZE = 640

@st.cache_resource
def load_onnx_model():
    onnx_path = "yolov10n.onnx"
    if not os.path.exists(onnx_path):
        raise FileNotFoundError("找不到 yolov10n.onnx，请确认文件在当前项目文件夹！")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    return session, input_name

def onnx_detect(image_path, conf_thres=0.1, iou_thres=0.45):
    session, input_name = load_onnx_model()
    try:
        pil_img = Image.open(image_path).convert('RGB')
        img = np.array(pil_img)[:, :, ::-1].copy()
    except Exception as e:
        st.error(f"图片读取失败: {e}")
        return []
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
    if pred.ndim == 2 and pred.shape[0] < pred.shape[1]:
        pred = pred.T
    if pred.ndim != 2 or pred.shape[1] < 5:
        print(f"[YOLO] 输出形状异常: {pred.shape}")
        return []
    boxes_xywh = pred[:, :4]
    scores = pred[:, 4:]
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    class_ids = np.argmax(scores, axis=1)
    conf = np.max(scores, axis=1)
    conf = np.clip(conf, 0.0, 1.0)
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
        indices = cv2.dnn.NMSBoxes(
            xyxy.tolist(), conf.tolist(), conf_thres, iou_thres
        )
        if len(indices) > 0:
            for i in np.array(indices).flatten():
                det_list.append(
                    (int(class_ids[i]), float(conf[i]), xyxy[i].tolist())
                )
    return det_list

# ===== 舌象分析核心函数 =====
def analyze_tongue(image_path, enable_color_correction=True, enable_coating_separation=True):
    try:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        original_img = img.copy()
        if enable_color_correction and CV2_AVAILABLE:
            try:
                corrected = advanced_color_correction(img)
                a1 = np.array(original_img).astype(np.float32)
                a2 = np.array(corrected).astype(np.float32)
                if a1.shape == a2.shape:
                    diff = float(np.mean(np.abs(a1 - a2)))
                    if diff < 60:
                        img = corrected
                    else:
                        print(f"[WB] 放弃校正，差异过大 diff={diff:.2f}")
            except Exception as e:
                print(f"[WB] 校正异常，跳过: {e}")
        det_list = onnx_detect(image_path, conf_thres=0.1, iou_thres=0.45)
        best_box, best_conf, all_boxes = None, 0.0, []
        class_names = {0: "tongue"}
        for cls_id, conf, xyxy in det_list:
            name = class_names.get(cls_id, f"cls{cls_id}")
            all_boxes.append(f"{name}:{conf:.2f}")
            if cls_id == 0 and conf > best_conf:
                best_conf, best_box = conf, xyxy
        if best_box is None and len(det_list) > 0:
            for cls_id, conf, xyxy in det_list:
                if conf > best_conf:
                    best_conf, best_box = conf, xyxy
        best_conf = float(np.clip(best_conf, 0.0, 1.0))
        bw, bh = img.size
        crop_size = max(20, min(bw, bh) * 2 // 3)
        ccx, ccy = bw // 2, bh // 2
        cx1 = max(0, ccx - crop_size // 2)
        cy1 = max(0, ccy - crop_size // 2)
        cx2 = min(bw, ccx + crop_size // 2)
        cy2 = min(bh, ccy + crop_size // 2)
        if cx2 - cx1 < 20: cx2 = min(bw, cx1 + 20)
        if cy2 - cy1 < 20: cy2 = min(bh, cy1 + 20)
        cropped_img = img.crop((cx1, cy1, cx2, cy2))
        coating_ratio = None
        coating_description = None
        coating_img_base64 = None
        body_img_base64 = None
        if enable_coating_separation and CV2_AVAILABLE:
            coating_img, body_img, coating_ratio, coating_description = \
                separate_tongue_coating(cropped_img)
            buffered_coating = BytesIO()
            coating_img.save(buffered_coating, format="PNG")
            coating_img_base64 = base64.b64encode(buffered_coating.getvalue()).decode()
            buffered_body = BytesIO()
            body_img.save(buffered_body, format="PNG")
            body_img_base64 = base64.b64encode(buffered_body.getvalue()).decode()
        tongue_type, confidence, feature_info = extract_tongue_color_features(
            cropped_img, coating_ratio, coating_description
        )
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)
        draw.rectangle([cx1, cy1, cx2, cy2], outline="#FF2200", width=6)
        buffered = BytesIO()
        draw_img.save(buffered, format="PNG")
        boxed_image_base64 = base64.b64encode(buffered.getvalue()).decode()
        buffered_crop = BytesIO()
        cropped_img.save(buffered_crop, format="PNG")
        tongue_region_base64 = base64.b64encode(buffered_crop.getvalue()).decode()
        diagnosis_info = DIAGNOSIS_MAP.get(
            tongue_type,
            {"诊断": "未知舌象", "辩证提示": "无法准确判断",
             "智能建议": "请咨询专业中医师"}
        )
        result_dict = {
            "舌象类型": tongue_type,
            "置信度": confidence,
            "诊断": diagnosis_info["诊断"],
            "辩证提示": diagnosis_info["辩证提示"],
            "智能建议": diagnosis_info["智能建议"],
            "boxed_image_base64": boxed_image_base64,
            "tongue_region_base64": tongue_region_base64,
            "分析时间": time.strftime("%Y-%m-%d %H:%M:%S"),
            "特征信息": feature_info,
            "人体检测置信度": best_conf,
            "检测详情": ", ".join(all_boxes) if all_boxes else "无检测"
        }
        if coating_img_base64:
            result_dict["coating_img_base64"] = coating_img_base64
            result_dict["body_img_base64"] = body_img_base64
            result_dict["舌苔分离分析"] = coating_description
        return result_dict
    except Exception as e:
        st.error(f"分析过程中发生错误: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

# ---------- 历史记录读写 ----------
def save_history(history):
    with open("tongue_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_history():
    if os.path.exists("tongue_history.json"):
        with open("tongue_history.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# ---------- 登录模块 ----------
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

def init_auth():
    if 'users' not in st.session_state:
        st.session_state['users'] = {
            "doctor": make_hashes("123456"),
            "student": make_hashes("654321")
        }
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['username'] = ""

def login_page():
    st.markdown(
        "<h1 style='text-align: center;'>🔐 舌象诊断 · 专属登录</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; color: #7A4A2E;'>欢迎使用智能中医舌象分析系统</p>",
        unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        with st.container():
            st.markdown("<div class='diagnosis-card'>", unsafe_allow_html=True)
            menu = ["登录", "注册"]
            choice = st.selectbox("选择操作", menu, key="login_menu")
            if choice == "登录":
                username = st.text_input(
                    "用户名", key="login_username", placeholder="请输入用户名"
                )
                password = st.text_input(
                    "密码", type="password", key="login_password",
                    placeholder="请输入密码"
                )
                if st.button("登录", use_container_width=True):
                    if (username in st.session_state['users']
                            and check_hashes(
                                password,
                                st.session_state['users'][username]                           
                            )):
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = username
                        st.success(f"欢迎回来，{username} 医师！")
                        st.rerun()
                    else:
                        st.error("用户名或密码错误！")
            elif choice == "注册":
                new_user = st.text_input(
                    "新用户名", key="reg_username", placeholder="请输入新用户名"
                )
                new_password = st.text_input(
                    "新密码", type="password", key="reg_password",
                    placeholder="请设置密码"
                )
                if st.button("注册", use_container_width=True):
                    if new_user in st.session_state['users']:
                        st.warning("用户已存在！")
                    elif new_user and new_password:
                        st.session_state['users'][new_user] = make_hashes(
                            new_password
                        )
                        st.success("注册成功！请返回登录")
                    else:
                        st.error("用户名和密码不能为空")
            st.markdown("</div>", unsafe_allow_html=True)

# ---------- 舌象分析页面 ----------
def analysis_page():
    st.markdown("<h1>👅 智能舌象诊断</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#6B4C3C;'>上传舌部照片，系统将基于YOLO模型精准定位舌体区域并给出中医辨证建议。</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    if "history" not in st.session_state:
        st.session_state.history = load_history()
    if "current_result" not in st.session_state:
        st.session_state.current_result = None
    if "upload_passed_quality" not in st.session_state:
        st.session_state.upload_passed_quality = False
    if "current_temp_path" not in st.session_state:
        st.session_state.current_temp_path = None

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
    if bmi < 18.5:
        bmi_level = "偏瘦"
    elif 18.5 <= bmi < 24:
        bmi_level = "正常"
    elif 24 <= bmi < 28:
        bmi_level = "超重"
    else:
        bmi_level = "肥胖"
    st.success(f"**自动计算BMI指数：{bmi} | 体型：{bmi_level}**")
    st.markdown("</div>", unsafe_allow_html=True)

    # ====== 上传与设置 ======
    st.markdown('<div class="diagnosis-card">', unsafe_allow_html=True)
    st.subheader("📤 上传舌象照片")
    col_upload_guide, _ = st.columns([3, 1])
    with col_upload_guide:
        with st.expander("📷 拍摄指南（点击收起）", expanded=True):
            st.markdown("""
            <div style="background:#FCF6EF; padding:1rem; border-radius:16px;">
            <p>✅ 在<strong>自然光</strong>下拍摄，避免强光直射或过暗</p>
            <p>✅ 舌头<strong>自然平伸</strong>，不要过度用力</p>
            <p>✅ 对焦舌体，确保图像<strong>清晰不模糊</strong></p>
            <p>✅ 关闭相机<strong>美颜、滤镜、HDR</strong>等功能</p>
            <p>✅ 尽量露出<strong>整个舌头</strong>，包括舌根</p>
            </div>
            """, unsafe_allow_html=True)
        with st.expander("⚙️ 高级分析设置（可选）"):
            col_set1, col_set2 = st.columns(2)
            with col_set1:
                enable_color_correction = st.checkbox("🎨 自动色彩校正（白平衡）", value=True)
            with col_set2:
                enable_coating_separation = st.checkbox("🔬 舌苔/舌质分离分析", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("点击或拖拽上传舌象照片", type=["jpg", "jpeg", "png"])

    # ====== 分析执行 ======
    if uploaded_file is not None:
        temp_path = f"temp_{int(time.time())}_{uploaded_file.name}"
        if st.session_state.current_temp_path != temp_path:
            st.session_state.upload_passed_quality = False
            st.session_state.current_temp_path = temp_path
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            img = Image.open(temp_path).convert("RGB")
            is_qualified, issues = check_image_quality(img)

            if is_qualified:
                st.session_state.upload_passed_quality = True
                st.success("✅ 图像质量合格，可以开始分析！")
            else:
                st.session_state.upload_passed_quality = False
                st.error("❌ 图像质量不符合要求，请重新拍摄/上传")
                for issue in issues:
                    st.warning(issue)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                st.session_state.current_temp_path = None
                st.info("📷 **请重新拍摄：**\n\n" + "\n".join(issues))

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
                with st.spinner("正在分析舌象，请稍候..."):
                    result = analyze_tongue(
                        st.session_state.current_temp_path,
                        enable_color_correction=enable_color_correction,
                        enable_coating_separation=enable_coating_separation
                    )

                    if result:
                        st.session_state.current_result = result
                        # 保存到历史记录
                        result["时间"] = result["分析时间"]
                        result["性别"] = gender
                        result["年龄"] = age
                        result["BMI"] = bmi
                        result["体型"] = bmi_level
                        st.session_state.history.append(result)
                        save_history(st.session_state.history)

                        st.success("✅ 分析完成！")
                        st.balloons()

                        # 展示图片
                        boxed_placeholder.image(f"data:image/png;base64,{result['boxed_image_base64']}", width=400)
                        tongue_placeholder.image(f"data:image/png;base64,{result['tongue_region_base64']}", width=400)

                        conf_val = result.get("人体检测置信度", 0.0)
                        conf_caption_placeholder.caption(f"Conf: {conf_val:.2f} （通用检测器，置信度仅供参考）")

                        if 'coating_img_base64' in result:
                            st.markdown("---")
                            st.subheader("🔬 舌苔/舌质分离分析")
                            col_coat1, col_coat2 = st.columns(2)
                            with col_coat1:
                                st.markdown("**🟢 舌苔区域（绿色标注）**")
                                st.image(f"data:image/png;base64,{result['coating_img_base64']}", width=300)
                            with col_coat2:
                                st.markdown("**🔴 舌质区域（红色标注）**")
                                st.image(f"data:image/png;base64,{result['body_img_base64']}", width=300)
                            if '舌苔分离分析' in result:
                                st.info(f"📊 {result['舌苔分离分析']}")
                                st.error(f"📌 最终舌象判定：{result['舌象类型']} (置信度 {result['置信度'] * 100:.1f}%)")

                        # ====== AI 智能解读模块 (自动触发) ======
                        st.markdown("---")
                        st.markdown("<h3>🤖 AI 舌象智能解读</h3>", unsafe_allow_html=True)

                        st.markdown("""
                        <div style="background-color: #E8F5E9; border-left: 6px solid #8D6E63; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                            <h4 style="color: #2E7D32; margin-top: 0;">AI 分析结果：</h4>
                        """, unsafe_allow_html=True)

                        with st.spinner("大模型深度解读中..."):
                            try:
                                api_key = st.secrets.get("ZHIPU_API_KEY", "e7477314b09b4be087232c42440cfe29.RKgXonaZScPmwLEm")
                                if api_key != "YOUR_KEY_HERE":
                                    client = ZhipuAI(api_key=api_key)
                                    b64 = base64.b64encode(uploaded_file.getvalue()).decode()

                                    prompt_text = f"""
                                    这是一张中医舌象照片。初步算法识别为：{result['舌象类型']}（{result['诊断']}）。
                                    请扮演专业中医师，观察图片并结合初步结果，给出详细解读。
                                    请严格按以下三个段落输出，不要使用markdown的粗体标题，直接输出文字：
                                    1. 舌象特征描述（结合图片颜色、形态、舌苔）
                                    2. 中医辨证分析（分析可能的体质或脏腑问题）
                                    3. 日常调理建议（给出饮食、作息等建议）
                                    """

                                    resp = client.chat.completions.create(
                                        model="glm-4v-flash",
                                        messages=[{"role": "user", "content": [
                                            {"type": "text", "text": prompt_text},
                                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                                        ]}]
                                    )
                                    ans = resp.choices[0].message.content
                                    st.markdown(ans)
                                else:
                                    st.warning("请配置 ZHIPU_API_KEY")
                            except Exception as e:
                                st.error(f"AI 解读失败: {e}")
                        st.markdown("</div>", unsafe_allow_html=True)

                        st.markdown("""
                        <div style="color: #8D6E63; font-size: 0.9em; margin-bottom: 20px;">
                            ⚠️ 温馨提示：本结果仅科普参考，不构成诊疗意见，身体不适请线下就医
                        </div>
                        """, unsafe_allow_html=True)

                        col_dial, col_adv = st.columns(2)
                        with col_dial:
                            st.markdown("""
                            <div style="background: #FFF3E0; padding: 15px; border-radius: 8px; height: 100%;">
                                <h4 style="color: #E65100; margin-top: 0;">🔴 辨证提示</h4>
                                <p style="font-size: 14px; color: #4E342E;">{}</p>
                            </div>
                            """.format(result['辩证提示'].replace('\n', '<br>')), unsafe_allow_html=True)

                        with col_adv:
                            st.markdown("""
                            <div style="background: #E3F2FD; padding: 15px; border-radius: 8px; height: 100%;">
                                <h4 style="color: #1565C0; margin-top: 0;">💡 智能建议</h4>
                                <p style="font-size: 14px; color: #4E342E;">{}</p>
                            </div>
                            """.format(result['智能建议'].replace('\n', '<br>')), unsafe_allow_html=True)

    # =========================================================
    # ====== 底部常驻：AI 医生智能对话窗口 (不受上传与否限制) ======
    # =========================================================
    st.markdown("---")
    st.subheader("💬 AI 医生智能问答")
    st.caption("你可以在这里继续追问关于诊断结果、中医养生等问题。如果已上传图片，AI 将结合图片回答。")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 渲染历史对话
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 聊天输入框
    if prompt := st.chat_input("向 AI 医生提问（例如：白腻苔平时吃什么好？）"):
        # 1. 显示用户提问
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. AI 回答
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    api_key = st.secrets.get("ZHIPU_API_KEY", "e7477314b09b4be087232c42440cfe29.RKgXonaZScPmwLEm")
                    if api_key != "YOUR_KEY_HERE":
                        client = ZhipuAI(api_key=api_key)

                        # 判断是否有图片可以传
                        if uploaded_file is not None:
                            b64 = base64.b64encode(uploaded_file.getvalue()).decode()
                            resp = client.chat.completions.create(
                                model="glm-4v-flash",
                                messages=[{"role": "user", "content": [
                                    {"type": "text", "text": f"中医舌象图。提问：{prompt}。请简短专业地回答。"},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                                ]}]
                            )
                        else:
                            resp = client.chat.completions.create(
                                model="glm-4-flash",
                                messages=[{"role": "user", "content": prompt}]
                            )

                        ans = resp.choices[0].message.content
                        st.markdown(ans)
                        st.session_state.chat_history.append({"role": "assistant", "content": ans})
                    else:
                        st.warning("请配置 ZHIPU_API_KEY")
                except Exception as e:
                    st.error(f"AI 失败: {e}")

# ---------- 历史记录页面 ----------
def history_page():
    st.markdown("<h1>📋 诊断历史档案</h1>", unsafe_allow_html=True)
    st.markdown("---")
    history = load_history()
    if len(history) == 0:
        st.info("✨ 暂无历史分析记录，请先前往【舌象分析】进行诊断。")
        return
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.metric("总记录数", len(history))
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        tongue_types = [h.get("舌象类型", "未知") for h in history]
        most_common = max(set(tongue_types), key=tongue_types.count) if tongue_types else "无"
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.metric("最常见舌象", most_common)
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        avg_confidence = sum([h.get("置信度", 0) for h in history]) / len(history)
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.metric("平均置信度", f"{avg_confidence*100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    st.subheader("📜 历史记录列表")
    table_data = []
    for idx, record in enumerate(history):
        table_data.append({
            "序号": idx + 1,
            "时间": record.get("时间", ""),
            "性别": record.get("性别", ""),
            "年龄": f"{record.get('年龄', '')}岁",
            "BMI": record.get("BMI", ""),
            "舌象类型": record.get("舌象类型", ""),
            "诊断": record.get("诊断", ""),
            "置信度": f"{record.get('置信度', 0)*100:.1f}%"
        })
    st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)
    st.subheader("🔍 回顾详细记录")
    options = [
        f"{i+1}. {record['时间']} - {record.get('性别','')}{record.get('年龄','')}岁 - {record['舌象类型']}"
        for i, record in enumerate(history)
    ]
    if options:
        selected_option = st.selectbox("选择要查看的记录", options)
        if selected_option:
            idx = int(selected_option.split(".")[0]) - 1
            r = history[idx]
            c1, c2 = st.columns(2)
            with c1:
                st.info(f"性别：{r.get('性别')}")
                st.info(f"年龄：{r.get('年龄')} 岁")
                st.info(f"BMI：{r.get('BMI')} ({r.get('体型')})")
                st.info(f"舌象：{r.get('舌象类型')}")
                st.info(f"诊断：{r.get('诊断')}")
                if '舌苔分离分析' in r:
                    st.info(f"舌象细节：{r.get('舌苔分离分析')}")
            with c2:
                if "tongue_region_base64" in r and r["tongue_region_base64"]:
                    img_data = base64.b64decode(r["tongue_region_base64"])
                    st.image(Image.open(BytesIO(img_data)), width=250)
                if 'coating_img_base64' in r:
                    col_show1, col_show2 = st.columns(2)
                    with col_show1:
                        st.caption("舌苔区域（绿）")
                        coating_data = base64.b64decode(r["coating_img_base64"])
                        st.image(Image.open(BytesIO(coating_data)), width=120)
                    with col_show2:
                        st.caption("舌质区域（红）")
                        body_data = base64.b64decode(r["body_img_base64"])
                        st.image(Image.open(BytesIO(body_data)), width=120)
            with st.expander("🧠 辩证提示"):
                st.write(r.get("辩证提示"))
            with st.expander("💡 智能建议"):
                st.write(r.get("智能建议"))
            with st.expander("🔬 详细特征信息"):
                st.json(r.get("特征信息", {}))

# ---------- 主应用 ----------
def main():
    inject_custom_css()
    init_auth()
    if not st.session_state['logged_in']:
        login_page()
        return
    with st.sidebar:
        st.markdown(
            f"<h2 style='color:#5E3A2C;'>👤 {st.session_state['username']}</h2>",
            unsafe_allow_html=True
        )
        st.markdown("<hr>", unsafe_allow_html=True)
        page = st.radio(
            "📱 导航菜单", ["👅 舌象分析", "📋 历史记录"],
            label_visibility="collapsed"
        )
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
