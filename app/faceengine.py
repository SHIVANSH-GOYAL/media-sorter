import cv2
import numpy as np
import onnxruntime as ort

from . import config

ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

FEAT_STRIDES = (8, 16, 32)
NUM_ANCHORS = 2

BANDS = (
    (2, "0-2"),
    (9, "3-9"),
    (19, "10-19"),
    (29, "20-29"),
    (39, "30-39"),
    (49, "40-49"),
    (59, "50-59"),
    (69, "60-69"),
)
TOP_BAND = "70+"

FAIRFACE_AGES = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
FAIRFACE_MIDPOINTS = [1, 6, 15, 24.5, 34.5, 44.5, 54.5, 64.5, 75]
FAIRFACE_GENDERS = ["male", "female"]
FAIRFACE_RACES = ["white", "black", "indian", "east asian", "southeast asian", "middle eastern", "latino hispanic"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def age_band(age):
    if age is None:
        return None
    for limit, label in BANDS:
        if age <= limit:
            return label
    return TOP_BAND


def _fairface_classes(race_id, gender_id, age_id):
    race_idx = int(np.asarray(race_id).reshape(-1)[0])
    gender_idx = int(np.asarray(gender_id).reshape(-1)[0])
    age_idx = int(np.asarray(age_id).reshape(-1)[0])
    if not 0 <= race_idx < len(FAIRFACE_RACES):
        raise ValueError(f"FairFace race class out of range: {race_idx}")
    if not 0 <= gender_idx < len(FAIRFACE_GENDERS):
        raise ValueError(f"FairFace gender class out of range: {gender_idx}")
    if not 0 <= age_idx < len(FAIRFACE_AGES):
        raise ValueError(f"FairFace age class out of range: {age_idx}")
    return FAIRFACE_MIDPOINTS[age_idx], FAIRFACE_AGES[age_idx], FAIRFACE_GENDERS[gender_idx], 0.5


def _distance2bbox(points, distance):
    return np.stack(
        [
            points[:, 0] - distance[:, 0],
            points[:, 1] - distance[:, 1],
            points[:, 0] + distance[:, 2],
            points[:, 1] + distance[:, 3],
        ],
        axis=-1,
    )


def _distance2kps(points, distance):
    out = []
    for i in range(0, distance.shape[1], 2):
        out.append(points[:, 0] + distance[:, i])
        out.append(points[:, 1] + distance[:, i + 1])
    return np.stack(out, axis=-1)


def _nms(dets, thresh):
    x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = areas.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= thresh)[0] + 1]
    return keep


class FaceEngine:
    def __init__(self, models_dir=None, threads=None, det_size=None):
        models_dir = models_dir or config.MODELS
        threads = threads or config.ORT_THREADS
        self.det_size = det_size or config.DET_SIZE
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, int(threads))
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        self.det = ort.InferenceSession(str(models_dir / "det_10g.onnx"), opts, providers=providers)
        self.rec = ort.InferenceSession(str(models_dir / "w600k_r50.onnx"), opts, providers=providers)
        self.attr = ort.InferenceSession(str(models_dir / "fairface.onnx"), opts, providers=providers)
        self.det_input = self.det.get_inputs()[0].name
        self.rec_input = self.rec.get_inputs()[0].name
        self.attr_input = self.attr.get_inputs()[0].name

    def detect(self, img_bgr, det_thresh=None):
        det_thresh = config.DET_THRESHOLD if det_thresh is None else det_thresh
        h, w = img_bgr.shape[:2]
        if h == 0 or w == 0:
            return []
        size = self.det_size
        scale = min(size / w, size / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:nh, :nw] = resized
        blob = (canvas.astype(np.float32) - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]

        outs = self.det.run(None, {self.det_input: blob})
        boxes_all, scores_all, kps_all = [], [], []
        for i, stride in enumerate(FEAT_STRIDES):
            scores = outs[i]
            bbox_preds = outs[i + len(FEAT_STRIDES)]
            kps_preds = outs[i + 2 * len(FEAT_STRIDES)]
            fh = size // stride
            fw = size // stride
            centers = np.stack(np.mgrid[:fh, :fw][::-1], axis=-1).astype(np.float32)
            centers = (centers * stride).reshape((-1, 2))
            centers = np.stack([centers] * NUM_ANCHORS, axis=1).reshape((-1, 2))
            pos = np.where(scores[:, 0] >= det_thresh)[0]
            if pos.size == 0:
                continue
            boxes_all.append(_distance2bbox(centers, bbox_preds * stride)[pos])
            scores_all.append(scores[pos, 0])
            kps_all.append(_distance2kps(centers, kps_preds * stride)[pos])
        if not boxes_all:
            return []
        boxes = np.vstack(boxes_all)
        scores = np.concatenate(scores_all)
        kps = np.vstack(kps_all)
        order = scores.argsort()[::-1]
        boxes, scores, kps = boxes[order], scores[order], kps[order]
        keep = _nms(boxes, 0.4)
        boxes, scores, kps = boxes[keep], scores[keep], kps[keep]
        boxes = boxes / scale
        kps = kps / scale
        np.clip(boxes[:, 0::2], 0, w - 1, out=boxes[:, 0::2])
        np.clip(boxes[:, 1::2], 0, h - 1, out=boxes[:, 1::2])
        faces = []
        for box, score, kp in zip(boxes, scores, kps):
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            if bw < 12 or bh < 12:
                continue
            faces.append({"bbox": box.tolist(), "score": float(score), "kps": kp.reshape(5, 2)})
        return faces

    def align(self, img_bgr, kps):
        src = np.asarray(kps, dtype=np.float32)
        mat, _ = cv2.estimateAffinePartial2D(src, ARCFACE_DST, method=cv2.LMEDS)
        if mat is None:
            mat = cv2.getAffineTransform(src[:3], ARCFACE_DST[:3])
        return cv2.warpAffine(img_bgr, mat, (112, 112), borderValue=0.0)

    def embed(self, aligned_bgr):
        blob = cv2.dnn.blobFromImage(
            aligned_bgr, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True
        )
        emb = self.rec.run(None, {self.rec_input: blob})[0].reshape(-1).astype(np.float32)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def attribute(self, img_bgr, bbox):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        px, py = int(w * 0.25), int(h * 0.25)
        x1, y1 = max(0, x1 - px), max(0, y1 - py)
        x2 = min(img_bgr.shape[1], x2 + px)
        y2 = min(img_bgr.shape[0], y2 + py)
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            crop = img_bgr
        crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        crop = (crop - IMAGENET_MEAN) / IMAGENET_STD
        blob = crop.transpose(2, 0, 1)[np.newaxis, ...]
        race, gender, age = self.attr.run(None, {self.attr_input: blob})

        age_est, band, gender_label, age_conf = _fairface_classes(race, gender, age)
        return float(age_est), band, gender_label, age_conf

    def analyze(self, img_bgr, det_thresh=None):
        results = []
        for face in self.detect(img_bgr, det_thresh=det_thresh):
            aligned = self.align(img_bgr, face["kps"])
            embedding = self.embed(aligned)
            age, band, gender, conf = self.attribute(img_bgr, face["bbox"])
            results.append(
                {
                    "bbox": face["bbox"],
                    "det_score": face["score"],
                    "embedding": embedding,
                    "age": age,
                    "age_band": band,
                    "gender": gender,
                    "age_conf": conf,
                    "aligned": aligned,
                }
            )
        return results
