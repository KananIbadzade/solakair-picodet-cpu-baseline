import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import cv2
import ailia

# ========== robust paths ==========
CODE_DIR = Path(__file__).resolve().parent          # .../solakair/code
ROOT = CODE_DIR.parent                               # .../solakair
UTIL_DIR = CODE_DIR / "util"
CLIPS_DIR = ROOT / "clips"
RESULTS_DIR = ROOT / "results"
sys.path.insert(0, str(UTIL_DIR))  # use our bundled helpers first

# --- ailia sample helpers we copied into util/ ---
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import load_image, write_predictions  # noqa
from webcamera_utils import get_capture, get_writer  # noqa

from logging import getLogger  # noqa
from picodet_utils import grid_priors, get_bboxes

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

# Save ONNX + prototxt next to this script (portable across working dirs)
WEIGHT_S_320_PATH = str(CODE_DIR / 'picodet_s_320_coco.onnx')
MODEL_S_320_PATH  = str(CODE_DIR / 'picodet_s_320_coco.onnx.prototxt')
WEIGHT_S_416_PATH = str(CODE_DIR / 'picodet_s_416_coco.onnx')
MODEL_S_416_PATH  = str(CODE_DIR / 'picodet_s_416_coco.onnx.prototxt')
WEIGHT_M_416_PATH = str(CODE_DIR / 'picodet_m_416_coco.onnx')
MODEL_M_416_PATH  = str(CODE_DIR / 'picodet_m_416_coco.onnx.prototxt')
WEIGHT_L_640_PATH = str(CODE_DIR / 'picodet_l_640_coco.onnx')
MODEL_L_640_PATH  = str(CODE_DIR / 'picodet_l_640_coco.onnx.prototxt')
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/picodet/'

COCO_CATEGORY = (
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork",
    "knife","spoon","bowl","banana","apple","sandwich","orange","broccoli",
    "carrot","hot dog","pizza","donut","cake","chair","couch","potted plant",
    "bed","dining table","toilet","tv","laptop","mouse","remote","keyboard",
    "cell phone","microwave","oven","toaster","sink","refrigerator","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush"
)

from pathlib import Path
IMAGE_PATH = str(Path(__file__).resolve())  # placeholder that always exists
     # unused if we auto-pick a video
SAVE_IMAGE_PATH = 'output.png'
THRESHOLD = 0.3
IOU = 0.6

# ======================
# Flying-object display controls
# ======================

ENABLE_FLYING_ONLY = True
FLYING_LABEL_MAP = {
    "bird": "bird",
    "airplane": "plane",
    "kite": "kite",
    "sports ball": "balloon",  # rough proxy
}

# Heuristic “drone?”
DRONE_HEURISTIC = True
DRONE_AREA_MIN = 0.00005     # ~0.005% of frame area
DRONE_AREA_MAX = 0.02        # ~2%
DRONE_ASPECT_MIN = 0.6
DRONE_ASPECT_MAX = 1.6

# ======================
# Quick-win accuracy boosts
# ======================

# A) ROI crop (focus on sky)
USE_ROI_TOP = True
ROI_KEEP_TOP = 0.60  # keep top 60% of frame

# B) Multi-frame confirmation (reduce 1-frame FPs)
USE_TRACK_CONFIRM = True
TRACK_WINDOW = 3     # look back N frames
MIN_HITS = 2         # must appear in >= MIN_HITS frames
TRACK_IOU = 0.30
_prev_boxes = deque(maxlen=TRACK_WINDOW)  # each: [x1,y1,x2,y2,prob,label]

# ======================
# Arg parser
# ======================

parser = get_base_parser('PP-PicoDet', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument('-th', '--threshold', default=THRESHOLD, type=float, help='object confidence threshold')
parser.add_argument('-iou', '--iou', default=IOU, type=float, help='IOU threshold for NMS')
parser.add_argument('-w', '--write_prediction', action='store_true', help='Flag to output the prediction file.')
parser.add_argument('-m', '--model_type', default='s-416', choices=('s-320','s-416','m-416','l-640'), help='model type')
parser.add_argument('-w', '--write_prediction', nargs='?', const='txt', choices=['txt','json'], type=str,
                    help='Output results to txt or json file.')
args = update_parser(parser)

# If the default 'demo.jpg' doesn't exist and no video was provided, clear inputs
def _normalize_inputs():
    """Keep only real media files; let main() auto-pick from clips/ if empty."""
    if args.video is not None:
        return
    MEDIA_SUFFIXES = {
        '.jpg','.jpeg','.png','.bmp','.tif','.tiff',
        '.mp4','.mov','.m4v','.avi','.mkv'
    }
    valid = []
    for p in args.input:
        pp = Path(p)
        if pp.exists() and pp.suffix.lower() in MEDIA_SUFFIXES:
            valid.append(p)
    args.input = valid  # empty means: we'll auto-pick a clip later

# ======================
# Helpers (flying-only overlay)
# ======================

def _flying_display_label(obj, im_w, im_h):
    orig = COCO_CATEGORY[int(obj.category)]
    disp = FLYING_LABEL_MAP.get(orig, None)
    if disp is None:
        return None
    if DRONE_HEURISTIC and orig in ("bird","airplane","kite"):
        box_w = obj.w * im_w
        box_h = obj.h * im_h
        area_frac = (box_w * box_h) / float(im_w * im_h + 1e-9)
        aspect = (box_w / (box_h + 1e-9))
        if (DRONE_AREA_MIN <= area_frac <= DRONE_AREA_MAX) and (DRONE_ASPECT_MIN <= aspect <= DRONE_ASPECT_MAX):
            disp = "drone?"
    return disp

def draw_flying_results(dets, frame):
    H, W = frame.shape[:2]
    out = frame.copy()
    for o in dets:
        if ENABLE_FLYING_ONLY:
            label = _flying_display_label(o, W, H)
            if label is None:
                continue
        else:
            label = COCO_CATEGORY[int(o.category)]
        x1 = int(o.x * W); y1 = int(o.y * H)
        x2 = int((o.x + o.w) * W); y2 = int((o.y + o.h) * H)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        txt = f"{label} {o.prob:.2f}"
        y_text = max(0, y1 - 6)
        cv2.putText(out, txt, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)
    return out

# ======================
# Tracking helpers
# ======================

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a[:4]; bx1, by1, bx2, by2 = b[:4]
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)

def confirm_multi_frame(dets, W, H, iou_thr=TRACK_IOU):
    cur = []
    for o in dets:
        x1 = int(o.x * W); y1 = int(o.y * H)
        x2 = int((o.x + o.w) * W); y2 = int((o.y + o.h) * H)
        cur.append([x1, y1, x2, y2, float(o.prob), int(o.category)])
    keep_idx = []
    for i, box in enumerate(cur):
        hits = 1
        for past in list(_prev_boxes):
            if any(_iou(box, pb) >= iou_thr for pb in past):
                hits += 1
        if hits >= MIN_HITS:
            keep_idx.append(i)
    _prev_boxes.append(cur)
    return [dets[i] for i in keep_idx]

# ======================
# Core pipeline
# ======================

def preprocess(img, image_shape):
    h, w = image_shape
    im_h, im_w, _ = img.shape
    img = img[:, :, ::-1]  # BGR -> RGB
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    w_scale = w / im_w; h_scale = h / im_h
    scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
    from image_utils import normalize_image
    img = normalize_image(img, normalize_type='ImageNet')
    divisor = 32
    pad_h = int(np.ceil(h / divisor)) * divisor
    pad_w = int(np.ceil(w / divisor)) * divisor
    padding = (0, 0, max(pad_w - w, 0), max(pad_h - h, 0))
    img = cv2.copyMakeBorder(img, padding[1], padding[3], padding[0], padding[2],
                             cv2.BORDER_CONSTANT, value=0)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img, scale_factor

def post_processing(img, output, out_shape, scale_factor):
    im_h, im_w = img.shape[:2]
    cls_scores = output[:4]
    bbox_preds = output[4:]
    num_levels = len(cls_scores)
    featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
    mlvl_priors = grid_priors(num_levels, featmap_sizes)
    cls_score_list = [cls_scores[i][0] for i in range(num_levels)]
    bbox_pred_list = [bbox_preds[i][0] for i in range(num_levels)]
    num_classes = len(COCO_CATEGORY)
    nms_thre = args.iou
    det_bboxes, det_labels = get_bboxes(
        cls_score_list, bbox_pred_list, mlvl_priors,
        out_shape, num_classes,
        scale_factor=scale_factor, with_nms=True,
        nms_thre=nms_thre, score_thr=args.threshold
    )
    detections = []
    for bbox, label in zip(det_bboxes, det_labels):
        x1, y1, x2, y2, prob = bbox
        if prob < args.threshold:
            break
        r = ailia.DetectorObject(
            category=label,
            prob=float(prob),
            x=x1 / im_w,
            y=y1 / im_h,
            w=(x2 - x1) / im_w,
            h=(y2 - y1) / im_h,
        )
        detections.append(r)
    return detections

def predict(net, img):
    dic_shape = {
        's-320': (320, 320),
        's-416': (416, 416),
        'm-416': (416, 416),
        'l-640': (640, 640),
    }
    shape = dic_shape[args.model_type]
    pp_img, scale_factor = preprocess(img, shape)
    if args.benchmark:
        logger.info('BENCHMARK mode')
        total_time_estimation = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            output = net.predict([pp_img])
            end = int(round(time.time() * 1000))
            estimation_time = (end - start)
            logger.info(f'\tailia processing estimation time {estimation_time} ms')
            if i != 0:
                total_time_estimation += estimation_time
        logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
    else:
        output = net.predict([pp_img])
    detect_object = post_processing(img, output, shape, scale_factor)
    return detect_object

def draw_flying_results(dets, frame):
    H, W = frame.shape[:2]
    out = frame.copy()
    for o in dets:
        if ENABLE_FLYING_ONLY:
            label = _flying_display_label(o, W, H)
            if label is None:
                continue
        else:
            label = COCO_CATEGORY[int(o.category)]
        x1 = int(o.x * W); y1 = int(o.y * H)
        x2 = int((o.x + o.w) * W); y2 = int((o.y + o.h) * H)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        txt = f"{label} {o.prob:.2f}"
        y_text = max(0, y1 - 6)
        cv2.putText(out, txt, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)
    return out

def recognize_from_image(net):
    for image_path in args.input:
        logger.info(image_path)
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        logger.info('Start inference...')
        dets = predict(net, img)
        res_img = draw_flying_results(dets, img)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        savepath = get_savepath(str(RESULTS_DIR / "out.png"), image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)
        if args.write_prediction is not None:
            ext = args.write_prediction
            pred_file = "%s.%s" % (savepath.rsplit('.', 1)[0], ext)
            write_predictions(pred_file, dets, img, category=COCO_CATEGORY, file_type=ext)
    logger.info('Script finished successfully.')

def recognize_from_video(net):
    video_file = args.video if args.video else args.input[0]
    capture = get_capture(video_file)
    assert capture.isOpened(), 'Cannot capture source'

    if args.savepath and args.savepath != SAVE_IMAGE_PATH:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = str((RESULTS_DIR / Path(args.savepath).name).resolve())
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = get_writer(out_path, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        H, W = frame.shape[:2]

        if USE_ROI_TOP:
            roi_h = int(H * ROI_KEEP_TOP)
            roi = frame[:roi_h, :, :]
            img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            dets_roi = predict(net, img)
            dets = []
            for o in dets_roi:
                x1 = o.x * 1.0
                y1 = o.y * (roi_h / H)
                w1 = o.w * 1.0
                h1 = o.h * (roi_h / H)
                r = ailia.DetectorObject(category=o.category, prob=o.prob, x=x1, y=y1, w=w1, h=h1)
                dets.append(r)
        else:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = predict(net, img)

        if USE_TRACK_CONFIRM:
            dets = confirm_multi_frame(dets, W, H, iou_thr=TRACK_IOU)

        res_img = draw_flying_results(dets, frame)

        cv2.imshow('frame', res_img)
        frame_shown = True
        if writer is not None:
            writer.write(res_img.astype(np.uint8))

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')

def _pick_default_clip():
    """Return the first video under solakair/clips/ (for one-click runs)."""
    clips_dir = (Path(__file__).resolve().parent.parent / "clips")
    exts = {'.mp4', '.mov', '.m4v', '.avi', '.mkv'}
    if clips_dir.exists():
        for p in sorted(clips_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                return str(p)
    raise RuntimeError(
        "No test video found under solakair/clips/. "
        "Add a clip there or run with --video <path> or --input <image>."
    )

def _normalize_inputs():
    """Keep only real media files; empty list means: we'll auto-pick a clip."""
    if args.video is not None:
        return
    MEDIA_SUFFIXES = {
        '.jpg','.jpeg','.png','.bmp','.tif','.tiff',
        '.mp4','.mov','.m4v','.avi','.mkv'
    }
    valid = []
    for p in args.input:
        pp = Path(p)
        if pp.exists() and pp.suffix.lower() in MEDIA_SUFFIXES:
            valid.append(p)
    args.input = valid


def main():
    dic_model = {
        's-320': (WEIGHT_S_320_PATH, MODEL_S_320_PATH),
        's-416': (WEIGHT_S_416_PATH, MODEL_S_416_PATH),
        'm-416': (WEIGHT_M_416_PATH, MODEL_M_416_PATH),
        'l-640': (WEIGHT_L_640_PATH, MODEL_L_640_PATH),
    }
    WEIGHT_PATH, MODEL_PATH = dic_model[args.model_type]
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # NEW: sanitize inputs and auto-pick a clip if none was provided
    _normalize_inputs()
    if args.video is None and len(args.input) == 0:
        try:
            args.video = _pick_default_clip()
            logger.info(f"No --input/--video. Using default clip: {args.video}")
        except RuntimeError as e:
            logger.error(str(e))
            return

    env_id = args.env_id
    net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=env_id)

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)


if __name__ == '__main__':
    main()
