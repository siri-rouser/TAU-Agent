"""
Tracking module: GroundingDINO-based open-vocabulary detection + ByteTrack.

Usage:
    tracker = Tracking(model_config, model_checkpoint, device="cuda")
    tracks = tracker.detect_and_track(video_path, text_prompt)
    summary = tracker.format_tracks(tracks, fps)
"""

import sys
import os
import tempfile

# Resolve paths relative to the repo root (parent of this RAG_retriever/ dir)
# so everything works regardless of where the repo is cloned/mounted.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)

# Make GroundingDINO importable from its repo clone
_GDINO_ROOT = os.path.join(_REPO_ROOT, "GroundingDINO")
if _GDINO_ROOT not in sys.path:
    sys.path.insert(0, _GDINO_ROOT)

# SLConfig copies the config to /tmp and imports it as a module;
# /tmp must be on sys.path for that import to succeed.
_TMPDIR = tempfile.gettempdir()
if _TMPDIR not in sys.path:
    sys.path.insert(0, _TMPDIR)

# Remove the Qwen3-VL local transformers checkout which fails GroundingDINO's
# huggingface-hub version check. This leaves any installed transformers intact.
sys.path = [p for p in sys.path if "Qwen3-VL/transformers" not in p]

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from boxmot.trackers.bbox.bytetrack.bytetrack import ByteTrack
from PIL import Image
from torchvision.ops import box_convert
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import get_phrases_from_posmap


# ---------------------------------------------------------------------------
# Detection routing
# ---------------------------------------------------------------------------
#
# A text prompt is resolved into one of three strategies:
#   1. Plain YOLO         – a coarse COCO category (e.g. "car", "truck").
#   2. YOLO + classifier  – a vehicle-attribute prompt the VehicleClassifier
#                           can verify (colour / fine type / car subtype),
#                           e.g. "black sedan", "white van", "blue bus".
#   3. GroundingDINO      – anything else / free-form descriptions.
#
# The VehicleClassifier vocabulary below must match the training checkpoint.
COLOUR_CLASSES = [
    "beige", "black", "blue", "brown", "gray", "green",
    "orange", "red", "silver", "white", "yellow",
]
CAR_SUBTYPE_CLASSES = ["sedan", "suv", "hatchback", "mpv", "estate"]

# Spelling variants and close shades normalised to the canonical classifier
# palette (e.g. "grey"->"gray"; "maroon"/"navy" map to the nearest base colour
# the classifier can actually verify).
_COLOUR_ALIASES = {
    "grey":   "gray",
    "maroon": "red",
    "navy":   "blue",
    "dark":   "black",
}

# Car-subtype synonyms normalised to a canonical CAR_SUBTYPE_CLASSES label.
# Checked longest-first so "station wagon" beats "wagon".
_SUBTYPE_ALIASES = {
    "minivan":        "mpv",
    "people carrier": "mpv",
    "station wagon":  "estate",
    "wagon":          "estate",
}

# Tokens that carry no class information and must not block a vocabulary match
# (articles, filler, and the "dark"/"colored" colour qualifiers the flat
# classifier palette cannot represent, e.g. "dark gray suv", "red-colored van").
_IGNORE_TOKENS = {
    "the", "a", "an", "colored", "coloured", "color", "colour",
}

_TYPE_PHRASES: list[tuple[str, tuple[str | None, list[int], bool]]] = [
    ("combo truck",   ("Vehicle.Combo Truck",  [7],          True)),
    ("single truck",  ("Vehicle.Single Truck", [7],          True)),
    ("semi truck",    ("Vehicle.Combo Truck",  [7],          True)),
    ("semi",          ("Vehicle.Combo Truck",  [7],          True)),
    ("box truck",     ("Vehicle.Single Truck", [7],          True)),
    ("dump truck",    ("Vehicle.Single Truck", [7],          True)),
    ("tow truck",     ("Vehicle.Single Truck", [7],          True)),
    ("garbage truck", ("Vehicle.Single Truck", [7],          True)),
    ("fire truck",    ("Vehicle.Single Truck", [7],          True)),
    ("cement mixer",  ("Vehicle.Single Truck", [7],          True)),
    ("cement truck",  ("Vehicle.Single Truck", [7],          True)),
    ("tanker truck",  ("Vehicle.Single Truck", [7],          True)),
    ("tanker",        ("Vehicle.Single Truck", [7],          True)),
    ("pickup truck",  ("Vehicle.Pickup Truck", [7],          True)),
    ("pickup",        ("Vehicle.Pickup Truck", [7],          True)),
    ("pick up",       ("Vehicle.Pickup Truck", [7],          True)),
    ("trailer",       ("Vehicle.Trailer",      [7],          True)),
    ("van",          ("Vehicle.Van",          [2, 7],       True)),
    ("bus",          ("Vehicle.Bus",          [5],          False)),
    ("police car",   ("Vehicle.Car",          [2],          False)),
    ("taxi",         ("Vehicle.Car",          [2],          False)),
    ("cab",          ("Vehicle.Car",          [2],          False)),
    ("car",          ("Vehicle.Car",          [2],          False)),
    ("motorcycle",   ("Vehicle.Motorcycle",   [3],          False)),
    ("motorbike",    ("Vehicle.Motorcycle",   [3],          False)),
    ("scooter",      ("Vehicle.Motorcycle",   [3],          False)),
    ("bicycle",      ("Vehicle.Bicycle",      [1],          False)),
    ("cyclist",      ("Vehicle.Bicycle",      [1],          False)),
    ("bike",         ("Vehicle.Bicycle",      [1],          False)),
    ("pedestrian",   ("Person",               [0],          False)),
    ("person",       ("Person",               [0],          False)),
    ("man",          ("Person",               [0],          False)),
    ("truck",        (None,                   [7],          False)),
    ("vehicle",      (None,                   [2, 3, 5, 7], False)),
]

# Fallback exact-match map for plain COCO keywords (kept for non-vehicle classes
# such as "traffic light" that the classifier does not cover).
YOLO_CLS_MAP: dict[str, int | list[int]] = {
    "pedestrian":    0,
    "traffic light": 9,
    "motorcycle":    3,
    "motorbike":     3,
    "bicycle":       1,
    "cyclist":       1,
    "vehicle":       [2, 3, 5, 7],
    "car":           2,
    "bus":           5,
    "truck":         7,
    "scooter":       3,
    "person":        0,
    "bike":          1,
}


@dataclass
class DetectionPlan:
    """Resolved detection strategy for a text prompt routed to YOLO."""
    yolo_cls_ids: list[int]
    colour: str | None = None        # classifier colour label to keep
    subtype: str | None = None       # classifier car subtype to keep (implies car)
    ftype: str | None = None         # fine classifier type label to keep
    use_classifier: bool = False     # run VehicleClassifier on each crop


def _match_yolo_cls(text_prompt: str) -> DetectionPlan | None:
    """Resolve *text_prompt* into a :class:`DetectionPlan`, or ``None`` for GroundingDINO.

    A plan is returned only when the prompt is *fully* described by COCO /
    classifier vocabulary (colour + vehicle type + car subtype, in any
    combination), e.g. "black sedan", "white van", "blue bus", "car".
    Free-form prompts containing unrecognised words (e.g. "person in red
    shirt") return ``None`` and fall back to open-vocabulary GroundingDINO.
    """
    text = text_prompt.strip().lower()
    # Normalise hyphens ("dark-colored" or "pickup_truck") to spaces.
    work = f" {text.replace('-', ' ').replace('_', ' ')} "

    # --- colour -----------------------------------------------------------
    colour = None
    for variant in list(COLOUR_CLASSES) + list(_COLOUR_ALIASES):
        token = f" {variant} "
        if token in work:
            colour = _COLOUR_ALIASES.get(variant, variant)
            work = work.replace(token, " ", 1)
            break

    # --- car subtype (canonical labels + aliases, longest phrase first) ----
    subtype = None
    _subtype_phrases = sorted(
        list(CAR_SUBTYPE_CLASSES) + list(_SUBTYPE_ALIASES),
        key=lambda p: -len(p),
    )
    for s in _subtype_phrases:
        token = f" {s} "
        if token in work:
            subtype = _SUBTYPE_ALIASES.get(s, s)
            work = work.replace(token, " ", 1)
            break

    # --- vehicle type (longest phrase first) ------------------------------
    type_label: str | None = None
    yolo_ids: list[int] | None = None
    is_fine = False
    type_matched = False
    for phrase, (label, ids, fine) in _TYPE_PHRASES:
        token = f" {phrase} "
        if token in work:
            type_label, yolo_ids, is_fine = label, list(ids), fine
            type_matched = True
            work = work.replace(token, " ", 1)
            break

    leftover = [t for t in work.split() if t not in _IGNORE_TOKENS]
    matched_any = (colour is not None) or (subtype is not None) or type_matched

    if matched_any and not leftover:
        if subtype is not None:
            # A car subtype only makes sense for cars.
            yolo_ids = [2]
            ftype = None
        elif is_fine:
            ftype = type_label
        else:
            ftype = None
        if yolo_ids is None:
            # Only a colour was given (e.g. "white") -> search all vehicles.
            yolo_ids = [2, 3, 5, 7]
        use_classifier = (colour is not None) or (subtype is not None) or is_fine
        return DetectionPlan(
            yolo_cls_ids=yolo_ids,
            colour=colour,
            subtype=subtype,
            ftype=ftype,
            use_classifier=use_classifier,
        )

    # --- fallback: plain exact COCO keyword -------------------------------
    ids = YOLO_CLS_MAP.get(text)
    if ids is not None:
        ids = ids if isinstance(ids, list) else [ids]
        return DetectionPlan(yolo_cls_ids=ids)

    return None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _preprocess_image(frame_bgr: np.ndarray) -> torch.Tensor:
    """Convert a BGR OpenCV frame to a normalised tensor for GroundingDINO."""
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    image_tensor, _ = transform(image_rgb, None)
    return image_tensor

# ---------------------------------------------------------------------------
# Track dataclass (plain dict for simplicity)
# ---------------------------------------------------------------------------

class Track:
    def __init__(self, track_id: int, frame_idx: int, box_xyxy: np.ndarray,
                 confidence: float, phrase: str):
        self.track_id = track_id
        self.observations = [{
            "frame": frame_idx,
            "box_xyxy": [int(v) for v in box_xyxy],
            "confidence": float(confidence),
            "phrase": phrase,
        }]

    def update(self, frame_idx: int, box_xyxy: np.ndarray,
               confidence: float, phrase: str):
        self.observations.append({
            "frame": frame_idx,
            "box_xyxy": [int(v) for v in box_xyxy],
            "confidence": float(confidence),
            "phrase": phrase,
        })

# ---------------------------------------------------------------------------
# Main Tracking class
# ---------------------------------------------------------------------------

class Tracking:
    """
    Open-vocabulary object detection + ByteTrack tracking using GroundingDINO.

    Parameters
    ----------
    model_config : str
        Path to the GroundingDINO config file (e.g. GroundingDINO_SwinT_OGC.py).
    model_checkpoint : str
        Path to the model weights (.pth file).
    device : str
        'cuda' or 'cpu'.
    classifier_checkpoint : str
        Path to the VehicleClassifier checkpoint (.pt) used to verify vehicle
        attribute prompts (colour / fine type / car subtype).
    dinov2_repo_path : str
        Path to the local DINOv2 repo (containing hubconf.py) the classifier needs.
    classifier_conf_threshold : float
        Minimum classifier confidence required to keep a crop when filtering.
    classifier_crop_padding : float
        Fractional padding added around each YOLO box before classification.
    """

    def __init__(
        self,
        model_config: str,
        model_checkpoint: str,
        device: str = "cuda",
        classifier_checkpoint: str = os.path.join(_THIS_DIR, "weights", "last.pt"),
        dinov2_repo_path: str = os.path.join(_REPO_ROOT, "dinov2"),
        classifier_conf_threshold: float = 0.4,
        classifier_crop_padding: float = 0.15,
        yolo_model_path: str = os.path.join(_THIS_DIR, "weights", "yolo26x.pt"),
        yolo_coco_to_native: dict | None = None,
        yolo_native_to_coco: dict | None = None,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        args = SLConfig.fromfile(model_config)
        args.device = self.device
        GD_model = build_model(args)
        checkpoint = torch.load(model_checkpoint, map_location="cuda" if self.device == "cuda" else "cpu")
        GD_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False, assign=True)
        GD_model.eval()
        self.GD_model = GD_model.to(self.device)

        # VehicleClassifier is lazy-loaded on the first attribute-filtered prompt.
        self.classifier_checkpoint = classifier_checkpoint
        self.dinov2_repo_path = dinov2_repo_path
        self.classifier_conf_threshold = classifier_conf_threshold
        self.classifier_crop_padding = classifier_crop_padding
        self._classifier = None

        # Object detector. When the checkpoint's classes are not COCO (e.g. a
        # custom Fisheye8K model), the caller supplies COCO<->native id maps so
        # the existing COCO-based prompt routing keeps working unchanged: the
        # requested COCO ids are translated to native ids for inference, and the
        # returned native ids are mapped back to representative COCO ids.
        self.yolo_model_path = yolo_model_path
        self.yolo_coco_to_native = yolo_coco_to_native
        self.yolo_native_to_coco = yolo_native_to_coco

    # ------------------------------------------------------------------
    # Detection on a single frame
    # ------------------------------------------------------------------

    def _GD_detect(
        self,
        frame_bgr: np.ndarray,
        caption: str,
        box_threshold: float,
        text_threshold: float,
    ):
        """
        Returns
        -------
        boxes_xyxy : np.ndarray  shape (N, 4)  absolute pixel coords
        confidences : np.ndarray shape (N,)
        phrases : list[str]
        """
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption += "."

        image_tensor = _preprocess_image(frame_bgr).to(self.device)
        h, w = frame_bgr.shape[:2]

        with torch.no_grad():
            outputs = self.GD_model(image_tensor[None], captions=[caption])

        logits = outputs["pred_logits"].sigmoid()[0]   # (nq, 256)
        boxes  = outputs["pred_boxes"][0]               # (nq, 4) cxcywh [0,1]

        mask = logits.max(dim=1)[0] > box_threshold
        logits_filt = logits[mask]
        boxes_filt  = boxes[mask]

        tokenizer = self.GD_model.tokenizer
        # Use the backend Rust encoder directly to avoid "Already borrowed" error
        # from the fast tokenizer's set_truncation_and_padding when the model's
        # forward pass still holds a borrow on the same tokenizer.
        _encoded = tokenizer.backend_tokenizer.encode(caption, add_special_tokens=True)
        tokenized = {"input_ids": _encoded.ids}
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace(".", "")
            for logit in logits_filt
        ]

        # Convert cxcywh [0,1] -> xyxy pixels
        boxes_abs = boxes_filt.cpu() * torch.tensor([w, h, w, h], dtype=torch.float32)
        boxes_xyxy = box_convert(boxes_abs, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        confidences = logits_filt.max(dim=1)[0].cpu().numpy()

        return boxes_xyxy, confidences, phrases

    # ------------------------------------------------------------------
    # YOLO detection on a single frame
    # ------------------------------------------------------------------

    def _yolo_raw(
        self,
        frame_bgr: np.ndarray,
        class_ids: list[int],
        conf_threshold: float = 0.25,
    ):
        """Run YOLO once and return (boxes_xyxy, confidences, cls_ids).

        Keeps the per-detection class id so a single inference can be routed to
        multiple prompts (e.g. cars vs trucks) without re-running the model.
        """
        if not hasattr(self, "_yolo_model"):
            from ultralytics import YOLO
            self._yolo_model = YOLO(self.yolo_model_path)
        class_ids = list(class_ids)
        # Translate requested COCO ids to the detector's native ids when the
        # checkpoint uses a non-COCO class map (e.g. custom Fisheye8K model).
        if self.yolo_coco_to_native is not None:
            run_ids = sorted({self.yolo_coco_to_native[c]
                              for c in class_ids if c in self.yolo_coco_to_native})
        else:
            run_ids = class_ids
        if not run_ids:
            return (np.empty((0, 4), dtype=np.float32),
                    np.empty((0,), dtype=np.float32),
                    np.empty((0,), dtype=int))
        results = self._yolo_model(
            frame_bgr, classes=run_ids, conf=conf_threshold, agnostic_nms=True, verbose=False
        )[0]
        boxes_xyxy  = results.boxes.xyxy.cpu().numpy()             # (N, 4)
        confidences = results.boxes.conf.cpu().numpy()            # (N,)
        cls_ids     = results.boxes.cls.cpu().numpy().astype(int)  # (N,)
        # Map native ids back to representative COCO ids so downstream routing
        # and filtering (which use COCO ids) remain unchanged.
        if self.yolo_native_to_coco is not None:
            cls_ids = np.array(
                [self.yolo_native_to_coco.get(int(c), -1) for c in cls_ids], dtype=int
            )
        # Safety check: ensure only the requested classes are returned.
        keep = np.isin(cls_ids, class_ids)
        return boxes_xyxy[keep], confidences[keep], cls_ids[keep]

    def _YOLO_detect(
        self,
        frame_bgr: np.ndarray,
        class_ids: list[int],
        label: str,
        conf_threshold: float = 0.25,
    ):
        """
        Returns
        -------
        boxes_xyxy : np.ndarray  shape (N, 4)  absolute pixel coords
        confidences : np.ndarray shape (N,)
        phrases : list[str]
        """
        boxes_xyxy, confidences, _ = self._yolo_raw(frame_bgr, class_ids, conf_threshold)
        phrases = [label] * len(confidences)
        return boxes_xyxy, confidences, phrases

    # ------------------------------------------------------------------
    # Vehicle attribute classifier (lazy)
    # ------------------------------------------------------------------

    def _get_classifier(self):
        """Load the VehicleClassifier on first use and cache it.

        Sourced from the MOT_classifier package (vendored as a sibling repo
        clone), which supersedes the original top-level ``vehicle_classifier``
        package but exposes the same ``vehicle_classifier(checkpoint_path, ...)``
        factory / ``.results(image)`` API.
        """
        if self._classifier is None:
            _MOT_CLASSIFIER_ROOT = os.path.join(_REPO_ROOT, "MOT_classifier")
            if _MOT_CLASSIFIER_ROOT not in sys.path:
                sys.path.insert(0, _MOT_CLASSIFIER_ROOT)
            from vehicle_classifier import vehicle_classifier
            self._classifier = vehicle_classifier(
                self.classifier_checkpoint,
                device=self.device,
                dinov2_repo_path=self.dinov2_repo_path,
            )
        return self._classifier

    def _crop_for_classifier(self, frame_bgr: np.ndarray, box_xyxy):
        """Crop a padded RGB PIL image around *box_xyxy*, or None if degenerate."""
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in box_xyxy)
        pad = self.classifier_crop_padding
        bw, bh = x2 - x1, y2 - y1
        x1 = int(max(0, x1 - pad * bw))
        y1 = int(max(0, y1 - pad * bh))
        x2 = int(min(w, x2 + pad * bw))
        y2 = int(min(h, y2 + pad * bh))
        if x2 <= x1 or y2 <= y1:
            return None
        crop_bgr = frame_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            return None
        return Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

    def _matches_plan(self, result: dict, plan: "DetectionPlan") -> bool:
        """Check a classifier result against the colour / subtype / type filters."""
        thr = self.classifier_conf_threshold
        if plan.colour is not None:
            colour = result["colour"]
            if colour["label"] != plan.colour or colour["confidence"] < thr:
                return False
        if plan.subtype is not None:
            # car_subtype is only meaningful when the type head says it's a car.
            if result["type"]["label"] != "Vehicle.Car":
                return False
            subtype = result["car_subtype"]
            if subtype["label"] != plan.subtype or subtype["confidence"] < thr:
                return False
        if plan.ftype is not None:
            vtype = result["type"]
            if vtype["label"] != plan.ftype or vtype["confidence"] < thr:
                return False
        return True

    def _classifier_filter(
        self, frame_bgr: np.ndarray, boxes_xyxy, confidences, phrases,
        plan: "DetectionPlan",
    ):
        """Keep only YOLO detections whose crop matches *plan*'s attributes."""
        classifier = self._get_classifier()
        keep_boxes, keep_conf, keep_phrases = [], [], []
        for box, conf, phrase in zip(boxes_xyxy, confidences, phrases):
            crop = self._crop_for_classifier(frame_bgr, box)
            if crop is None:
                continue
            result = classifier.results(crop)
            if self._matches_plan(result, plan):
                keep_boxes.append(box)
                keep_conf.append(conf)
                keep_phrases.append(phrase)
        if keep_boxes:
            return (
                np.asarray(keep_boxes, dtype=np.float32),
                np.asarray(keep_conf, dtype=np.float32),
                keep_phrases,
            )
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            [],
        )

    def _filter_indices_by_plan(self, frame_bgr, union_boxes, indices, plan, cls_cache):
        """Return the subset of *indices* whose crop matches *plan*'s attributes.

        ``cls_cache`` maps a union box index to its (cached) classifier result so
        the same crop is classified at most once per frame even when several
        prompts need it (e.g. "black sedan" and "white suv" both inspect cars).
        """
        classifier = self._get_classifier()
        keep = []
        for i in indices:
            i = int(i)
            if i not in cls_cache:
                crop = self._crop_for_classifier(frame_bgr, union_boxes[i])
                cls_cache[i] = None if crop is None else classifier.results(crop)
            result = cls_cache[i]
            if result is not None and self._matches_plan(result, plan):
                keep.append(i)
        return keep

    # ------------------------------------------------------------------
    # Detection / tracking helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_dets(boxes_xyxy, confidences):
        """Pack boxes + confidences into ByteTrack's [x1,y1,x2,y2,conf,cls] layout."""
        if len(boxes_xyxy) > 0:
            return np.column_stack([
                boxes_xyxy,
                confidences,
                np.zeros(len(confidences), dtype=np.float32),
            ]).astype(np.float32)
        return np.empty((0, 6), dtype=np.float32)

    def _ingest_tracker_output(
        self, byte_tracker, dets, frame, frame_idx, phrases, default_phrase, tracks_by_id,
    ):
        """Update *byte_tracker* with *dets* and record observations into *tracks_by_id*."""
        # results columns: [x1, y1, x2, y2, track_id, conf, cls, det_ind]
        results = byte_tracker.update(dets, frame)
        seen_tids = set()
        for row in results:
            x1, y1, x2, y2, tid, conf, _, det_ind = row
            tid = int(tid)
            seen_tids.add(tid)
            box = np.array([x1, y1, x2, y2])
            det_ind_int = int(det_ind)
            phrase = phrases[det_ind_int] if 0 <= det_ind_int < len(phrases) else default_phrase
            if tid not in tracks_by_id:
                tracks_by_id[tid] = Track(tid, frame_idx, box, float(conf), phrase)
            else:
                tracks_by_id[tid].update(frame_idx, box, float(conf), phrase)

        # ByteTrack only emits a track once it has been confirmed (matched in at
        # least two frames). Capture still-unconfirmed tracks updated on this
        # frame so observations from sparse detections are not lost.
        for st in byte_tracker.active_tracks:
            if st.is_activated or st.frame_id != byte_tracker.frame_count:
                continue
            tid = int(st.id)
            if tid in seen_tids:
                continue
            box = np.asarray(st.xyxy, dtype=np.float32)
            det_ind_int = int(st.det_ind)
            phrase = phrases[det_ind_int] if 0 <= det_ind_int < len(phrases) else default_phrase
            if tid not in tracks_by_id:
                tracks_by_id[tid] = Track(tid, frame_idx, box, float(st.conf), phrase)
            else:
                tracks_by_id[tid].update(frame_idx, box, float(st.conf), phrase)

    def _make_byte_tracker(self, fps, det_conf):
        """Create a ByteTrack whose thresholds match the detector confidence."""
        return ByteTrack(
            frame_rate=max(1, int(fps)),
            track_thresh=float(det_conf),
            min_conf=min(0.1, float(det_conf) / 2.0),
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def detect_and_track(
        self,
        video_path: str,
        text_prompt: str,
        yolo_conf_threshold: float = 0.25,
        gd_box_threshold: float = 0.4,
        gd_text_threshold: float = 0.3,
        frame_stride: int = 2,
    ) -> list:
        """
        Detect and track objects described by *text_prompt* across a video.

        Thin wrapper around :meth:`detect_and_track_multi` for a single prompt.

        Parameters
        ----------
        video_path : str
            Path to the video file.
        text_prompt : str
            Free-text description of the object to track (e.g. "person in red shirt").
        yolo_conf_threshold : float
            Minimum detection confidence for YOLO bounding boxes.
        gd_box_threshold : float
            Minimum detection confidence for GroundingDINO bounding boxes.
        gd_text_threshold : float
            Minimum text-alignment score for phrase extraction.
        frame_stride : int
            Run detection/tracking on every N-th frame (skipped frames are not
            decoded). Higher values trade temporal resolution for speed.

        Returns
        -------
        List of Track objects, each with `.track_id` and `.observations`.
        """
        result = self.detect_and_track_multi(
            video_path,
            [text_prompt],
            yolo_conf_threshold=yolo_conf_threshold,
            gd_box_threshold=gd_box_threshold,
            gd_text_threshold=gd_text_threshold,
            frame_stride=frame_stride,
        )
        return result[text_prompt]

    def detect_and_track_multi(
        self,
        video_path: str,
        text_prompts: list,
        yolo_conf_threshold: float = 0.25,
        gd_box_threshold: float = 0.4,
        gd_text_threshold: float = 0.3,
        frame_stride: int = 2,
        start_track_id: int = 0,
    ) -> dict:
        """
        Detect and track several object descriptions in a SINGLE video pass.

        The video is decoded once and YOLO runs once per frame on the union of
        all requested COCO classes; the resulting detections (and any shared
        classifier crops) are then routed to a per-prompt ByteTrack. Open-
        vocabulary prompts still need their own GroundingDINO forward pass, but
        they reuse the same decoded frames.

        Parameters
        ----------
        video_path : str
            Path to the video file.
        text_prompts : list[str]
            One or more object descriptions to track (e.g. ["black sedan", "white suv"]).
        yolo_conf_threshold, gd_box_threshold, gd_text_threshold : float
            Detector thresholds (see :meth:`detect_and_track`).
        frame_stride : int
            Run detection/tracking on every N-th frame (skipped frames are not
            decoded). Higher values trade temporal resolution for speed.

        Returns
        -------
        dict[str, list[Track]]
            Maps each input prompt to its list of Track objects.
        """
        # Preserve order, drop duplicates.
        prompts = list(dict.fromkeys(text_prompts))
        if not prompts:
            return {}

        stride = max(1, int(frame_stride))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # The tracker only sees one frame per stride, so its buffer/timeouts must
        # be sized against the effective (sub-sampled) frame rate.
        eff_fps = fps / stride
        # Set the class-level track ID counter so new track IDs start after
        # any already-cached tracks, avoiding ID collisions across runs.
        # (BaseTrack._count is shared across all ByteTrack instances.)
        try:
            from boxmot.trackers.bbox.bytetrack.bytetrack import BaseTrack
            BaseTrack._count = start_track_id
        except Exception:
            pass

        # Build per-prompt state and the union of YOLO classes to detect once.
        states: dict[str, dict] = {}
        union_cls: set[int] = set()
        for p in prompts:
            plan = _match_yolo_cls(p)
            det_conf = yolo_conf_threshold if plan is not None else gd_box_threshold
            states[p] = {
                "plan": plan,
                "tracker": self._make_byte_tracker(eff_fps, det_conf),
                "tracks": {},
            }
            if plan is not None:
                union_cls.update(plan.yolo_cls_ids)
        union_cls = sorted(union_cls)
        has_yolo = len(union_cls) > 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frame_idx = 0
        while True:
            # Cheaply skip frames we don't process: grab() advances the decoder
            # without the expensive full decode that retrieve() performs.
            ret = cap.grab()
            if not ret:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue
            ret, frame = cap.retrieve()
            if not ret:
                break

            n_tracks = sum(len(st["tracks"]) for st in states.values())
            print(f"\r  frame {frame_idx}/{total_frames} ({n_tracks} tracks across "
                  f"{len(prompts)} queries)  ", end="", flush=True)

            # One YOLO inference for every prompt that routes to YOLO.
            if has_yolo:
                u_boxes, u_conf, u_cls = self._yolo_raw(frame, union_cls, yolo_conf_threshold)
            else:
                u_boxes = np.empty((0, 4), dtype=np.float32)
                u_conf = np.empty((0,), dtype=np.float32)
                u_cls = np.empty((0,), dtype=int)
            cls_cache: dict[int, dict] = {}  # union index -> classifier result

            for p, st in states.items():
                plan = st["plan"]
                if plan is None:
                    boxes_xyxy, confidences, phrases = self._GD_detect(
                        frame, p, gd_box_threshold, gd_text_threshold
                    )
                else:
                    idxs = np.nonzero(np.isin(u_cls, plan.yolo_cls_ids))[0]
                    if plan.use_classifier and len(idxs) > 0:
                        idxs = self._filter_indices_by_plan(frame, u_boxes, idxs, plan, cls_cache)
                    if len(idxs) > 0:
                        boxes_xyxy = u_boxes[idxs]
                        confidences = u_conf[idxs]
                        phrases = [p] * len(idxs)
                    else:
                        boxes_xyxy = np.empty((0, 4), dtype=np.float32)
                        confidences = np.empty((0,), dtype=np.float32)
                        phrases = []

                dets = self._build_dets(boxes_xyxy, confidences)
                self._ingest_tracker_output(
                    st["tracker"], dets, frame, frame_idx, phrases, p, st["tracks"]
                )

            frame_idx += 1

        cap.release()
        print()  # newline after progress
        self._fps = fps
        return {p: list(st["tracks"].values()) for p, st in states.items()}


    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------

    @staticmethod
    def format_tracks(tracks: list, fps: float, frame_stride: int = 5) -> str:
        """
        Convert a list of Track objects to a readable trajectory string.

        Each track shows first/last appearance (in seconds) and average bbox.
        ``frame_stride`` subsamples each track's observations (every N-th one)
        before summarising, trading temporal resolution for a lighter summary.
        """
        if not tracks:
            return "No objects matching the description were detected in the video."

        stride = max(1, int(frame_stride))
        lines = []
        for track in tracks:
            obs = track.observations[::stride] or track.observations
            first_frame = obs[0]["frame"]
            last_frame  = obs[-1]["frame"]
            first_sec   = round(first_frame / fps, 2)
            last_sec    = round(last_frame  / fps, 2)
            avg_conf    = round(float(np.mean([o["confidence"] for o in obs])), 3)
            phrase      = obs[0]["phrase"] or "object"
            boxes = np.array([o["box_xyxy"] for o in obs])
            avg_box = [int(v) for v in boxes.mean(axis=0).tolist()]

            lines.append(
                f"Track {track.track_id} ({phrase}): "
                f"first seen at {first_sec}s, last seen at {last_sec}s, "
                f"avg confidence {avg_conf}, "
                f"total observations: {len(obs)}, "
                f"avg bbox [x1={avg_box[0]}, y1={avg_box[1]}, x2={avg_box[2]}, y2={avg_box[3]}]."
            )

        return "\n".join(lines)