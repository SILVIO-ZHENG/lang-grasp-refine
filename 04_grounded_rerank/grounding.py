"""Offline, prompt-only Grounding DINO localization for grasp reranking."""

import re
from pathlib import Path

import numpy as np
import torch


_LEADING_ACTION = re.compile(
    r"^(?:please\s+)?(?:take hold of|pick up|give me|grab|grasp|grip|hold|lift|fetch|take)\s+"
)
_ARTICLE = re.compile(r"^(?:the|an|a)\s+")
_OBJECT_AND_HINT = re.compile(r"^(.+?)\s+(?:at|by|on)\s+(?:its|the)\s+(.+)$")
_PART_OF_OBJECT = re.compile(r"^(.+?)\s+of\s+(.+)$")

# Localizable surfaces or components. The source prompts also give non-spatial
# hints such as material, flavor, or function, which contribute no part query.
_VISIBLE_PARTS = frozenset(
    "attachments|barrel|base|binding|body|bowl|bristles|buttons|cables|camera|cap|"
    "clamping ends|clasp|clipping ends|closure|cover|earpieces|edge|eraser|face|"
    "feathers|grip|handle|head|hinge|keys|lenses|lid|metal clasp|neck|notches|"
    "pages|pendant|petals|pivot|prongs|rim|ring|screen|shackle|shell|spine|stem|"
    "strands|tear-off sheets|tip|tube|webbed feet|wick".split("|")
)


def extract_phrases(prompt):
    """Return the object phrase and an optional object-qualified part phrase."""
    text = " ".join(prompt.lower().strip(" .!?;,\t\n").split())
    text = _ARTICLE.sub("", _LEADING_ACTION.sub("", text))

    target_and_hint = _OBJECT_AND_HINT.fullmatch(text)
    if target_and_hint:
        target, hint = target_and_hint.groups()
        return target, f"{target} {hint}" if hint in _VISIBLE_PARTS else None

    part_of_object = _PART_OF_OBJECT.fullmatch(text)
    if part_of_object and part_of_object[1] in _VISIBLE_PARTS:
        part, target = part_of_object.groups()
        target = _ARTICLE.sub("", target)
        return target, f"{target} {part}"

    return text, None


def _object_crop(box, height, width):
    """Return integer square-crop bounds, or None when zoom would be negligible."""
    x1, y1, x2, y2 = box[:4] * (width, height, width, height)
    side = max(32, int(np.ceil(1.2 * max(x2 - x1, y2 - y1))))
    if side >= 0.9 * min(height, width):
        return None
    left = min(max(round((x1 + x2 - side) / 2), 0), width - side)
    top = min(max(round((y1 + y2 - side) / 2), 0), height - side)
    return left, top, side


class Grounder:
    """Use the official Transformers 4.57.3 processor and a local HF snapshot."""

    def __init__(self, model_path, device, *, batch_size=4, threshold=0.25, zoom_parts=False):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        model_path = Path(model_path)
        if not (model_path / "model.safetensors").is_file():
            raise FileNotFoundError(f"Missing Grounding DINO model.safetensors in {model_path}")
        if batch_size < 1:
            raise ValueError("Grounding batch size must be positive")
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.threshold = threshold
        self.zoom_parts = zoom_parts
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True, use_fast=True, trust_remote_code=False
        )
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_path, local_files_only=True, use_safetensors=True, trust_remote_code=False
        ).to(self.device).eval()

    @torch.inference_mode()
    def predict(self, images, prompts):
        """Return independent object and visible-part boxes in padded input coordinates."""
        if images.ndim != 4 or images.shape[1] != 3 or len(images) != len(prompts):
            raise ValueError("Expected B×3×H×W images and one prompt per image")

        images = images.detach().to(device="cpu", dtype=torch.float32)
        images = images.mul(255).round().clamp(0, 255).to(torch.uint8)
        results = [
            {
                "object_boxes": np.empty((0, 5), dtype=np.float32),
                "part_boxes": np.empty((0, 5), dtype=np.float32),
            }
            for _ in prompts
        ]
        requests = []
        part_requests = []
        for index, prompt in enumerate(prompts):
            target, part = extract_phrases(prompt)
            if target:
                requests.append((index, "object_boxes", images[index], target + "."))
            if part:
                requests.append((index, "part_boxes", images[index], part + "."))
                part_requests.append((index, part.removeprefix(target + " ")))

        for (index, kind, _, _), boxes in zip(requests, self._detect(requests), strict=True):
            results[index][kind] = boxes
        if not self.zoom_parts:
            return results

        height, width = images.shape[-2:]
        zoom_requests = []
        crop_bounds = []
        for result in results:
            result["zoom_part_boxes"] = np.empty((0, 5), dtype=np.float32)
        for index, part in part_requests:
            object_boxes = results[index]["object_boxes"]
            if not len(object_boxes):
                continue
            bounds = _object_crop(object_boxes[0], height, width)
            if bounds is None:
                continue
            left, top, side = bounds
            crop = images[index, :, top : top + side, left : left + side]
            # The crop already identifies the object. The bare part phrase
            # keeps the model from scoring a proposal on the object noun alone.
            zoom_requests.append((index, "zoom_part_boxes", crop, part + "."))
            crop_bounds.append(bounds)

        for (index, kind, _, _), bounds, boxes in zip(
            zoom_requests, crop_bounds, self._detect(zoom_requests), strict=True
        ):
            left, top, side = bounds
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] * side + left) / width
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] * side + top) / height
            results[index][kind] = boxes

        return results

    def _detect(self, requests):
        for offset in range(0, len(requests), self.batch_size):
            batch = requests[offset : offset + self.batch_size]
            inputs = self.processor(
                images=[image for _, _, image, _ in batch],
                text=[phrase for _, _, _, phrase in batch],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**inputs)
            detections = self.processor.post_process_grounded_object_detection(
                outputs, input_ids=inputs.input_ids, threshold=self.threshold
            )
            for detection in detections:
                boxes = detection["boxes"].float().clamp(0.0, 1.0)
                scores = detection["scores"].float()
                keep = (boxes[:, 2:] > boxes[:, :2]).all(dim=1) & scores.isfinite()
                boxes = boxes[keep]
                scores = scores[keep]
                order = scores.argsort(descending=True)[:8]
                yield torch.cat((boxes[order], scores[order, None]), dim=1).cpu().numpy()
