"""Locate objects and visible parts for grasp reranking with frozen Grounding DINO.

Queries come from text instructions. The detector runs on the padded RGB
image, with an optional second pass inside the highest-scoring object box.

用冻结的 Grounding DINO 定位物体与可见部位，为抓取重排提供空间线索。
查询由指令文本生成。检测器先处理补边后的 RGB 图像，可在置信度最高的物体框
周围裁出局部图像，再检测一次指定部位。
"""

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
# 只将可定位的表面或部件转成部位查询。材质、口味或用途等描述保留物体查询即可。
_VISIBLE_PARTS = frozenset(
    "attachments|barrel|base|binding|body|bowl|bristles|buttons|cables|camera|cap|"
    "clamping ends|clasp|clipping ends|closure|cover|earpieces|edge|eraser|face|"
    "feathers|grip|handle|head|hinge|keys|lenses|lid|metal clasp|neck|notches|"
    "pages|pendant|petals|pivot|prongs|rim|ring|screen|shackle|shell|spine|stem|"
    "strands|tear-off sheets|tip|tube|webbed feet|wick".split("|")
)


def extract_phrases(prompt):
    """Return the object phrase and an optional object-qualified part phrase.

    Prompts follow the dataset's English templates. For example, "grasp the cup
    by its handle" becomes "cup" for object detection and "cup handle" for the
    full-image part query. The part list keeps parsing limited to visible terms.

    返回物体短语和可选的、带物体名称的部位短语。
    这里按数据集的英语句式解析。例如，"grasp the cup by its handle"
    会拆成物体查询 "cup" 和全图部位查询 "cup handle"。只有词表中的
    可见部位才会触发后者。
    """
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
    """Convert normalized corners to a padded square crop in pixel coordinates.

    Use 1.2 times the longer box side, with a 32-pixel minimum. Skip a crop
    that would cover nearly the full image. Return (left, top, side) or None.

    把归一化物体框转换成带少量边距的正方形裁剪区域。边长为物体框长边的
    1.2 倍且至少为 32 像素。接近全图时跳过放大。输出 (左, 上, 边长) 或 None。
    """
    x1, y1, x2, y2 = box[:4] * (width, height, width, height)
    side = max(32, int(np.ceil(1.2 * max(x2 - x1, y2 - y1))))
    if side >= 0.9 * min(height, width):
        return None
    left = min(max(round((x1 + x2 - side) / 2), 0), width - side)
    top = min(max(round((y1 + y2 - side) / 2), 0), height - side)
    return left, top, side


class Grounder:
    """Run the Transformers 4.57.3 processor and Grounding DINO from a local snapshot.

    Pass a directory with model.safetensors and the processor and tokenizer
    configuration files. Loading is local-only and prediction uses eval and
    inference mode, so no detector parameters are updated.

    使用 Transformers 4.57.3 从本地快照加载 Grounding DINO 和处理器。
    目录需包含 model.safetensors 及处理器和分词器配置。模型只读本地文件，
    在评估和推理模式下运行，检测器参数保持固定。
    """

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
        """Return per-prompt detections in normalized, padded-image coordinates.

        Input images are [B, 3, H, W] in [0, 1]. Each result contains one
        [N, 5] array per query type: (x1, y1, x2, y2, confidence). Empty arrays
        represent missing evidence and keep the result aligned with prompts.

        返回每条指令对应的检测框，坐标相对于补边后的输入图像进行归一化。
        输入图像为取值 [0, 1] 的 [B, 3, H, W]。每个查询类型得到一个
        [N, 5] 数组，依次为左上角、右下角和置信度。空数组表示缺少该类
        定位线索，每张图像仍保留一项结果。
        """
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

        # Batch object and full-image part requests, then route detections back
        # to the original image index and query type.
        # 物体查询与全图部位查询合批推理，再按原始图像索引和查询类型回填。
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
            # 裁剪区域已提供物体上下文，第二次只查询部位，减少检测器因为
            # 匹配到物体名称就给出高分的情况。
            zoom_requests.append((index, "zoom_part_boxes", crop, part + "."))
            crop_bounds.append(bounds)

        # Convert crop coordinates back to the same frame used by grasp candidates.
        # 把裁剪图坐标映射回全图归一化坐标，与抓取候选对齐。
        for (index, kind, _, _), bounds, boxes in zip(
            zoom_requests, crop_bounds, self._detect(zoom_requests), strict=True
        ):
            left, top, side = bounds
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] * side + left) / width
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] * side + top) / height
            results[index][kind] = boxes

        return results

    def _detect(self, requests):
        """Yield up to eight valid detections per request, sorted by confidence.

        Without a target image size, the processor returns normalized corners.
        Remove zero-area boxes and non-finite scores before taking the top eight.

        逐条查询返回最多八个有效检测框，按置信度从高到低排列。处理器
        未指定目标图像尺寸时返回归一化角点。排序前先去除零面积框和非有限分数。
        """
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
