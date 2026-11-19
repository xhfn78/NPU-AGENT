
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights
)

from torchvision.transforms.functional import (
    to_tensor
)

from torchvision.utils import (
    draw_bounding_boxes,
    draw_segmentation_masks
)

# =====================================
# 1. GPU 확인
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("사용 장치:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# =====================================
# 2. Mask R-CNN 모델 불러오기
# =====================================

weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT

model = maskrcnn_resnet50_fpn(
    weights=weights
)

model = model.to(device)
model.eval()

print("Mask R-CNN 모델 로딩 완료")


# =====================================
# 3. 테스트 이미지 불러오기
# =====================================

image_path = Path(__file__).parent / "test.jpg"

image = Image.open(image_path).convert("RGB")

image_tensor = to_tensor(image).to(device)


# =====================================
# 4. 객체 탐지 및 Mask 예측
# =====================================

with torch.inference_mode():
    prediction = model([image_tensor])[0]


# =====================================
# 5. Confidence Score 필터링
# =====================================

threshold = 0.7

keep = prediction["scores"] >= threshold

boxes = prediction["boxes"][keep].cpu()
labels = prediction["labels"][keep].cpu()
scores = prediction["scores"][keep].cpu()

masks = prediction["masks"][keep].cpu()

# Mask 확률값을 True / False로 변환
masks = masks[:, 0] >= 0.5

categories = weights.meta["categories"]

label_names = []

for label, score in zip(labels, scores):

    class_name = categories[label.item()]

    label_names.append(
        f"{class_name}: {score.item():.2f}"
    )

    print(
        f"객체: {class_name}, "
        f"신뢰도: {score.item():.3f}"
    )


# =====================================
# 6. 이미지 위에 Mask 그리기
# =====================================

original = torch.from_numpy(
    np.array(image)
).permute(2, 0, 1)

result = draw_segmentation_masks(
    original,
    masks=masks,
    alpha=0.5
)


# =====================================
# 7. Bounding Box 그리기
# =====================================

result = draw_bounding_boxes(
    result,
    boxes=boxes,
    labels=label_names,
    colors="red",
    width=3
)


# =====================================
# 8. 결과 저장 및 시각화
# =====================================

result_image = result.permute(1, 2, 0).numpy()

output_path = Path(__file__).parent / "mask_result.jpg"

Image.fromarray(result_image).save(output_path)

plt.figure(figsize=(12, 8))
plt.imshow(result_image)
plt.axis("off")
plt.title("Mask R-CNN Detection")
plt.show()

print("탐지된 객체 수:", len(boxes))
print("결과 저장:", output_path)
