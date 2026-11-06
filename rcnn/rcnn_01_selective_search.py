"""R-CNN 실습 1: OpenCV Selective Search로 객체 후보 영역 생성하기.

실행 예시
--------
python rcnn_01_selective_search.py --image images/test.jpg --mode fast --num-boxes 100
python rcnn_01_selective_search.py --image images/test.jpg --mode quality --num-boxes 200 --no-show

필요 패키지
-----------
pip install opencv-contrib-python matplotlib

주의: Selective Search 모듈은 일반 opencv-python이 아니라
opencv-contrib-python에 포함되어 있습니다.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


def parse_arguments() -> argparse.Namespace:
    """명령줄에서 실험 설정을 받습니다."""
    parser = argparse.ArgumentParser(
        description="OpenCV Selective Search로 R-CNN 후보 영역을 생성합니다."
    )
    parser.add_argument(
        "--image", "-i", required=True, type=Path,
        help="분석할 입력 이미지의 경로 (예: images/test.jpg)",
    )
    parser.add_argument(
        "--mode", choices=("fast", "quality"), default="fast",
        help="fast: 빠른 탐색, quality: 더 많은 후보를 찾는 느린 탐색 (기본값: fast)",
    )
    parser.add_argument(
        "--num-boxes", type=int, default=100,
        help="결과 이미지에 그릴 후보 영역 수 (기본값: 100)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="저장할 결과 이미지 경로 (기본값: 입력 이미지 폴더/selective_search_result.png)",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Matplotlib 창을 띄우지 않고 파일만 저장합니다.",
    )
    return parser.parse_args()


def check_opencv_contrib() -> None:
    """Selective Search가 포함된 OpenCV contrib 설치 여부를 확인합니다."""
    has_module = hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "segmentation")
    if not has_module:
        raise RuntimeError(
            "Selective Search 모듈을 찾지 못했습니다.\n"
            "다음 명령으로 설치한 뒤 다시 실행하세요:\n"
            "    pip install opencv-contrib-python"
        )


def draw_rectangles(image, rectangles, number_of_boxes: int):
    """원본을 보존한 복사본 위에 후보 영역 사각형을 그립니다."""
    result = image.copy()

    # 후보는 [x, y, w, h] 형식입니다.
    # 앞쪽 후보가 '가장 좋은 후보'라는 뜻은 아니며, 보기 편하게 일부만 그립니다.
    for x, y, width, height in rectangles[:number_of_boxes]:
        cv2.rectangle(
            result,
            (int(x), int(y)),
            (int(x + width), int(y + height)),
            (0, 255, 0),  # OpenCV 색상 순서는 BGR입니다. (0, 255, 0)은 초록색입니다.
            2,
        )
    return result


def main() -> None:
    args = parse_arguments()

    if args.num_boxes < 0:
        raise ValueError("--num-boxes 값은 0 이상이어야 합니다.")
    if not args.image.is_file():
        raise FileNotFoundError(
            f"입력 이미지를 찾을 수 없습니다: {args.image.resolve()}\n"
            "--image 뒤에 올바른 이미지 파일 경로를 지정했는지 확인하세요."
        )

    check_opencv_contrib()

    # OpenCV는 이미지를 BGR 순서의 NumPy 배열로 읽습니다.
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"이미지를 읽지 못했습니다: {args.image.resolve()}")

    # R-CNN의 첫 단계: 이미지에서 객체가 있을 법한 후보 영역을 만듭니다.
    selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    selective_search.setBaseImage(image)

    if args.mode == "fast":
        selective_search.switchToSelectiveSearchFast()
    else:
        selective_search.switchToSelectiveSearchQuality()

    start_time = time.perf_counter()
    rectangles = selective_search.process()
    elapsed_seconds = time.perf_counter() - start_time

    boxes_to_draw = min(args.num_boxes, len(rectangles))
    visualized_image = draw_rectangles(image, rectangles, boxes_to_draw)

    # --output을 생략하면 입력 이미지와 같은 폴더에 결과를 저장합니다.
    output_path = args.output or args.image.parent / "selective_search_result.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved = cv2.imwrite(str(output_path), visualized_image)
    if not saved:
        raise OSError(f"결과 이미지를 저장하지 못했습니다: {output_path.resolve()}")

    print("\n=== Selective Search 실행 결과 ===")
    print(f"입력 이미지: {args.image.resolve()}")
    print(f"탐색 모드: {args.mode}")
    print(f"생성된 전체 후보 영역 수: {len(rectangles):,}개")
    print(f"시각화한 후보 영역 수: {boxes_to_draw:,}개")
    print(f"후보 영역 생성 시간: {elapsed_seconds:.3f}초")
    print(f"결과 이미지 저장 경로: {output_path.resolve()}")

    if not args.no_show:
        # Matplotlib은 RGB 순서를 사용하므로 BGR 이미지를 변환해 표시합니다.
        rgb_image = cv2.cvtColor(visualized_image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8))
        plt.imshow(rgb_image)
        plt.title(
            f"Selective Search ({args.mode}) - {boxes_to_draw} proposals",
            fontsize=12,
        )
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
