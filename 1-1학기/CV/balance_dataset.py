import os
import json
import random
import shutil
from pathlib import Path

from PIL import Image


# =========================
# 경로 설정
# =========================

RAW_IMAGE_ROOT = Path("datasets/Training/원천데이터")
RAW_LABEL_ROOT = Path("datasets/Training/라벨링데이터")

OUTPUT_ROOT = Path("dataset_final")

TRAIN_DIR = OUTPUT_ROOT / "train"
TEST_DIR = OUTPUT_ROOT / "test"

SAMPLE_COUNT = 246
TRAIN_RATIO = 0.8
RANDOM_SEED = 42


CLASS_MAP = {
    "소파": "Sofa",
    "의자": "Chair",
    "책상": "Desk",
    "TV": "TV",
    "냉장고": "Refrigerator",
    "선풍기": "Fan",
    "컴퓨터": "Computer",
}

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]
LABEL_EXTS = [".json"]


def reset_output_dir():
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    for split_dir in [TRAIN_DIR, TEST_DIR]:
        for eng_name in CLASS_MAP.values():
            (split_dir / eng_name).mkdir(parents=True, exist_ok=True)


def find_image_files(folder: Path):
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def find_json_files(folder: Path):
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in LABEL_EXTS
    ]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_int(value):
    return int(float(value))


def get_bbox_from_box(box):
    drawing = box.get("Drawing", "").upper()

    # =====================
    # 1. BOX 처리
    # =====================
    if drawing == "BOX":
        required_keys = ["x1", "y1", "x2", "y2"]

        if not all(k in box for k in required_keys):
            return None

        x1 = safe_int(box["x1"])
        y1 = safe_int(box["y1"])
        x2 = safe_int(box["x2"])
        y2 = safe_int(box["y2"])

        return x1, y1, x2, y2

    # =====================
    # 2. POLYGON 처리
    # =====================
    if drawing == "POLYGON":
        points = []

        polygon_points = box.get("PolygonPoint", [])

        for point_dict in polygon_points:
            for value in point_dict.values():
                try:
                    x, y = value.split(",")
                    points.append((safe_int(x), safe_int(y)))
                except Exception:
                    continue

        if len(points) == 0:
            return None

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        return min(xs), min(ys), max(xs), max(ys)

    return None


def crop_by_json(image_path: Path, json_path: Path, save_path: Path, target_kor_class: str):
    try:
        data = load_json(json_path)
    except Exception:
        return False, "json_load_error"

    file_name = data.get("FILE NAME", "")

    if file_name and file_name != image_path.name:
        return False, "file_name_mismatch"

    bounding_list = data.get("Bounding", [])

    if not bounding_list:
        return False, "no_bounding"

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return False, "image_open_error"

    img_w, img_h = img.size
    saved = False

    for idx, box in enumerate(bounding_list):
        detail = box.get("DETAILS", "")

        if detail != target_kor_class:
            continue

        bbox = get_bbox_from_box(box)

        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox

        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = img.crop((x1, y1, x2, y2))

        if crop.size[0] < 30 or crop.size[1] < 30:
            continue

        final_save_path = save_path

        if len(bounding_list) > 1:
            final_save_path = save_path.with_name(
                f"{save_path.stem}_{idx}{save_path.suffix}"
            )

        try:
            crop.save(final_save_path)
            saved = True
        except Exception:
            continue

    if saved:
        return True, "saved"

    return False, "class_not_found_or_no_valid_bbox"


def main():
    random.seed(RANDOM_SEED)
    reset_output_dir()

    summary = {}

    for kor_class, eng_class in CLASS_MAP.items():
        image_class_root = RAW_IMAGE_ROOT / kor_class
        label_class_root = RAW_LABEL_ROOT / kor_class

        if not image_class_root.exists():
            print(f"[경고] 이미지 폴더 없음: {image_class_root}")
            continue

        if not label_class_root.exists():
            print(f"[경고] 라벨 폴더 없음: {label_class_root}")
            continue

        image_set_dirs = [
            p for p in image_class_root.iterdir()
            if p.is_dir()
        ]

        valid_sets = []

        for img_set_dir in image_set_dirs:
            label_set_dir = label_class_root / img_set_dir.name

            if not label_set_dir.exists():
                continue

            image_files = find_image_files(img_set_dir)
            json_files = find_json_files(label_set_dir)

            if not image_files or not json_files:
                continue

            image_names = {p.name for p in image_files}
            matched = False

            for json_path in json_files:
                try:
                    data = load_json(json_path)
                    json_file_name = data.get("FILE NAME", "")
                except Exception:
                    continue

                if json_file_name in image_names:
                    matched = True
                    break

            if matched:
                valid_sets.append(img_set_dir.name)

        print(f"\n[{kor_class}] 사용 가능 세트: {len(valid_sets)}개")

        if len(valid_sets) == 0:
            continue

        if len(valid_sets) > SAMPLE_COUNT:
            selected_sets = random.sample(valid_sets, SAMPLE_COUNT)
        else:
            selected_sets = valid_sets

        random.shuffle(selected_sets)

        train_count = int(len(selected_sets) * TRAIN_RATIO)
        train_sets = selected_sets[:train_count]
        test_sets = selected_sets[train_count:]

        split_info = {
            "train": train_sets,
            "test": test_sets
        }

        save_count = {
            "train": 0,
            "test": 0
        }

        fail_count = 0
        reason_count = {}

        for split_name, set_names in split_info.items():
            for set_name in set_names:
                img_set_dir = image_class_root / set_name
                label_set_dir = label_class_root / set_name

                image_files = find_image_files(img_set_dir)
                json_files = find_json_files(label_set_dir)

                image_dict = {p.name: p for p in image_files}

                for json_path in json_files:
                    try:
                        data = load_json(json_path)
                        image_file_name = data.get("FILE NAME", "")
                    except Exception:
                        fail_count += 1
                        reason_count["json_load_error"] = reason_count.get("json_load_error", 0) + 1
                        continue

                    if image_file_name not in image_dict:
                        fail_count += 1
                        reason_count["image_not_found"] = reason_count.get("image_not_found", 0) + 1
                        continue

                    image_path = image_dict[image_file_name]

                    save_name = f"{kor_class}_{set_name}_{Path(image_file_name).stem}.jpg"
                    save_path = OUTPUT_ROOT / split_name / eng_class / save_name

                    ok, reason = crop_by_json(
                        image_path=image_path,
                        json_path=json_path,
                        save_path=save_path,
                        target_kor_class=kor_class
                    )

                    if ok:
                        save_count[split_name] += 1
                    else:
                        fail_count += 1
                        reason_count[reason] = reason_count.get(reason, 0) + 1

        summary[kor_class] = {
            "selected_sets": len(selected_sets),
            "train_saved": save_count["train"],
            "test_saved": save_count["test"],
            "fail": fail_count,
            "reasons": reason_count
        }

        print(f"[완료] {kor_class}")
        print(f"  선택 세트: {len(selected_sets)}")
        print(f"  train crop: {save_count['train']}")
        print(f"  test crop : {save_count['test']}")
        print(f"  실패/제외 : {fail_count}")

        if reason_count:
            print("  실패 사유:")
            for reason, cnt in reason_count.items():
                print(f"    - {reason}: {cnt}")

    print("\n===== 최종 요약 =====")

    for kor_class, info in summary.items():
        print(
            f"{kor_class} | "
            f"세트 {info['selected_sets']} | "
            f"train {info['train_saved']} | "
            f"test {info['test_saved']} | "
            f"fail {info['fail']}"
        )

    print("\n완료: dataset_final 생성됨")


if __name__ == "__main__":
    main()