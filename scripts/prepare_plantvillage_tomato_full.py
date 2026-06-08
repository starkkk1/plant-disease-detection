import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PV_COLOR_DIR = PROJECT_ROOT.parent / "plantvillage dataset" / "color"
DEFAULT_PV_GRAY_DIR = PROJECT_ROOT.parent / "plantvillage dataset" / "grayscale"
DEFAULT_PV_SEGMENTED_DIR = PROJECT_ROOT.parent / "plantvillage dataset" / "segmented"
DEFAULT_EXISTING_THREE_CLASS_TEST = PROJECT_ROOT / "data" / "processed_custom" / "test"
DEFAULT_BANGLADESH_ROOT = PROJECT_ROOT.parent / "data+source" / "Tomato_Leaves"
DEFAULT_MULTILEAF_ROOT = PROJECT_ROOT.parent / "data+source" / "Tomato-Village" / "Variant-c(Object Detection)"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "plantvillage_tomato_full"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PV_CLASS_RENAMES = {
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato___Spider_mites_Two_spotted_spider_mite",
}
BANGLADESH_CLASS_MAP = {
    0: "Tomato___healthy",
    1: "Tomato___diseased",
}
MULTILEAF_CLASS_MAP = {
    0: "Tomato___Early_blight",
    1: "Tomato___healthy",
    2: "Tomato___Late_blight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the full PlantVillage tomato dataset and register external evaluation sets."
    )
    parser.add_argument("--pv-color-dir", type=Path, default=DEFAULT_PV_COLOR_DIR)
    parser.add_argument("--pv-gray-dir", type=Path, default=DEFAULT_PV_GRAY_DIR)
    parser.add_argument("--pv-segmented-dir", type=Path, default=DEFAULT_PV_SEGMENTED_DIR)
    parser.add_argument("--existing-three-class-test", type=Path, default=DEFAULT_EXISTING_THREE_CLASS_TEST)
    parser.add_argument("--bangladesh-root", type=Path, default=DEFAULT_BANGLADESH_ROOT)
    parser.add_argument("--multileaf-root", type=Path, default=DEFAULT_MULTILEAF_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-bangladesh", action="store_true")
    parser.add_argument("--skip-multileaf", action="store_true")
    parser.add_argument("--skip-three-class-test", action="store_true")
    return parser.parse_args()


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def iter_image_files(path: Path) -> list[Path]:
    return sorted(
        file_path for file_path in path.iterdir() if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def normalized_class_name(source_name: str) -> str:
    return PV_CLASS_RENAMES.get(source_name, source_name)


def safe_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(src, dest)


def make_unique_name(prefix: str, original_name: str, seen_names: set[str]) -> str:
    candidate = f"{prefix}_{original_name}"
    if candidate not in seen_names:
        seen_names.add(candidate)
        return candidate

    stem = Path(original_name).stem
    suffix = Path(original_name).suffix.lower()
    counter = 1
    while True:
        candidate = f"{prefix}_{stem}_{counter:04d}{suffix}"
        if candidate not in seen_names:
            seen_names.add(candidate)
            return candidate
        counter += 1


def prepare_plantvillage_train_val(
    pv_color_dir: Path, pv_gray_dir: Path, pv_segmented_dir: Path, output_root: Path, train_ratio: float, seed: int
) -> tuple[dict[str, dict[str, int]], list[str]]:
    stats: dict[str, dict[str, int]] = {"train": {}, "val": {}}
    class_names: list[str] = []
    rng = random.Random(seed)

    train_root = output_root / "train"
    val_root = output_root / "val"

    source_dirs = [path for path in sorted(pv_color_dir.iterdir()) if path.is_dir() and path.name.startswith("Tomato___")]
    if not source_dirs:
        raise FileNotFoundError(f"No tomato class folders found in {pv_color_dir}")

    for class_dir in source_dirs:
        class_name = normalized_class_name(class_dir.name)
        class_names.append(class_name)

        images = iter_image_files(class_dir)
        if len(images) < 2:
            raise ValueError(f"Class {class_dir.name} has too few images to split safely.")

        rng.shuffle(images)
        train_count = max(1, min(len(images) - 1, int(len(images) * train_ratio)))
        train_images = images[:train_count]
        val_images = images[train_count:]

        seen_names: set[str] = set()
        for split_name, split_images, split_root in (
            ("train", train_images, train_root),
            ("val", val_images, val_root),
        ):
            class_dest = split_root / class_name
            class_dest_gray = output_root / f"{split_name}_gray" / class_name
            class_dest_segmented = output_root / f"{split_name}_segmented" / class_name

            class_dest.mkdir(parents=True, exist_ok=True)
            class_dest_gray.mkdir(parents=True, exist_ok=True)
            class_dest_segmented.mkdir(parents=True, exist_ok=True)

            copied = 0
            for src_path in split_images:
                dest_name = make_unique_name("pv", src_path.name, seen_names)
                safe_copy(src_path, class_dest / dest_name)

                if pv_gray_dir.exists():
                    gray_src = pv_gray_dir / class_dir.name / src_path.name
                    if gray_src.exists():
                        safe_copy(gray_src, class_dest_gray / dest_name)

                if pv_segmented_dir.exists():
                    segmented_src = None
                    for ext in [".jpg", ".JPG", ".jpeg", ".png", src_path.suffix]:
                        cand = pv_segmented_dir / class_dir.name / f"{src_path.stem}_final_masked{ext}"
                        if cand.exists():
                            segmented_src = cand
                            break
                    if segmented_src is None:
                        cand = pv_segmented_dir / class_dir.name / src_path.name
                        if cand.exists():
                            segmented_src = cand
                    
                    if segmented_src:
                        segmented_dest_name = Path(dest_name).with_suffix(segmented_src.suffix).name
                        safe_copy(segmented_src, class_dest_segmented / segmented_dest_name)

                copied += 1
            stats[split_name][class_name] = copied

    return stats, sorted(class_names)


def copy_existing_three_class_test(source_root: Path, output_root: Path) -> dict[str, int]:
    if not source_root.exists():
        raise FileNotFoundError(f"Three-class test source not found: {source_root}")

    stats: dict[str, int] = {}
    target_root = output_root / "eval" / "three_class_external"

    for class_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        seen_names: set[str] = set()
        copied = 0
        for src_path in iter_image_files(class_dir):
            dest_name = make_unique_name("ext3", src_path.name, seen_names)
            safe_copy(src_path, target_root / class_dir.name / dest_name)
            copied += 1
        stats[class_dir.name] = copied

    return stats


def find_matching_image(image_dir: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate

        candidate_upper = image_dir / f"{stem}{extension.upper()}"
        if candidate_upper.exists():
            return candidate_upper

    for file_path in image_dir.iterdir():
        if file_path.is_file() and file_path.stem == stem and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            return file_path

    return None


def clamp_box(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def crop_yolo_objects(
    dataset_name: str,
    image_dir: Path,
    label_dir: Path,
    class_map: dict[int, str],
    output_root: Path,
    filename_prefix: str,
) -> dict[str, int]:
    stats: dict[str, int] = defaultdict(int)
    target_root = output_root / "eval" / dataset_name

    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = find_matching_image(image_dir, label_path.stem)
        if image_path is None:
            continue

        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            width, height = rgb_image.size

            with label_path.open("r", encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    try:
                        class_id = int(float(parts[0]))
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        box_width = float(parts[3])
                        box_height = float(parts[4])
                    except ValueError:
                        continue

                    class_name = class_map.get(class_id)
                    if class_name is None:
                        continue

                    x1 = clamp_box(int((x_center - (box_width / 2)) * width), 0, width)
                    y1 = clamp_box(int((y_center - (box_height / 2)) * height), 0, height)
                    x2 = clamp_box(int((x_center + (box_width / 2)) * width), 0, width)
                    y2 = clamp_box(int((y_center + (box_height / 2)) * height), 0, height)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = rgb_image.crop((x1, y1, x2, y2))
                    class_dir = target_root / class_name
                    class_dir.mkdir(parents=True, exist_ok=True)
                    filename = f"{filename_prefix}_{label_path.stem}_{index:03d}.jpg"
                    crop.save(class_dir / filename, format="JPEG", quality=95)
                    stats[class_name] += 1

    return dict(stats)


def build_bangladesh_eval(source_root: Path, output_root: Path) -> dict[str, int]:
    stats: dict[str, int] = defaultdict(int)
    for split_name in ("train", "val"):
        image_dir = source_root / "Images" / split_name
        label_dir = source_root / "labels" / split_name
        if image_dir.exists() and label_dir.exists():
            split_stats = crop_yolo_objects(
                dataset_name="bangladesh_binary",
                image_dir=image_dir,
                label_dir=label_dir,
                class_map=BANGLADESH_CLASS_MAP,
                output_root=output_root,
                filename_prefix=f"bangladesh_{split_name}",
            )
            for class_name, count in split_stats.items():
                stats[class_name] += count
    return dict(stats)


def build_multileaf_eval(source_root: Path, output_root: Path) -> dict[str, int]:
    stats: dict[str, int] = defaultdict(int)
    for split_name in ("train", "val"):
        split_root = source_root / split_name
        image_dir = split_root / "images"
        label_dir = split_root / "yolo"
        if image_dir.exists() and label_dir.exists():
            split_stats = crop_yolo_objects(
                dataset_name="multileaf_late_blight",
                image_dir=image_dir,
                label_dir=label_dir,
                class_map=MULTILEAF_CLASS_MAP,
                output_root=output_root,
                filename_prefix=f"multileaf_{split_name}",
            )
            for class_name, count in split_stats.items():
                stats[class_name] += count
    return dict(stats)


def write_metadata(
    output_root: Path,
    class_names: list[str],
    split_stats: dict[str, dict[str, int]],
    eval_stats: dict[str, dict[str, int]],
    args: argparse.Namespace,
) -> None:
    metadata_root = output_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)

    class_to_idx = {class_name: index for index, class_name in enumerate(class_names)}
    with (metadata_root / "class_to_idx.json").open("w", encoding="utf-8") as handle:
        json.dump(class_to_idx, handle, indent=2)

    summary = {
        "sources": {
            "plantvillage_color_dir": str(args.pv_color_dir),
            "plantvillage_gray_dir": str(args.pv_gray_dir),
            "plantvillage_segmented_dir": str(args.pv_segmented_dir),
            "existing_three_class_test": str(args.existing_three_class_test),
            "bangladesh_root": str(args.bangladesh_root),
            "multileaf_root": str(args.multileaf_root),
        },
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "num_classes": len(class_names),
        "class_names": class_names,
        "splits": split_stats,
        "evaluation_sets": eval_stats,
    }
    with (metadata_root / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    args = parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1.")
    if not args.pv_color_dir.exists():
        raise FileNotFoundError(f"PlantVillage color directory not found: {args.pv_color_dir}")

    args.output_root.mkdir(parents=True, exist_ok=True)

    split_stats, class_names = prepare_plantvillage_train_val(
        pv_color_dir=args.pv_color_dir,
        pv_gray_dir=args.pv_gray_dir,
        pv_segmented_dir=args.pv_segmented_dir,
        output_root=args.output_root,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    eval_stats: dict[str, dict[str, int]] = {}

    if not args.skip_three_class_test:
        eval_stats["three_class_external"] = copy_existing_three_class_test(
            source_root=args.existing_three_class_test,
            output_root=args.output_root,
        )

    if not args.skip_bangladesh:
        eval_stats["bangladesh_binary"] = build_bangladesh_eval(
            source_root=args.bangladesh_root,
            output_root=args.output_root,
        )

    if not args.skip_multileaf:
        eval_stats["multileaf_late_blight"] = build_multileaf_eval(
            source_root=args.multileaf_root,
            output_root=args.output_root,
        )

    write_metadata(
        output_root=args.output_root,
        class_names=class_names,
        split_stats=split_stats,
        eval_stats=eval_stats,
        args=args,
    )

    print(f"Prepared dataset at: {args.output_root}")
    print(f"Classes: {len(class_names)}")
    for split_name, class_counts in split_stats.items():
        print(f"{split_name}: {sum(class_counts.values())} images")
    for eval_name, class_counts in eval_stats.items():
        print(f"{eval_name}: {sum(class_counts.values())} images")


if __name__ == "__main__":
    main()
