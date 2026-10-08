"""Evaluate a trained checkpoint with the same RGB preprocessing as inference."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.reporting import (
    add_model_arguments, collect_samples, create_predictor, dataset_evidence,
    evaluate_samples, metadata, positive_int, save_report,
)


def generate_gradcam(predictor, samples, output_dir, count=5):
    """Optional explanation pass: gradients run outside inference_mode."""
    import cv2
    import numpy as np
    from PIL import Image
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    model = predictor.model
    if predictor.model_name == "convnext_tiny":
        layers = [model.stages[-1].blocks[-1].conv_dw]
    elif hasattr(model, "blocks"):
        layers = [model.blocks[-1]]
    else:
        raise ValueError(f"Grad-CAM layer not defined for {predictor.model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with GradCAMPlusPlus(model=model, target_layers=layers) as cam:
        for index, (path, label) in enumerate(samples[:count]):
            prediction = predictor.predict(path, top_k=1)["predictions"][0]
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                tensor = predictor.transform(rgb).unsqueeze(0).to(predictor.device)
                tensor.requires_grad_(True)
                heatmap = cam(tensor, targets=[ClassifierOutputTarget(prediction["class_id"])])[0]
                resized = cv2.resize(np.asarray(rgb), (predictor.image_size, predictor.image_size))
            overlay = show_cam_on_image(resized.astype(np.float32) / 255, heatmap, use_rgb=True)
            output = output_dir / f"{index}_true_{label}_pred_{prediction['class_id']}.png"
            if not cv2.imwrite(str(output), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
                raise OSError(f"Could not write Grad-CAM image: {output}")
            paths.append(str(output))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--dataset", required=True, type=Path, help="Labeled ImageFolder directory")
    parser.add_argument("--limit-per-class", type=positive_int)
    parser.add_argument("--gradcam", action="store_true", help="Optional Grad-CAM++ pass")
    parser.add_argument("--gradcam-count", type=positive_int, default=5)
    args = parser.parse_args()
    predictor, config = create_predictor(args)
    samples = collect_samples(args.dataset, predictor.class_names, args.limit_per_class)
    report = metadata(predictor, config)
    report["dataset"] = dataset_evidence(samples, args.dataset, predictor.class_names)
    report["limit_per_class"] = args.limit_per_class
    report["metrics"] = evaluate_samples(predictor, samples)
    if args.gradcam:
        report["gradcam_images"] = generate_gradcam(
            predictor, samples, args.output.parent / f"{args.output.stem}_gradcam", args.gradcam_count)
    save_report(args.output, report)
    print(f"{predictor.model_version}: n={len(samples)}, "
          f"accuracy={report['metrics']['accuracy']:.6f}, "
          f"macro-F1={report['metrics']['macro_f1']:.6f}; saved {args.output}")


if __name__ == "__main__":
    main()
