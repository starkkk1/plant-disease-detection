"""Grad-CAM++ for the same checkpoint, class order and input as prediction.

Call under the API's shared CPU slot: backward hooks must not overlap inference.
"""
import base64
import io

import cv2
import numpy as np
import torch
from PIL import Image


class ExplanationUnavailable(RuntimeError):
    pass


def target_layer(predictor):
    model = predictor.model
    if predictor.model_name == "convnext_tiny":
        layer = model.stages[-1].blocks[-1].conv_dw
    elif predictor.model_name in ("mobilenetv3_small_100", "efficientnet_b0"):
        layer = model.blocks[-1]
    else:
        raise ExplanationUnavailable("No Grad-CAM layer configured for this architecture")
    name = next(name for name, module in model.named_modules() if module is layer)
    return layer, name


def png_base64(pixels):
    with io.BytesIO() as output:
        Image.fromarray(pixels).save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("ascii")


def explain_image(predictor, image, top_k=3, class_id=None):
    try:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError as exc:
        raise ExplanationUnavailable("Install the backend Grad-CAM dependency") from exc
    if class_id is not None and (isinstance(class_id, bool) or not isinstance(class_id, int)
                                or not 0 <= class_id < len(predictor.class_names)):
        raise ValueError("class_id must identify a trained class")
    layer, name = target_layer(predictor)
    prediction = predictor.predict(image, top_k=len(predictor.class_names))
    target = prediction["predictions"][0] if class_id is None else next(
        item for item in prediction["predictions"] if item["class_id"] == class_id)
    with torch.inference_mode(False), torch.enable_grad():
        tensor = predictor.transform(image).unsqueeze(0).to(predictor.device).requires_grad_(True)
        try:
            # The context manager removes every activation/gradient hook, also on failure.
            with GradCAMPlusPlus(model=predictor.model, target_layers=[layer]) as cam:
                heatmap = cam(tensor, targets=[ClassifierOutputTarget(target["class_id"])])[0]
        finally:
            predictor.model.zero_grad(set_to_none=True)
    if heatmap.shape != (predictor.image_size, predictor.image_size) or not np.isfinite(heatmap).all():
        raise RuntimeError("Grad-CAM returned an invalid heatmap")
    heatmap = np.clip(heatmap, 0, 1)
    resized = cv2.resize(np.asarray(image.convert("RGB")),
                         (predictor.image_size, predictor.image_size), interpolation=cv2.INTER_LINEAR)
    overlay = show_cam_on_image(resized.astype(np.float32) / 255, heatmap, use_rgb=True)
    return {"model_version": prediction["model_version"], "predictions": prediction["predictions"][:top_k],
            "explanation": {"method": "gradcam++", "target": target, "target_layer": name,
                "width": predictor.image_size, "height": predictor.image_size,
                "has_signal": bool(np.any(heatmap > 1e-6)),
                "overlay_png_base64": png_base64(overlay),
                "heatmap_png_base64": png_base64(np.rint(heatmap * 255).astype(np.uint8))}}
