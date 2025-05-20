import os
import json
import torch
import urllib.request
from PIL import Image
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification

# need a sample input for the traced model. this can be random data, but using an actual
# image and running it through the preprocessor is easier.
url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT-gEyKCLxnUlgpBmWhazDi7ZKWXeVoOVhoMA'

# Opening the image using PIL
img = Image.open(urllib.request.urlretrieve(url)[0])

# Loading the model and preprocessor from HuggingFace
preprocessor = EfficientNetImageProcessor.from_pretrained("dennisjooo/Birds-Classifier-EfficientNetB2")
model = EfficientNetForImageClassification.from_pretrained("dennisjooo/Birds-Classifier-EfficientNetB2")
model.config.return_dict = False
model.eval()

# Preprocessing the input
inputs = preprocessor(img, return_tensors="pt")
input_shape = inputs["pixel_values"].shape

model_config = {
    "input_shape": input_shape,
    "files": [
    ],
    "labels": model.config.id2label,
}

traced_model = torch.jit.trace(model, inputs["pixel_values"])



def export_openvino():
    import openvino as ov

    ov_model = ov.convert_model(traced_model)

    path = "openvino"
    os.system(f"rm -rf {path}")

    ov.save_model(ov_model, f"{path}/model.xml")
    model_config["files"] = [
        f"{path}/model.xml",
        f"{path}/model.bin"
    ]

    with open(f"{path}/config.json", "w") as f:
        json.dump(model_config, f)

def export_coreml():
    import coremltools as ct

    path = "coreml"
    os.system(f"rm -rf {path}")
    
    model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[ct.TensorType(shape=input_shape)],
    )

    model.save(path + "/model.mlpackage")

    model_config["files"] = [
        f"{path}/model.mlpackage/Manifest.json",
        f"{path}/model.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
        f"{path}/model.mlpackage/Data/com.apple.CoreML/model.mlmodel",
    ]
    with open(f"{path}/config.json", "w") as f:
        json.dump(model_config, f)

def export_onnx():
    path = "onnx"

    os.system(f"rm -rf {path}")
    os.system(f"mkdir -p {path}")

    torch.onnx.export(
        traced_model,
        inputs["pixel_values"],
        "onnx/model.onnx",
        verbose=False,
        input_names=["input"],
    )

    os.system(f"mv model.onnx {path}/model.onnx")

    model_config["files"] = [
        f"{path}/model.onnx",
    ]
    with open(f"{path}/config.json", "w") as f:
        json.dump(model_config, f)

def export_ncnn():
    path = "ncnn"
    os.system(f"rm -rf {path}")
    os.system(f"mkdir -p {path}")
    traced_model.save(f"{path}/model.pt")
    input_shape_str = json.dumps(input_shape)
    os.system(f"pnnx {path}/model.pt 'inputshape={input_shape_str}'")


# comment/uncomment the ones you want to export.
# some exports may not work depending on the model or host operating system.
# openvino may require intel system.
# ncnn has limited model/op support.
export_openvino()
export_coreml()
export_onnx()
export_ncnn()
