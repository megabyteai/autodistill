<span class="cls-button">Object Detection</span>
<span class="tm-button">Target Model</span>

# What is YOLO-NAS?

[YOLO-NAS](https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md) is an object detection model developed by [Deci AI](https://deci.ai/).

You can use `autodistill` to train a YOLO-NAS object detection model on a dataset of labelled images generated by the base models that `autodistill` supports.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [YOLO-NAS Autodistill documentation](https://autodistill.github.io/autodistill/target_models/yolonas/).

## Installation

To use the YOLO-NAS target model, you will need to install the following dependency:

```bash
pip3 install autodistill-yolonas
```

## Quickstart

```python
from autodistill_yolonas import YOLONAS

target_model = YOLONAS()

# train a model
# specify the directory where your annotations (in YOLO format) are stored
target_model.train("./context_images_labeled", epochs=20)

# run inference on the new model
pred = target_model.predict("./context_images_labeled/train/images/dog-7.jpg", confidence=0.01)
```

## License

The YOLO-NAS model is licensed under the [YOLO-NAS License](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md).