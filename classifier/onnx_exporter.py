import torch
import onnx
from config import Config


class ONNXExporter:
    @staticmethod
    def export(model, input_shape=(1, 3, 224, 224), opset_version=17):
        model.eval()
        input = torch.randn(input_shape, device=Config.DEVICE)
        onnx_path = f"{Config.ONNX_SAVE_NAME}.onnx"

        torch.onnx.export(
            model,
            input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            export_params=True,
            opset_version=opset_version,
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )

        print(f"Model exported to {onnx_path}")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid.")

