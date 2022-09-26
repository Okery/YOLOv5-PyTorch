import torch
import torch_tensorrt


def pytorch_miniaturise_and_save(
    model, input_shape, batch_size, data_path, include_half_precision=False
):
    assert model is not None, "self.model is None"
    assert input_shape is not None, "self.input_shape is None"

    if include_half_precision:
        enabled_precisions = {torch_tensorrt.dtype.float, torch_tensorrt.dtype.half}
    else:
        enabled_precisions = {torch_tensorrt.dtype.float}

    traced_model = torch.jit.trace(model.forward, (torch.randn(1, 3, *input_shape).to("cuda")))
    trt_model = torch_tensorrt.compile(
        traced_model,
        **{
            "inputs": [torch_tensorrt.Input((batch_size, 3, *input_shape))],
            "enabled_precisions": enabled_precisions,
            "workspace_size": 1 << 20,
        }
    )

    torch.jit.save(trt_model, data_path)
