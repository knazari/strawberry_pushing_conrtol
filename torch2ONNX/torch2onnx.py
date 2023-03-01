import sys
# sys.path.append("/home/kiyanoush/Cpp_ws/src/robotTest2/python scripts")

import time
import numpy as np

import torch
import intel_extension_for_pytorch as ipex
import torch.onnx
import torch.nn as nn

from openvino.runtime import Core

import onnx
import onnxruntime
from ATCVP.ATCVP import Model

device = torch.device("cpu")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def main():
    n_past = 5
    n_future = 10
    model_dir = "/home/kiyanoush/Cpp_ws/src/haptic_finger_control/src/ATCVP/model_22_02_2023_16_32/"
    model_name_save_appendix = "ATCVP_model"

    features = dict([("device", device), ("n_past", n_past), ("n_future", n_future), ("model_dir", model_dir),\
                                    ("model_name_save_appendix", model_name_save_appendix), ("criterion", nn.MSELoss())])
    pred_model = Model(features)
    pred_model.load_state_dict(torch.load("/home/kiyanoush/Cpp_ws/src/haptic_finger_control/src/ATCVP/model_22_02_2023_16_32/ATCVP_model", map_location='cpu'))
    pred_model = pred_model.float()
    pred_model.eval()
    # for param in pred_model.parameters():
    #     param.grad = None

    touch_dummy_input = torch.randn(5, 1, 3, 64, 64).float()
    action_dummy_input = torch.randn(15, 1, 6).float()
    # scene_dummy_input = torch.randn(15, 1, 3, 256, 256).float()
   
    start = time.time()
    # with torch.no_grad():
    output = pred_model.forward(action_dummy_input, touch_dummy_input, test=True)
    print(time.time() - start)
    # print(output.shape)

  
    # # Export the model
    # torch.onnx.export(pred_model,               # model being run
    #                   (action_dummy_input, touch_dummy_input),  # model input (or a tuple for multiple inputs)
    #                   "ATCVP64crop_onnx.onnx",   # where to save the model (can be a file or file-like object)
    #                   export_params=True,        # store the trained parameter weights inside the model file
    #                   opset_version=12,          # the ONNX version to export the model to
    #                   do_constant_folding=True,  # whether to execute constant folding for optimization
    #                   input_names = ['input1', 'input2', 'input3'],   # the model's input names
    #                   output_names = ['output'], # the model's output names
    #                   dynamic_axes={'input1' : {1 : 'batch_size'},
    #                                 'input2' : {1 : 'batch_size'},    # variable length axes
    #                                 'output' : {1 : 'batch_size'}})
    
    onnx_model = onnx.load("ATCVP64crop_onnx.onnx")
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession("ATCVP64crop_onnx.onnx")
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(action_dummy_input), ort_session.get_inputs()[1].name: to_numpy(touch_dummy_input)}
    t1 = time.time()
    ort_outs = ort_session.run(None, ort_inputs)
    print(time.time() - t1)
    # print(ort_outs[0].shape)
    # print(ort_outs[0][5, 0, 10])
    # print(np.testing.assert_allclose(to_numpy(output), ort_outs[0], rtol=1e-03, atol=1e-05))

    # Load the network to OpenVINO Runtime.
    ie = Core()
    model_onnx = ie.read_model(model="/home/kiyanoush/Cpp_ws/src/haptic_finger_control/torch2ONNX/ATCVP64crop_onnx.onnx")
    compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")

    output_layer_onnx = compiled_model_onnx.output(0)

    # Run inference on the input image.
    tt22 = time.time()
    res_onnx = compiled_model_onnx([action_dummy_input, touch_dummy_input])[output_layer_onnx]
    print(time.time() - tt22)
   

if __name__ == "__main__":
    main()