import sys
# sys.path.append("/home/kiyanoush/Cpp_ws/src/robotTest2/python scripts")

import time
import numpy as np

import torch
import torch.onnx
import torch.nn as nn

import onnx
import onnxruntime
from ATCVP.ATCVP import Model

device = torch.device("cpu")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def main():
    n_past = 10
    n_future = 10
    model_dir = "/home/kiyanoush/Downloads/TactileEnabledModel/2DPushing_500_Epochs/model_27_01_2023_15_33/"
    model_name_save_appendix = "ATCVP_model"

    features = dict([("device", device), ("n_past", n_past), ("n_future", n_future), ("model_dir", model_dir),\
                                    ("model_name_save_appendix", model_name_save_appendix), ("criterion", nn.MSELoss())])
    pred_model = Model(features)
    pred_model.load_state_dict(torch.load("/home/kiyanoush/Cpp_ws/src/haptic_finger_control/src/ATCVP/Tactile_Predictive_Model_200_Epochs/model_07_02_2023_12_50/ATCVP_model", map_location='cpu'))
    pred_model = pred_model.double()
    pred_model.eval()

    touch_dummy_input = torch.randn(10, 1, 3, 256, 256).double()
    action_dummy_input = torch.randn(20, 1, 6).double()
    scene_dummy_input = torch.randn(20, 1, 3, 256, 256).double()
    
    start = time.time()
    output = pred_model.forward(scene_dummy_input, action_dummy_input, touch_dummy_input, test=True)
    print(time.time() - start)
    print(output.shape)
  
    # Export the model
    torch.onnx.export(pred_model,               # model being run
                      (scene_dummy_input, action_dummy_input, touch_dummy_input),  # model input (or a tuple for multiple inputs)
                      "ATCVP_onnx.onnx",   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=12,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input1', 'input2', 'input3'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input1' : {1 : 'batch_size'},
                                    'input2' : {1 : 'batch_size'},
                                    'input3' : {1 : 'batch_size'},    # variable length axes
                                    'output' : {1 : 'batch_size'}})
    
    onnx_model = onnx.load("ATCVP_onnx.onnx")
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession("ATCVP_onnx.onnx")
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(scene_dummy_input), ort_session.get_inputs()[1].name: to_numpy(action_dummy_input), ort_session.get_inputs()[2].name: to_numpy(touch_dummy_input)}
    t1 = time.time()
    ort_outs = ort_session.run(None, ort_inputs)
    print(time.time() - t1)
    print(ort_outs[0].shape)
    # # print(ort_outs[0][5, 0, 10])
    # # # np.testing.assert_allclose(to_numpy(output), ort_outs[0], rtol=1e-03, atol=1e-05)

   

if __name__ == "__main__":
    main()