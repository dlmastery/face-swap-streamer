"""Verify ONNX Runtime can actually create a CUDA session (not just list the provider)."""
import os
import glob
import sys

sp = r'C:\Users\evija\anaconda3\envs\faceswap\Lib\site-packages'
nvidia_subs = ['cudnn', 'cublas', 'cuda_runtime', 'curand', 'cufft', 'cuda_nvrtc']
for sub in nvidia_subs:
    bin_dir = os.path.join(sp, 'nvidia', sub, 'bin')
    if os.path.isdir(bin_dir):
        os.add_dll_directory(bin_dir)
        os.environ['PATH'] = bin_dir + os.pathsep + os.environ['PATH']

import onnxruntime as ort
import numpy as np

print('onnxruntime:', ort.__version__)
print('available providers:', ort.get_available_providers())

# Build a trivial 1-op model and try to run it on CUDA
from onnx import helper, TensorProto, save_model
node = helper.make_node('Identity', ['X'], ['Y'])
graph = helper.make_graph([node], 'g',
    [helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 2])],
    [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 2])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
model_path = os.path.join(os.path.dirname(__file__), '_cuda_test.onnx')
save_model(model, model_path)

try:
    sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    print('CUDA session providers actually used:', sess.get_providers())
    out = sess.run(None, {'X': np.eye(2, dtype=np.float32)})
    print('CUDA inference output:', out[0].tolist())
    print('VERDICT: CUDA works')
except Exception as e:
    print('VERDICT: CUDA FAILED ->', repr(e))
    sys.exit(1)
finally:
    if os.path.exists(model_path):
        os.remove(model_path)
