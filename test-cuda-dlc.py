"""Same CUDA-session test as test-cuda.py but for the dlc env."""
import os
import glob
import sys

sp = r'C:\Users\evija\anaconda3\envs\dlc\Lib\site-packages'
_dll_cookies = []  # MUST keep references alive — cookies hold the registration
for sub in ['cudnn', 'cublas', 'cuda_runtime', 'curand', 'cufft', 'cuda_nvrtc', 'nvjitlink']:
    bin_dir = os.path.join(sp, 'nvidia', sub, 'bin')
    if os.path.isdir(bin_dir):
        _dll_cookies.append(os.add_dll_directory(bin_dir))
        os.environ['PATH'] = bin_dir + os.pathsep + os.environ['PATH']
        print(f'  added: {bin_dir}')

import onnxruntime as ort
import numpy as np
from onnx import helper, TensorProto, save_model

print('\nonnxruntime:', ort.__version__)
node = helper.make_node('Identity', ['X'], ['Y'])
graph = helper.make_graph([node], 'g',
    [helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 2])],
    [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 2])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
save_model(model, '_t.onnx')

try:
    sess = ort.InferenceSession('_t.onnx', providers=['CUDAExecutionProvider'])
    used = sess.get_providers()
    print('actual providers:', used)
    if 'CUDAExecutionProvider' in used:
        out = sess.run(None, {'X': np.eye(2, dtype=np.float32)})
        print('CUDA inference:', out[0].tolist())
        print('VERDICT: CUDA works in dlc env')
    else:
        print('VERDICT: fell back to CPU')
        sys.exit(1)
finally:
    if os.path.exists('_t.onnx'):
        os.remove('_t.onnx')
