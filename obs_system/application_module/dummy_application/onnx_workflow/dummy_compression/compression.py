from obs_system.application_module.dummy_application.onnx_workflow.interface.onnx_interface import ONNXInterface
from onnxruntime.quantization import QuantType, quantize_dynamic

import onnx
import onnxruntime
import os 
import numpy as np

class Compression(ONNXInterface): 
    def __init__(self, model_path):
        super().__init__(model_path)
        self.session = None
        self.model_path = model_path
        self.model_name = os.path.basename(model_path).split(".")[0]
        self.model = None
        self.quantized_model_path = self.model_name + "_quantized.onnx"

    def quantize_model(self):
        print("Quantization")
        self.model = quantize_dynamic(
            model_input=self.model_path,
            model_output=self.quantized_model_path, 
            weight_type=QuantType.QUInt8,
            per_channel=False,
            reduce_range=True,
            nodes_to_exclude=['/model.24/Mul_1','/model.24/Mul_3','/model.24/Concat',
                            '/model.24/Mul_5','/model.24/Mul_7','/model.24/Concat_1',
                            '/model.24/Mul_9','/model.24/Mul_11','/model.24/Concat_2',
                            '/model.24/Reshape_1','/model.24/Reshape_3','/model.24/Reshape_5',
                            '/model.24/Concat_3']
        )

    def session_create(self):
        print("Session Create")
        self.session = onnxruntime.InferenceSession(self.quantized_model_path)
        
    def session_run(self, input_data) -> np.ndarray:
        print("Session Rnun")
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        results = self.session.run([output_name], {input_data: input_data})
        print(results)

    def load_model(self, source:str): 
        print("LOAD MODEL")
        self.model = onnx.load(source)
        onnx.checker.check_model(self.model)
