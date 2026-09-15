import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.logger.log_info("Classifier (stub) initialized")

    def execute(self, requests):
        responses = []
        for request in requests:
            # Заглушка: всегда возвращаем класс 1 (вопрос)
            logits = np.array([[0.1, 0.9]], dtype=np.float32)
            out_tensor = pb_utils.Tensor("logits", logits)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        return responses

    def finalize(self):
        self.logger.log_info("Classifier (stub) finalized")