import numpy as np
import triton_python_backend_utils as pb_utils
from sentence_transformers import SentenceTransformer
import torch

class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.logger.log_info("Loading SentenceTransformer...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.model.to('cuda')
        self.logger.log_info("Embedder ready")

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            text = input_tensor.as_numpy()[0].decode("utf-8")
            embedding = self.model.encode(text, convert_to_numpy=True)
            out_tensor = pb_utils.Tensor("embedding", embedding.astype(np.float32).reshape(1, -1))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        return responses

    def finalize(self):
        self.logger.log_info("Embedder finalized")