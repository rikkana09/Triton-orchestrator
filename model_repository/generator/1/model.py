import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.logger.log_info("Loading TinyLlama...")
        self.tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float16,
            device_map="cuda"
        )
        self.logger.log_info("Generator ready")

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                input_tensor = pb_utils.get_input_tensor_by_name(request, "prompt")
                prompt = input_tensor.as_numpy()[0].decode("utf-8")
                inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=150)
                response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # ✅ Кодируем строку в байты
                out_tensor = pb_utils.Tensor("text_output", np.array([response_text.encode("utf-8")], dtype=np.object_))
                responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
            except Exception as e:
                self.logger.log_error(f"Generator error: {e}")
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[], error=pb_utils.TritonError(str(e))
                ))
        return responses

    def finalize(self):
        self.logger.log_info("Generator finalized")