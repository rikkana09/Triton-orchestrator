import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer
import torch

class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.logger.log_info("Orchestrator initialized")
        try:
            #self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            self.tokenizer = AutoTokenizer.from_pretrained("shahrukhx01/bert-mini-finetune-question-detection")
            self.tokenizer.model_max_length = 128
            self.logger.log_info("Tokenizer loaded")
        except Exception as e:
            self.logger.log_error(f"Tokenizer load error: {e}")
            raise

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                input_tensor = pb_utils.get_input_tensor_by_name(request, "text")
                text = input_tensor.as_numpy()[0].decode("utf-8")
                self.logger.log_info(f"Processing: {text[:50]}...")
            except Exception as e:
                self.logger.log_error(f"Failed to get input: {e}")
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[], error=pb_utils.TritonError(str(e))
                ))
                continue

            # 1. Вызываем эмбеддер
            self.logger.log_info("Calling embedder...")
            try:
                embed_req = pb_utils.InferenceRequest(
                    model_name="embedder",
                    requested_output_names=["embedding"],
                    inputs=[pb_utils.Tensor("text", np.array([text.encode("utf-8")], dtype=np.object_))]
                )
                embed_resp = embed_req.exec()
                if embed_resp.has_error():
                    error_msg = f"Embedder error: {embed_resp.error().message()}"
                    self.logger.log_error(error_msg)
                    responses.append(pb_utils.InferenceResponse(
                        output_tensors=[], error=pb_utils.TritonError(error_msg)
                    ))
                    continue
                self.logger.log_info("Embedder done.")
            except Exception as e:
                self.logger.log_error(f"Embedder exception: {e}")
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[], error=pb_utils.TritonError(str(e))
                ))
                continue

            # 2. Токенизация для классификатора
            self.logger.log_info("Tokenizing for classifier...")
            try:
                encoded = self.tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors="pt")
                input_ids = encoded["input_ids"].numpy().astype(np.int64)
                attention_mask = encoded["attention_mask"].numpy().astype(np.int64)
                token_type_ids = encoded["token_type_ids"].numpy().astype(np.int64)
                self.logger.log_info("Tokenization done.")
            except Exception as e:
                self.logger.log_error(f"Tokenizer exception: {e}")
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[], error=pb_utils.TritonError(str(e))
                ))
                continue

            # 3. Вызываем классификатор
            self.logger.log_info("Calling classifier...")
            try:
                class_req = pb_utils.InferenceRequest(
                    model_name="classifier",
                    requested_output_names=["logits"],
                    inputs=[
                        pb_utils.Tensor("input_ids", input_ids),
                        pb_utils.Tensor("attention_mask", attention_mask),
                        pb_utils.Tensor("token_type_ids", token_type_ids)
                    ],
                    preferred_memory=pb_utils.PreferredMemory(pb_utils.TRITONSERVER_MEMORY_CPU)
                )
                class_resp = class_req.exec()
                if class_resp.has_error():
                    error_msg = f"Classifier error: {class_resp.error().message()}"
                    self.logger.log_error(error_msg)
                    responses.append(pb_utils.InferenceResponse(
                        output_tensors=[], error=pb_utils.TritonError(error_msg)
                    ))
                    continue
                self.logger.log_info("Classifier done.")
            except Exception as e:
                self.logger.log_error(f"Classifier exception: {e}")
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[], error=pb_utils.TritonError(str(e))
                ))
                continue

            logits_tensor = pb_utils.get_output_tensor_by_name(class_resp, "logits")
            logits = logits_tensor.as_numpy()
            predicted_class = int(np.argmax(logits[0]))
            self.logger.log_info(f"Predicted class: {predicted_class}")

            final_answer = text
            if predicted_class == 1:
                self.logger.log_info("Detected question, calling generator...")
                try:
                    prompt_text = f"Question: {text}\nAnswer:"
                    gen_req = pb_utils.InferenceRequest(
                        model_name="generator",
                        requested_output_names=["text_output"],
                        inputs=[pb_utils.Tensor("prompt", np.array([prompt_text.encode("utf-8")], dtype=np.object_))],
                        preferred_memory=pb_utils.PreferredMemory(pb_utils.TRITONSERVER_MEMORY_CPU)
                    )
                    gen_resp = gen_req.exec()
                    if gen_resp.has_error():
                        error_msg = f"Generator error: {gen_resp.error().message()}"
                        self.logger.log_error(error_msg)
                        responses.append(pb_utils.InferenceResponse(
                            output_tensors=[], error=pb_utils.TritonError(error_msg)
                        ))
                        continue
                    self.logger.log_info("Generator done.")
                    output_tensor = pb_utils.get_output_tensor_by_name(gen_resp, "text_output")
                    final_answer = output_tensor.as_numpy()[0].decode("utf-8")
                except Exception as e:
                    self.logger.log_error(f"Generator exception: {e}")
                    responses.append(pb_utils.InferenceResponse(
                        output_tensors=[], error=pb_utils.TritonError(str(e))
                    ))
                    continue
            else:
                self.logger.log_info("Not a question, returning original text")

            out_tensor = pb_utils.Tensor("response", np.array([final_answer.encode("utf-8")], dtype=np.object_))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses

    def finalize(self):
        self.logger.log_info("Orchestrator finalized")