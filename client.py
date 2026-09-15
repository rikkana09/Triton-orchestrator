import numpy as np
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(
    url="localhost:8000",
    network_timeout=120.0  
)

text = "What is machine learning?"

input_tensor = httpclient.InferInput("text", [1], "BYTES")
input_tensor.set_data_from_numpy(
    np.array([text.encode("utf-8")], dtype=object),
    binary_data=True
)

output_tensor = httpclient.InferRequestedOutput("response", binary_data=True)

response = client.infer(
    model_name="orchestrator",
    inputs=[input_tensor],
    outputs=[output_tensor]
)

result = response.as_numpy("response")
print("Ответ:", result[0].decode("utf-8"))