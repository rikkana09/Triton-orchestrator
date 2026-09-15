import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Используем модель, обученную на классификацию тональности (2 класса)
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

model.eval()

# Пример входных данных (для определения динамических размеров)
dummy_input = tokenizer("This is a test", return_tensors="pt")
input_ids = dummy_input["input_ids"]
attention_mask = dummy_input["attention_mask"]

# Экспорт в ONNX
torch.onnx.export(
    model,
    (input_ids, attention_mask),
    "model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
    },
    opset_version=11,
)
print("✅ ONNX модель сохранена как model.onnx")
