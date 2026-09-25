from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded!")


print("\nLoading language model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

print("Language model loaded successfully!")


print("\n==============================")
print("MODEL INFORMATION")
print("==============================")


print("\nModel:")
print(MODEL_NAME)


print("\nModel device:")
print(model.device)


print("\nModel class:")
print(type(model).__name__)


print("\nLLM TEST COMPLETE")