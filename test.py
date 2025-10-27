from transformers import AutoTokenizer
 
from QEfficient import QEFFAutoModelForCausalLM
import torch
# Initialize the model using from_pretrained similar to transformers.AutoModelForCausalLM
model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
# model_name="GPT2"
# model_name="Qwen/Qwen2-1.5B-Instruct"
import time
# t1=time.time()
model=QEFFAutoModelForCausalLM.from_pretrained(model_name)
# model=torch.compile(model.model,fullgraph=True)
# t2=time.time()
# print("torch.compile run for model.model")
# print("time ",t2-t1)
# print("done")
inputs="Help me with this"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokens=tokenizer([input], return_tensors="pt")
# model(**tokens)
model.compile(num_devices=1,num_cores=16) # Considering you have a Cloud AI 100 Standard SKU
print(model.generate(prompts=["Help me with this"], tokenizer=tokenizer))