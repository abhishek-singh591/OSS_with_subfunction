# # -----------------------------------------------------------------------------
# #
# # Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# # SPDX-License-Identifier: BSD-3-Clause
# #
# # -----------------------------------------------------------------------------

# from transformers import AutoTokenizer, TextStreamer

# from QEfficient import QEFFAutoModelForCausalLM

# model_id = "openai/gpt-oss-20b"

# qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id)
# tokenizer = AutoTokenizer.from_pretrained(model_id)

# onnx_model_path = qeff_model.export()
# qpc_path = qeff_model.compile(
#     prefill_seq_len=1,  # Currently we can get best perf using PL=1 i.e. decode-only model, prefill optimizations are being worked on.
#     ctx_len=256,
#     num_cores=16,
#     mxfp6_matmul=True,
#     mxint8_kv_cache=True,
#     num_devices=8,
#     mos=1,
#     aic_enable_depth_first=True,
#     num_speculative_tokens=None,
# )
# print(f"qpc path is {qpc_path}")
# streamer = TextStreamer(tokenizer)
# exec_info = qeff_model.generate(
#     tokenizer,
#     streamer=streamer,
#     prompts="Who is your creator? and What all you are allowed to do?",
#     device_ids=[0, 1, 2, 3],
# )





# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

## BEFORE RUNNING PLS, RUN THE CONVERT SCRIPT TO CONVERT THE SAFETENSORS FROM FP4 to BF16
## SEE DETAILS HERE: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_oss/convert_gpt_oss_weights_to_hf.py
## ONCE CONVERTED, PASS THE MODIFIED WEIGHTS TO THE MODEL_ID BELOW
import torch
from transformers import AutoConfig, TextStreamer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils._utils import load_hf_tokenizer
from QEfficient.utils.constants import Constants
from QEfficient.utils.run_utils import ApiRunner

torch.manual_seed(42)
model_id = "openai/gpt-oss-20b"  # See Comments above to convert saftensors to BF16
config = AutoConfig.from_pretrained(model_id)

config.num_hidden_layers = 2
qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float32, attn_implementation="eager", config=config
)

# model.eval()

tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model_id)
# exit(0)

# config = model.config
batch_size = len(Constants.INPUT_STR)

api_runner = ApiRunner(batch_size, tokenizer, config, Constants.INPUT_STR, Constants.PROMPT_LEN, Constants.CTX_LEN)
# pytorch_hf_tokens = api_runner.run_hf_model_on_pytorch(model)

# qeff_model = QEFFAutoModelForCausalLM(model, continuous_batching=False)
# pytorch_kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)

onnx_model_path = qeff_model.export()
# ort_tokens = api_runner.run_kv_model_on_ort(onnx_model_path, is_tlm=False)

qpc_path = qeff_model.compile(
    prefill_seq_len=1,
    ctx_len=Constants.CTX_LEN,
    num_cores=16,
    mxfp6_matmul=False,
    mxint8_kv_cache=False,
    num_devices=1,
    mos=1,
    aic_enable_depth_first=True,
    num_speculative_tokens=None,
)
# print(f"qpc path is {qpc_path}")
streamer = TextStreamer(tokenizer)
exec_info = qeff_model.generate(
    tokenizer,
    prompts=Constants.INPUT_STR,
    device_ids=[0, 1, 2, 3],
)
 