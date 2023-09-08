import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig


peft_model_path = 'saved_files/chatGLM_6B_QLoRA_t32'

config = PeftConfig.from_pretrained(peft_model_path)
q_config = BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.float32)

base_model = AutoModel.from_pretrained(config.base_model_name_or_path,
                                       quantization_config=q_config,
                                       trust_remote_code=True,
                                       device_map='auto')

while True:
    input_text = input("> ")
    print(f'输入：\n{input_text}')
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
    
    response, history = base_model.chat(tokenizer=tokenizer, query=input_text)
    print(f'微调前：\n{response}')
    
    model = PeftModel.from_pretrained(base_model, peft_model_path)
    response, history = model.chat(tokenizer=tokenizer, query=input_text)
    print(f'微调后: \n{response}')
