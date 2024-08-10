import bitsandbytes as bnb  # BITSANDBYTES IS A TOOL PROVIDED BY HUGGINGFACE TO BE ABLE TO USE QUANTIZATION WITH LARGE MODELS
from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperProcessor
from peft import BitsAndBytesConfig, prepare_model_for_kbit_training, LoraConfig, get_peft_model #PEFT IS A TOOL PROVIDED BY HUGGINGFACE TO USE LoRA AND OTHER ADAPTATION METHODS


def generate_model__and_processor(model_path, quantization_level, lora_rank, lora_alpha, language) :
    processor = WhisperProcessor.from_pretrained(model_path, language=language, task = "transcribe")
    tokenizer = WhisperTokenizer.from_pretrained(model_path, language = language, task = "transcribe")
    
    if quantization_level == 8 :
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    if quantization_level == 4 :
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    else :
        bnb_config = None
        
    custom_model = WhisperForConditionalGeneration.from_pretrained(model_path, quantization_config = bnb_config, device_map='auto')
    q_model = prepare_model_for_kbit_training(custom_model)

    peft_config = LoraConfig(inference_mode=False, target_modules=["q_proj", "v_proj"], r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.1)
    model = get_peft_model(q_model, peft_config)
    return model, tokenizer, processor