import bitsandbytes as bnb  # BITSANDBYTES IS A TOOL PROVIDED BY HUGGINGFACE TO BE ABLE TO USE QUANTIZATION WITH LARGE MODELS
from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperProcessor
from peft import BitsAndBytesConfig, prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType #PEFT IS A TOOL PROVIDED BY HUGGINGFACE TO USE LoRA AND OTHER ADAPTATION METHODS

class CustomWhisperModel(WhisperForConditionalGeneration):
    def __init__(self, model_name) :
        super().__init__(model_name)
        
    def forward(self, input_ids=None,
                    input_features=None,
                    inputs_embeds = None,
                    attention_mask=None,
                    decoder_input_ids=None,
                    decoder_attention_mask=None,
                    labels=None,
                    decoder_inputs_embeds = None,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=None,
                    output_attention= None,
                    task_type =None):
        
        inputs = {"input_features": input_ids, 'decoder_input_ids' : decoder_input_ids, 'attention_mask' : attention_mask, 'decoder_attention_mask' : decoder_attention_mask,
                'labels' : labels, 'return_dict' : return_dict, 'output_hidden_states' : output_hidden_states, 'output_attentions' : output_attention}
        if input_features != None : 
            inputs['input_features'] = input_features    

        outputs = super().forward(**inputs)
        return outputs
    
def generate_model__and_processor(model_path, quantization_level, lora_rank, lora_alpha, language) :
    processor = WhisperProcessor.from_pretrained(model_path, language=language, task = "transcribe")
    tokenizer = WhisperTokenizer.from_pretrained(model_path, language = language, task = "transcribe")
    
    if quantization_level == 8 :
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    if quantization_level == 4 :
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    else :
        bnb_config = None
        
    custom_model = CustomWhisperModel.from_pretrained(model_path, quantization_config = bnb_config, device_map='auto')
    q_model = prepare_model_for_kbit_training(custom_model)

    peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, target_modules=["q_proj", "v_proj"], 
                            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.1)
    model = get_peft_model(q_model, peft_config)
    return model, tokenizer, processor