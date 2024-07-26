# whisper-large-with-LoRA

Whisper large is a robust and powerful ASR model with an encoder-decoder structure, consisting of 32 transformer layers for both the encoder and decoder, 16 attention heads, and approximately 1.55 billion parameters. Fine-tuning this model on a 16GB VRAM GPU is challenging due to its large size. To address this, we can use LoRA (Low-Rank Adaptation) and quantization, which significantly reduce the model's parameter count and computational requirements without substantially compromising accuracy.

LoRA:
Low-Rank Adaptation (LoRA) adds low-rank matrices between existing layers, drastically reducing the number of trainable parameters. For instance, if a layer originally has 1,048,576 parameters (1024 input and output dimensions), LoRA can replace it with two smaller layers (1024x16 and 16x1024, given rank=16), totaling 32,768 parameters. This represents about 3.125% of the original parameter count, significantly reducing the memory and computation needs.

Quantization:
Quantization involves reducing the precision of the model's weights and activations, typically converting from 32-bit floating-point numbers to 8-bit integers (or even 4-bit), by mapping the continuous range of values into a fixed set of discrete levels. This process reduces the memory footprint and computational load, allowing for efficient execution on hardware with limited resources. For example, the output of a sigmoid layer, typically ranging from 0 to 1, can be quantized to a smaller set of discrete values, significantly reducing the required storage and computation.

These techniques enable the fine-tuning and deployment of large models like Whisper on hardware with limited resources, such as GPUs with lower VRAM. For further details on LoRA and quantization. 
You can refer to the LoRA research paper https://arxiv.org/abs/2106.09685, 
and this https://huggingface.co/docs/optimum/concept_guides/quantization Hugging Face tutorial on quantization.