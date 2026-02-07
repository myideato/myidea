---
library_name: transformers
license: other
license_name: deepseek
license_link: LICENSE
tags:
- muiltimodal
- text-to-image
- unified-model
pipeline_tag: image-text-to-text
---

# DeepSeek-VL: Towards Real-World Vision-LanguageUnderstanding

![image/png](assets/sample.jpg)

This is the transformers version of Deepseek-VL-Hybrid, a foundation model for Visual Language Modeling.

## Table of Contents

- [DeepSeek-VL: Towards Real-World Vision-LanguageUnderstanding](#deepseek-vl-towards-real-world-vision-languageunderstanding)
    - [Table of Contents](#table-of-contents)
    - [Model Details](#model-details)
        - [Model Sources](#model-sources)
    - [How to Get Started with the Model](#how-to-get-started-with-the-model)
    - [Training Details](#training-details)
        - [Training Data](#training-data)
        - [Training Pipeline](#training-pipeline)
        - [Training Hyperparameters](#training-hyperparameters)
    - [Evaluation](#evaluation)
    - [Citation](#citation)
    - [Model Card Authors](#model-card-authors)

## Model Details

[Deepseek-VL-Hybrid](https://arxiv.org/abs/2403.05525) was introduced by the DeepSeek AI team. It is a vision-language model (VLM) designed to process both text and images for generating contextually relevant responses. The model leverages LLaMA as its text encoder, while SigLip is used for encoding low-resolution images and SAM (Segment Anything Model) is incorporated to handle high-resolution image encoding, enhancing the model’s ability to process fine-grained visual details. Deepseek-VL-Hybrid is a variant of Deepseek-VL that uses SAM (Segment Anything Model) to handle high-resolution image encoding.

The abstract from the paper is the following:

> We present DeepSeek-VL, an open-source Vision-Language (VL) Model designed for real-world vision and language understanding applications. Our approach is structured around three key dimensions: We strive to ensure our data is diverse, scalable, and extensively covers real-world scenarios including web screenshots, PDFs, OCR, charts, and knowledge-based content, aiming for a comprehensive representation of practical contexts. Further, we create a use case taxonomy from real user scenarios and construct an instruction tuning dataset accordingly. The fine-tuning with this dataset substantially improves the model's user experience in practical applications. Considering efficiency and the demands of most real-world scenarios, DeepSeek-VL incorporates a hybrid vision encoder that efficiently processes high-resolution images (1024 x 1024), while maintaining a relatively low computational overhead. This design choice ensures the model's ability to capture critical semantic and detailed information across various visual tasks. We posit that a proficient Vision-Language Model should, foremost, possess strong language abilities. To ensure the preservation of LLM capabilities during pretraining, we investigate an effective VL pretraining strategy by integrating LLM training from the beginning and carefully managing the competitive dynamics observed between vision and language modalities. The DeepSeek-VL family (both 1.3B and 7B models) showcases superior user experiences as a vision-language chatbot in real-world applications, achieving state-of-the-art or competitive performance across a wide range of visual-language benchmarks at the same model size while maintaining robust performance on language-centric benchmarks. We have made both 1.3B and 7B models publicly accessible to foster innovations based on this foundation model.

This is the model card of a 🤗 [transformers](https://huggingface.co/docs/transformers/index) model that has been pushed on the Hub.

- **Developed by:** Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, Yaofeng Sun, Chengqi Deng, Hanwei Xu, Zhenda Xie, Chong Ruan.
- **Model type:** [Deepseek-VL-Hybrid](https://huggingface.co/docs/transformers/main/en/model_doc/deepseek_vl_hybrid)
- **License:** deepseek

### Model Sources

<!-- Provide the basic links for the model. -->

- **HF Docs:** [Deepseek-VL-Hybrid](https://huggingface.co/docs/transformers/main/en/model_doc/deepseek_vl_hybrid)
- **Repository:** https://github.com/deepseek-ai/DeepSeek-VL
- **Paper:** https://arxiv.org/abs/2403.05525

## How to Get Started with the Model

> **Note:** Ensure you have `transformers` version **4.54.0** or later installed:
>
> ```bash
> pip install -U "transformers>=4.54.0"
> ```

The example below demonstrates how to generate text based on an image with `Pipeline`.

```py
import torch
from transformers import pipeline

pipe = pipeline(
    task="image-text-to-text",
    model="deepseek-community/deepseek-vl-7b-base",
    device=0,
    torch_dtype=torch.float16
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
            },
            { "type": "text", "text": "Describe this image."},
        ]
    }
]

pipe(text=messages, max_new_tokens=20, return_full_text=False)
```

Generate text based on an image with `AutoModel`.

```py
import torch
from transformers import DeepseekVLHybridForConditionalGeneration, AutoProcessor

model = DeepseekVLHybridForConditionalGeneration.from_pretrained(
    "deepseek-community/deepseek-vl-7b-base",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

processor = AutoProcessor.from_pretrained("deepseek-community/deepseek-vl-7b-base")

messages = [
    {
        "role":"user",
        "content":[
            {
                "type":"image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
            },
            {
                "type":"text",
                "text":"Describe this image."
            }
        ]
    }

]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device, dtype=model.dtype)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text)
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](https://huggingface.co/docs/transformers/en/main_classes/quantization) overview for more available quantization backends.

The example below uses [TorchAo](https://huggingface.co/docs/transformers/en/main_classes/quantization#transformers.TorchAoConfig) to only quantize the weights to int4.

```py
import torch
from transformers import TorchAoConfig, DeepseekVLHybridForConditionalGeneration, AutoProcessor

quantization_config = TorchAoConfig(
    "int4_weight_only",
    group_size=128
)

model = DeepseekVLHybridForConditionalGeneration.from_pretrained(
    "deepseek-community/deepseek-vl-7b-base",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)
```

Do inference with multiple images in a single conversation.

```py
import torch
from transformers import DeepseekVLHybridForConditionalGeneration, AutoProcessor

model = DeepseekVLHybridForConditionalGeneration.from_pretrained(
    "deepseek-community/deepseek-vl-7b-base",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

processor = AutoProcessor.from_pretrained("deepseek-community/deepseek-vl-7b-base")

messages = [
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s the difference between"},
                {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
                {"type": "text", "text": " and "},
                {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"}
            ]
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"},
                {"type": "text", "text": "What do you see in this image?"}
            ]
        }
    ]
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    padding=True,
    truncation=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device, dtype=model.dtype)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text)
```

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

The Deepseek-VL-Hybrid model was trained on the following datasets:

![image/jpeg](assets/datasets.png)

### Training Pipeline

Training pipelines consist of three stages.
- Stage 1 involves training the Vision-Language (VL) adaptor while keeping the hybrid vision encoder and language model fixed.
- Stage 2 is the crucial part of the joint vision and language pretraining, where both VL adaptor and language model are trainable.
- Stage 3 is the supervised fine-tuning phase, during which the low-resolution vision encoder SigLIP-L, VL adaptor, and language model will be trained

![image/jpeg](assets/training_pipeline.png)

### Training Hyperparameters

![image/jpeg](assets/hyperparameters.png)

## Evaluation

![image/png](assets/evaluation.png)

## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

```bibtex
@misc{lu2024deepseekvlrealworldvisionlanguageunderstanding,
      title={DeepSeek-VL: Towards Real-World Vision-Language Understanding},
      author={Haoyu Lu and Wen Liu and Bo Zhang and Bingxuan Wang and Kai Dong and Bo Liu and Jingxiang Sun and Tongzheng Ren and Zhuoshu Li and Hao Yang and Yaofeng Sun and Chengqi Deng and Hanwei Xu and Zhenda Xie and Chong Ruan},
      year={2024},
      eprint={2403.05525},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2403.05525},
}
```

## Model Card Authors

[Armaghan Shakir (geetu040)](https://github.com/geetu040)
