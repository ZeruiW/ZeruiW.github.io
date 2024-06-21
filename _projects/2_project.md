---
layout: page
title: LLAMA Job Skills Extractor Project
description: AI-powered tool for extracting key skills from job descriptions
img: assets/img/llama2logo.jpeg
importance: 1
category: work
related_publications: true
---

# LLAMA Job Skills Extractor: AI-Powered Tool for Identifying Key Skills from Job Descriptions

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/transformervsllama.webp" title="LLAMA Model" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

## [**Model: Visit LLAMA Job Skills Extractor on Hugging Face**](https://huggingface.co/wangzerui/Job-Skiils-Analysis)

[https://huggingface.co/wangzerui/Job-Skiils-Analysis](https://huggingface.co/wangzerui/Job-Skiils-Analysis)

The LLAMA Job Skills Extractor is a cutting-edge AI-powered tool designed to analyze job descriptions and extract the key skills required for each position. This project leverages the powerful LLAMA 2 model, utilizing advanced techniques such as quantization and Low-Rank Adaptation (LoRA) to optimize performance and efficiency. Here’s a detailed overview of the features and functionalities of the LLAMA Job Skills Extractor:

### Key Features

- **Efficient Data Processing**: Clean and format job descriptions to maximize model understanding.
- **Advanced Model Fine-Tuning**: Use of LLAMA 2 with 4-bit or 8-bit quantization for efficient training.
- **Low-Rank Adaptation (LoRA)**: Reducing the number of trainable parameters while maintaining model performance.
- **Comprehensive Skill Extraction**: Analyze job descriptions to accurately identify required skills.
- **Real-Time Performance Tracking**: Monitor training metrics with Weights & Biases for continuous optimization.

### Technologies Used

- **LLAMA 2 13B Model**: A state-of-the-art language model architecture.
- **Quantization**: Reducing the precision of model weights to decrease memory usage and increase training speed.
- **LoRA (Low-Rank Adaptation)**: A technique to reduce the number of trainable parameters, making fine-tuning more efficient.
- **Weights & Biases**: A tool for tracking and visualizing training metrics, aiding in optimization and debugging.
- **PEFT (Parameter-Efficient Fine-Tuning)**: A library used to prepare models for k-bit training, optimizing resource usage.

## Project Workflow

### Environment Setup

Setting up the environment involves installing necessary libraries and configuring system settings to ensure smooth operation. Key libraries include `transformers` for the model, `datasets` for handling data, and `wandb` for tracking experiments.

```bash
pip install transformers datasets wandb peft
```

We start by loading the necessary environment and libraries, ensuring that all dependencies are correctly installed.

```python
import os
import wandb
from transformers import LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, integrate_lora
```

### Loading and Formatting the Dataset

The first step is to load and format the dataset, ensuring that the job descriptions are properly structured for model training. We use the `datasets` library to load our dataset.

```python
dataset = load_dataset('your_dataset')
```

Formatting the data is crucial for ensuring the model can effectively learn from it. We define a function to format each job description.

```python
def formatting_func(example):
    text = f"### Job Description: {example['job_description']}"
    return {'formatted_text': text}
    
dataset = dataset.map(formatting_func)
```

### Loading the Base Model

We use the LLAMA 2 13B model, employing quantization to manage computational resources efficiently. This involves loading the model in 8-bit precision.

```python
model_name = 'meta-llama/Llama-2-13b-hf'
model = LlamaForCausalLM.from_pretrained(model_name, load_in_8bit=True)
tokenizer = LlamaTokenizer.from_pretrained(model_name)
```

### Tokenization

Tokenization is critical for preparing text data for training. We analyze the token length distribution to set an appropriate `model_max_length`, ensuring efficient processing of inputs.

```python
token_lens = [len(tokenizer.encode(text)) for text in dataset['formatted_text']]
max_len = max(token_lens)
print(f"Maximum token length: {max_len}")
```

### Setting Up LoRA

We use LoRA to inject low-rank matrices into the model’s architecture, reducing the number of trainable parameters. This step involves preparing the model for k-bit training and configuring the LoRA parameters.

```python
model = prepare_model_for_kbit_training(model, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'lm_head'])

lora_config = LoraConfig(r=32, alpha=64)
model = integrate_lora(model, lora_config)
```

### Tracking Training Metrics with Weights & Biases

Using Weights & Biases to monitor the training process is crucial for tracking various metrics and optimizing the model. This tool helps visualize the model's performance in real-time.

```python
wandb.init(project='llama-job-skills')
model.train()
```

### Training the Model

Training involves running the model through multiple epochs to fine-tune it. We use a training loop to update the model weights based on the loss computed from the predictions and actual values.

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
)

trainer.train()
```

### Evaluating the Model

Evaluating the model's performance on a validation set involves calculating metrics such as precision, recall, and F1-score to ensure the model generalizes well to unseen data.

```python
from sklearn.metrics import precision_score, recall_score, f1_score

preds, labels = [], []
for batch in dataset['validation']:
    outputs = model(**batch)
    preds.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
    labels.extend(batch['labels'].cpu().numpy())

precision = precision_score(labels, preds, average='weighted')
recall = recall_score(labels, preds, average='weighted')
f1 = f1_score(labels, preds, average='weighted')

wandb.log({"precision": precision, "recall": recall, "f1": f1})
```

[https://api.wandb.ai/links/zeruiw/9ogsmdst](https://api.wandb.ai/links/zeruiw/9ogsmdst)

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/llamawandb.png" title="LLAMA Job Skills Extractor Interface" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    WanDB training metadata analysis.
</div>

## Conclusion

The LLAMA Job Skills Extractor exemplifies the powerful integration of AI technology with job description analysis, offering precise, data-driven insights. The project showcases the application of advanced AI models and efficient fine-tuning techniques, resulting in a practical tool that can significantly benefit recruiters and job seekers.

