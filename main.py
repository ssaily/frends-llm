from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=torch.float16,
)

# Prepare model for training
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    use_rslora=False,
)

# Load dataset
dataset = load_dataset("json", data_files="datasets/frends-bpmn-dataset.jsonl", split="train")

# Format dataset for chat template
def format_chat_template(example):
    return {
        "text": tokenizer.apply_chat_template(
            [
                {"role": "user", "content": example["instruction"] + "\n" + example.get("input", "")},
                {"role": "assistant", "content": example["output"]}
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
    }

dataset = dataset.map(format_chat_template, remove_columns=["instruction", "input", "output"])

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    weight_decay=0.01,
    learning_rate=2e-4,
    save_steps=100,
    logging_steps=10,
    optim="paged_adamw_8bit",
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
)

# Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=2048,
)

# Train
trainer.train()

# Save model
model.save_pretrained("./llama-3.2-1b-instruct-finetuned")
tokenizer.save_pretrained("./llama-3.2-1b-instruct-finetuned")