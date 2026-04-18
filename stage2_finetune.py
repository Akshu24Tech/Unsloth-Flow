import os, torch
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

HF_TOKEN        = os.environ["HF_TOKEN"]
HF_DATASET_REPO = "Akshu2424/rick-llm-dataset-2026"
HF_MODEL_REPO   = "Akshu2424/rick-llm-qwen3"
BASE_MODEL      = "unsloth/Qwen3-4B"
OUTPUT_DIR      = "./outputs/rick-llm-qwen3"


def main():
    login(token=HF_TOKEN)

    # Load model
    model, tokenizer = FastModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=2048,
        load_in_4bit=True,
        token=HF_TOKEN,
    )

    # Apply LoRA
    model = FastModel.get_peft_model(
        model,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16, lora_alpha=16, lora_dropout=0,
        bias="none", random_state=42,
        use_gradient_checkpointing="unsloth",
    )
    model.print_trainable_parameters()

    # Dataset
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-3")
    dataset = load_dataset(HF_DATASET_REPO, split="train", token=HF_TOKEN)

    def format_conversations(examples):
        return {"text": [
            tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False, enable_thinking=False)
            for c in examples["conversations"]
        ]}

    dataset = dataset.map(format_conversations, batched=True)
    print(f"Dataset: {len(dataset)} samples")

    # Train
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=512,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=120,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=OUTPUT_DIR,
            report_to="none",
        ),
    )
    stats = trainer.train()
    print(f"Done in {round(stats.metrics['train_runtime']/60, 2)} min")

    # Save GGUF + push
    os.makedirs("./ollama_files", exist_ok=True)
    model.save_pretrained_gguf("./ollama_files", tokenizer, quantization_method="q4_k_m")
    model.push_to_hub_gguf(HF_MODEL_REPO, tokenizer, quantization_method="q4_k_m", token=HF_TOKEN)

    # Write Modelfile
    with open("./ollama_files/Modelfile", "w") as f:
        f.write("""FROM ./rick-llm-qwen3-unsloth.Q4_K_M.gguf

SYSTEM \"\"\"You are Rick Sanchez — sarcastic, nihilistic, brilliant. Burp mid-sentence (*burp*). \\
Reference science and the multiverse. Use 'Morty', 'wubba lubba dub dub'. Keep it punchy.\"\"\"

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER top_k 20
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
""")
    print(f"Pushed to https://huggingface.co/{HF_MODEL_REPO}")
    print("Modelfile written to ./ollama_files/Modelfile")


if __name__ == "__main__":
    main()
