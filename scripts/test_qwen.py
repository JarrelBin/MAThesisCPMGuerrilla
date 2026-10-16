
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


MODEL_NAME = "Qwen/Qwen3-8B"


def main():

    print(f"Loading {MODEL_NAME}...")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not available."
        )

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    # 4-bit NF4 quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        dtype=torch.float16,
        device_map={"": 0},
    )

    print(
        f"Model device: {model.device}"
    )

    print(
        f"GPU memory allocated after loading: "
        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You identify OCR recognition "
                "errors in historical documents. "
                "Preserve historical vocabulary, "
                "names, and original wording. "
                "Do not rewrite correctly "
                "recognized text."
            ),
        },
        {
            "role": "user",
            "content": (
                "OCR text:\n"
                "Subject was assigned to the "
                "Special Distriot in 1978.\n\n"
                "Which token is probably an OCR "
                "error? Answer briefly."
            ),
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {
        key: value.to("cuda")
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )

    print()
    print("=== QWEN RESPONSE ===")
    print()
    print(response)

    print()
    print("=== GPU MEMORY ===")

    print(
        f"Allocated: "
        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )

    print(
        f"Peak allocated: "
        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
    )


if __name__ == "__main__":
    main()
