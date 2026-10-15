import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


MODEL_NAME = "Qwen/Qwen3-4B"


def main():

    print(
        f"Loading {MODEL_NAME}..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto",
        )
    )

    print(
        f"Model device: {model.device}"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You identify OCR recognition "
                "errors. Do not rewrite text."
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

    inputs = (
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,

            # Important for this task:
            enable_thinking=False,

            return_tensors="pt",
            return_dict=True,
        )
    )

    inputs = {
        key: value.to(
            model.device
        )
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
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


if __name__ == "__main__":
    main()
