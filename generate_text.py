from transformers import pipeline


def get_generator(model_name="gpt2"):
    generator = pipeline("text-generation", model=model_name)
    return generator


def generate_text(generator, prompt, max_len=100, temp=0.7, top_k=50):
    output = generator(
        prompt,
        max_length=max_len,
        temperature=temp,
        top_k=top_k,
        num_return_sequences=1
    )
    return output[0]['generated_text']
