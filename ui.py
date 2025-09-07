import os
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# 모델 경로
model_dir = "./finetuned_model/finetuned_V1_quantized_pruned"  # 또는 'final' 경로
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda", torch_dtype='auto')

def chat(instruction, input_text=""):
    full_instruction = (
        "Strictly follow the user's requirements without exception. When generating code or providing explanations:\n"
        "1. Always adhere precisely to the input specifications and constraints.\n"
        "2. Ensure that all generated code is syntactically correct and follows best programming practices.\n"
        "3. Provide code explanations only in clear, grammatically correct English, maintaining proper technical language.\n"
        "4. Do not include any unnecessary or unrelated information; only respond with what is requested.\n"
        "5. If the requirements are ambiguous, ask for clarification instead of making assumptions.\n"
        "6. Structure your output with clear separation between code and explanations.\n"
        "7. Code must be fully tested and free of any logical or syntactical errors.\n"
        "8. If there is expected output, code ouput must be same with expected one.\n"
    )
    prompt = f"### Instruction:\n{instruction}\n\n"
    if input_text.strip():
        prompt += f"### Input:\n{input_text}\n\n"
    prompt += "### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=2048,
        temperature=0.7,
        do_sample=True,
        pad_token_id = tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:\n")[1].strip() if "### Response:" in response else response

demo = gr.Interface(
    fn=chat,
    inputs=[
        gr.Textbox(label="Instruction"),
        gr.Textbox(label="Input ")
    ],
    outputs="text",
    title="Prototype Code Assistant"
)

demo.launch(share=True)  # share=True면 외부 접속도 가능
