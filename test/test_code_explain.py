import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bertscore

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
smoothing = SmoothingFunction().method1

def compute_bleu(reference, hypothesis):
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    if len(hyp_tokens) == 0:
        return 0.0
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)

def compute_bertscore(reference, hypothesis, lang='en'):
    # ref, hyp는 각각 리스트(1개 이상)여야 함
    P, R, F1 = bertscore([hypothesis], [reference], lang=lang, rescale_with_baseline=True)
    return float(F1[0])  # 단일 샘플 기준

def generate_code_summary(code, tokenizer, model, device='cuda', max_new_tokens=128):
    prompt = f"Generate a concise and clear explanation for the following code:\n\n{code}\n\nExplanation:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.95,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    summary = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return summary.strip()

def main():
    model_dir = "jack0503/my-hf-model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    model.to(device)
    model.eval()

    dataset = load_dataset("code_x_glue_ct_code_to_text", "python")["validation"]
    total, total_bleu, total_bertscore = 0, 0.0, 0.0

    for i, sample in enumerate(dataset):
        code = sample.get("code", "")
        reference = sample.get("docstring", "")
        if not code or not reference:
            continue

        generated_summary = generate_code_summary(code, tokenizer, model, device=device)
        bleu_score = compute_bleu(reference, generated_summary)
        bert_score = compute_bertscore(reference, generated_summary)  # 의미 유사도 기반 점수

        total += 1
        total_bleu += bleu_score
        total_bertscore += bert_score

        print(f"[{i}] BLEU: {bleu_score:.4f}  |  BERTScore(F1): {bert_score:.4f}")
        print(f"Reference: {reference[:100]}...")
        print(f"Generated: {generated_summary[:100]}...")
        print("=" * 30)

        if i >= 99:  # 최대 100 샘플만 평가
            break

    print(f"평균 BLEU: {(total_bleu/total):.4f} ({total} samples)")
    print(f"평균 BERTScore(F1): {(total_bertscore/total):.4f}")

if __name__ == "__main__":
    main()
