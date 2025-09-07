import pandas as pd
import json

# CSV 파일 로드
df = pd.read_csv("./dataset/train.csv")
print(f"원본 데이터 샘플 수: {len(df)}")

# 문자열로 변환 후 공백 제거
df["input"] = df["input"].astype(str).str.strip()
df["output"] = df["output"].astype(str).str.strip()

# 전처리 조건: input/output이 비어있거나 'not applicable'이 포함된 행 제거
filtered_df = df[
    (df["input"].str.lower() != "Not applicable") &
    (df["output"].str.lower() != "Not applicable") &
    (df["input"].str.lower() != "nan") &
    (df["input"] != "") &
    (df["output"] != "")
].copy()

filtered_df = filtered_df[filtered_df["prompt"].str.contains("def")].copy()

def extract_def_content(text):
    """Extracts the content of the first 'def' function in the text."""
    idx = text.find("def ")
    return text[idx:] if idx != -1 else text

# 원하는 형식으로 JSON 리스트 변환
json_data = filtered_df[["input", "output", "instruction", "prompt"]].to_dict(orient="records")

# JSON 저장
with open("cleaned_data1.json", "w", encoding="utf-8") as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

print(f"전처리 후 샘플 수: {len(json_data)}")
