import json
from sklearn.model_selection import train_test_split

# 파일 경로 설정
input_file = "cleaned_data.json"

# JSON 데이터 불러오기
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# instruction과 output만 추출
filtered_data = [
    {"instruction": item["instruction"], "output": item["output"], "prompt": item["prompt"]}
    for item in data
]

# train/test 분할 (80/20)
train_data, temp_data = train_test_split(filtered_data, test_size=0.2, random_state=42)

val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# 저장
with open("train_data.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open("test_data.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

with open("val_data.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)
