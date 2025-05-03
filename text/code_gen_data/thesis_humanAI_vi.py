import os
import random
from openai import OpenAI
import jsonlines

system_content = "Bạn là một mô hình ngôn ngữ AI được giao nhiệm vụ diễn đạt lại đoạn văn dưới đây, được trích từ luận văn của một sinh viên đại học. Hãy viết lại nội dung theo cách tự nhiên, rõ ràng và mạch lạc nhưng vẫn giữ nguyên ý nghĩa, ngữ cảnh ban đầu và phong cách học thuật. Nếu đoạn văn chứa thuật ngữ chuyên ngành hoặc từ ngữ tiếng Anh sử dụng có chủ đích, hãy giữ nguyên để đảm bảo tính chính xác và nhất quán. Yêu cầu quan trọng:Chỉ đưa ra nội dung đã diễn đạt lại, không thêm phần giới thiệu hoặc giải thích, đầu ra phải là một đoạn văn duy nhất, không chia thành nhiều đoạn., không diễn đạt lại tài liệu tham khảo, nhãn hình (ví dụ: Hình A, B, C) và trích dẫn, hãy giữ nguyên các phần này như trong đoạn gốc."

model_list = {
    "gpt-4o-mini": "GPT-4o-mini", 
    "deepseek-chat": "Deepseek V3", 
    "gemini-2.0-flash": "Gemini 2.0", 
    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free": "Llama-3.3-70B-Instruct-Turbo"
}

input_file = r"data/generated/text/human/graduation_thesis/data/vietnamese_cleaned.jsonl"
output_file = r"data/generated/text/graduation_thesis/data/vie_data.jsonl"

client_llama = OpenAI(api_key=os.environ.get("LLAMA_API_KEY"), base_url="https://api.together.xyz/v1")
client_gpt = OpenAI(api_key=os.environ.get("GPT_API_KEY_2"))
client_deepseek = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
client_gemini = OpenAI(api_key=os.environ.get("GEMINI_API_KEY_2"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

data = []
with jsonlines.open(input_file) as file:
    for line in file:
        data.append(line)

def get_user_content(passage):
    return f"Dưới đây là đoạn văn cần được diễn đạt lại: {passage}"

count = 0

for item in data[0:]:
    passage = item['content']
    origin_id = item['ID']

    try:
        selected_models = random.sample(list(model_list.items()), 2)
        for model_name, model in selected_models:
            process = []
            if model_name == 'gemini-2.0-flash':
                client = client_gemini
            elif model_name == 'gpt-4o-mini':
                client = client_gpt
            elif model_name == 'deepseek-chat':
                client = client_deepseek
            elif model_name == 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free':
                client = client_llama

            user_content = get_user_content(passage)

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ]
            )

            count += 1

            process.append({
                "oid": origin_id,
                "id": count,
                "text": response.choices[0].message.content,
                "label": "human+AI",
                "label_detailed": f"human + {model}"
            })

            with jsonlines.open(output_file, mode="a") as file:
                file.write_all(process)

    except Exception as e:
        print(f"Error with item {origin_id}: {e}")
        break  
