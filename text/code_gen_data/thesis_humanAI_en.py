import os
from openai import OpenAI
import jsonlines

system_content = "You are an AI language model tasked with paraphrasing the following paragraph, which has been extracted from the thesis of an undergraduate student.Paraphrase the given thesis content while preserving its original meaning and context. Maintain clarity, coherence, and an academic tone. OUTPUT THE PARAPHASED TEXT ONLY, WITHOUT ANY INTRODUCTION OR EXPLAINATION.THE OUTPUT SHOULD BE ONLY ONE PARAGRAPH.DO NOT PARAPHRASE: References, figure labels (e.g., Figure A, B, C), and citations should remain unchanged."
model_list = {
    "gpt-4o-mini": "GPT-4o-mini", 
    "deepseek-chat": "Deepseek V3", 
    "gemini-2.0-flash": "Gemini 2.0", 
    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free": "Llama-3.3-70B-Instruct-Turbo"}
input_file = r"data/human/graduation_thesis/data/english_cleaned.jsonl"
output_file = r"data/human/graduation_thesis/data/human+ai.jsonl"

client_llama = OpenAI(api_key=os.environ.get("LLAMA_API_KEY"), base_url="https://api.together.xyz/v1")
client_gpt = OpenAI(api_key=os.environ.get("GPT_API_KEY_2"))
client_deepseek = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
client_gemini = OpenAI(api_key=os.environ.get("GEMINI_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

data = []
with jsonlines.open(input_file) as file:
    for line in file:
        data.append(line)

def get_user_content(passage):
    return f"Here is the paragraph to paraphrase: {passage}"

count = 0
for item in data[0:]:
    passage = item['content']
    origin_id = item['ID']
    try:
        for model_name, model in model_list.items():
            process = []
            if model == 'Gemini 2.0':
                client = client_gemini
            elif model == 'GPT-4o-mini':
                client = client_gpt
            elif model == 'Deepseek V3':
                client = client_deepseek
            elif model == 'Llama-3.3-70B-Instruct-Turbo':
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

            if process:
              with jsonlines.open(output_file, mode="a") as file:
                file.write_all(process)
    except Exception as e:
        print(f"Error with item {origin_id}: {e}")
        stop_process = True
        break

