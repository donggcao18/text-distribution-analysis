import os
import random
import jsonlines
from openai import OpenAI

# System prompt
system_content = (
    "You are an AI language model tasked with paraphrasing the following paragraph, "
    "which has been extracted from the abstract of the science paper in computer science domain. "
    "Paraphrase the given abstract content while preserving its original meaning and context. "
    "Maintain clarity, coherence, and an academic tone. OUTPUT THE PARAPHRASED TEXT ONLY, "
    "WITHOUT ANY INTRODUCTION OR EXPLAINATION. THE OUTPUT SHOULD BE ONLY ONE PARAGRAPH. "
    "DO NOT PARAPHRASE: References, figure labels (e.g., Figure A, B, C), and citations should remain unchanged."
)

# Model mapping
model_map = {
    "gpt-4o-mini": {
        "name": "GPT-4o-mini",
        "client": OpenAI(
            api_key=""
        )
    },
    "deepseek-chat": {
        "name": "Deepseek V3",
        "client": OpenAI(
            api_key="",
            base_url="https://api.deepseek.com"
        )
    },
    "gemini-2.0-flash": {
        "name": "Gemini 2.0",
        "client": OpenAI(
            api_key="",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    },
    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free": {
        "name": "Llama-3.3-70B-Instruct-Turbo",
        "client": OpenAI(
            api_key="",
            base_url="https://api.together.xyz/v1"
        )
    }
}

# File paths
input_file = r""
output_file = r""

# Function to build user prompt
def get_user_content(passage):
    return f"Here is the paragraph to paraphrase: {passage}"

# Load input data
with jsonlines.open(input_file) as file:
    data = list(file)

solution_id_counter = 1

# Process each record with a randomly chosen model
for item in data:
    passage = item['content']
    origin_id = item['ID']

    # Chọn ngẫu nhiên 1 model
    model_key = random.choice(list(model_map.keys()))
    model_info = model_map[model_key]
    model_client = model_info["client"]
    model_display = model_info["name"]

    try:
        user_content = get_user_content(passage)
        response = model_client.chat.completions.create(
            model=model_key,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]
        )

        # Tạo bản ghi kết quả
        entry = {
            "solution_id": f"{model_display.lower().replace(' ', '').replace('+', '').replace('.', '')}-{solution_id_counter:04d}",
            "problem_id": origin_id,
            "text": response.choices[0].message.content.strip(),
            "label": "human+AI",
            "label_detailed": f"human + {model_display}"
        }

        with jsonlines.open(output_file, mode="a") as outfile:
            outfile.write(entry)

        print(f"✅ {entry['solution_id']} | {origin_id} | {model_display}")
        solution_id_counter += 1

    except Exception as e:
        print(f"❌ Error with {origin_id} using {model_display}: {e}")
        continue
