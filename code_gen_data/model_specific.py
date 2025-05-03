import os
from openai import OpenAI
import jsonlines

system_content = "You are an AI language model tasked with generating a new abstract section based on an extracted passage that summarizes the key idea from an arxiv abstract.\
                Using the extracted idea as your reference, compose a fully AI-authored paragraph that clearly and coherently expresses the concept in academic language.\
                Instructions:\
                Use only the extracted idea as the foundation for your writing.\
                Do not attempt to reconstruct or paraphrase any original paragraph.\
                The final output should be one logically structured paragraph, written in a clear, academic tone.\
                Do not fabricate citations, and avoid using placeholder references like “Figure 1” unless explicitly included in the extracted idea.\
                Do not include introductory phrases or meta-instructions in your output — return only the generated paragraph."

model_list = {"gemini-2.0-flash": "Gemini 2.0", 
              "gemini-2.0-flash-lite": "Gemini 2.0 Flash-Lite",
              "gemini-1.5-pro":"Gemini 1.5 Pro",
              "gemini-1.5-flash":"Gemini 1.5 Flash",
}

input_file = r"data/text/labels/human/abstract_arxiv_human_en.jsonl"
output_file = r"data/text/code_gen_data/abstract_gemini.jsonl"

client = OpenAI(api_key=os.environ.get("GEMINI_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

data = []
with jsonlines.open(input_file) as file:
    for line in file:
        data.append(line)


def summerize_content(passage):
    prompt = f"Summarize the main idea of the following abstract briefly but still include all the necessary information:\n{passage}"
    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content


count = 0
for item in data[0:]:
    passage = item['content']
    origin_id = item['ID']
    try:
        if passage:
            summerize_passage = summerize_content(passage)
        for model_name, model in model_list.items():
            process = []
            user_content = f"Write a paragraph based on the content below.\nAvoid starting the paragraph with phrases like 'Here is the new paragraph:'.\nThe length of the paragraph should be close to {len(passage)} characters.\nHere is the extracted information from the original passageContent: {summerize_passage}"
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
                "content": response.choices[0].message.content,
                "label": "AI",
                "label_detailed": f"{model}"
            })

            if process:
              with jsonlines.open(output_file, mode="a") as file:
                file.write_all(process)
    except Exception as e:
        print(f"Error with item {origin_id}: {e}")
        stop_process = True
        break

