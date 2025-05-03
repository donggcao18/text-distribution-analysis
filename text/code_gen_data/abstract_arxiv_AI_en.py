import json
import os
import google.generativeai as genai

genai.configure(api_key="")  

model = genai.GenerativeModel("gemini-2.0-flash")

system_prompt = """
You are a university student working in the field of computer science. 
You have been assigned to write the abstract section for a scientific paper based on the given paper title. 
Only output is given, no explanation of how to write, no citation of article link with the name given.
In the field of Information Technology (IT), the abstract of a scientific paper must meet several key requirements to ensure clarity, accuracy, and value for readers. Your task is to generate an abstract for this paper following the given guidelines:
1. Clear Structure
An abstract in IT typically follows a standard structure, including: 
- Objective/Purpose: A brief introduction to the problem or research question addressed. 
- Methods/Approach: Description of the approach, model, algorithm, or technology used. 
- Results/Findings: Summary of key findings, system performance, or comparisons with other methods. 
- Conclusion/Implications: Important insights, contributions of the research, and potential applications. 
2. Concise and Succinct
- The abstract is typically limited to 150-300 words (depending on the journal or conference requirements). 
- Avoid lengthy explanations or excessive theoretical background. 
3. Avoid Excessive Technical Jargon
- While IT is a technical field, the abstract should be clear and understandable, even for readers who are not deeply specialized in a specific subfield. 
- If abbreviations or technical terms are used, ensure they are widely recognized or briefly explained. 
4. No Citations or References
- The abstract should be self-contained and not rely on external references.
5. Emphasize Novelty and Contributions
- Highlight what makes the research unique compared to previous works.
- If the study involves AI, Machine Learning, 3D Reconstruction, or other advanced technologies, clearly state the innovations or improvements. 
6. Include Key Keywords
- Helps the paper be easily discoverable in scientific databases. 
Example: For research on large language models, relevant keywords might include GPT, transformer, fine-tuning, AI models. 
"""

def generate_text(prompt, problem_id):
    try:
        full_prompt = f"{system_prompt}\n\n{prompt}"  
        response = model.generate_content(full_prompt)
        return response.text if response else ""
    except Exception as e:
        print(f"Generation failed - Problem ID: {problem_id}, Error: {str(e)}")
        return ""

base_path = ""
inputfile = os.path.join(base_path, "")
output_jsonl = os.path.join(base_path, "")

solution_id_counter = 1

with open(inputfile, 'r', encoding='utf-8') as file, open(output_jsonl, "a", encoding="utf-8") as output_file:
    for line in file:
        try:
            problem_data = json.loads(line)
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line: {line}")
            continue
        
        problem_id = problem_data.get("ID", "unknown")
        problem_title = problem_data.get("file_name", "")

        if not problem_title:
            print(f"Skipping problem ID {problem_id} due to missing title")
            continue

        prompt = f"Title of the paper: {problem_title}\n\nYour task is to generate an abstract for this paper following the given guidelines."

        generated_text = generate_text(prompt, problem_id)

        entry = {
            "solution_id": f"gemini-{solution_id_counter:04d}",
            "problem_id": problem_id,
            "content": generated_text if generated_text else "",
            "label": "AI",
            "label_detailed": "Gemini 2.0"
        }

        output_file.write(json.dumps(entry) + "\n")
        output_file.flush()  

        print(f"✅ Processed record {solution_id_counter}: Problem ID {problem_id}")

        solution_id_counter += 1

print(f"Dataset saved to {output_jsonl}")
