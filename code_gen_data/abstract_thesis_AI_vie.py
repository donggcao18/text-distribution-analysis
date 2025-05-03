import json
import os
from data_generation.model_gen import get_deepseek_response, get_gemini_response, get_llama_response, get_gpt4omini_response

def generate_abstract(file_name):
    system_prompt = '''
    # Bạn là một sinh viên đại học, có kỹ năng viết abstract ngắn gọn, súc tích.
    # Hãy viết một đoạn abstract KHOẢNG 400 - 700 CHỮ dựa trên chủ đề thể hiện qua tên tệp được cung cấp.
    # Abstract PHẢI ĐƯỢC VIẾT LIỀN MẠCH, KHÔNG XUỐNG DÒNG, KHÔNG CHIA ĐOẠN.
    # KHÔNG ĐƯỢC bắt đầu bằng các cụm từ như "Dựa trên", "Theo", "Tài liệu này" hoặc bất cứ cụm từ tương tự nào. Hãy đi thẳng vào vấn đề, trình bày súc tích, khoa học, đúng phong cách viết academic.
    # Viết HOÀN TOÀN BẰNG TIẾNG VIỆT, sử dụng câu văn chuẩn xác, dễ hiểu, không mắc lỗi chính tả.
    # KHÔNG ĐƯỢC đề cập đến 'tên tệp', 'tập tin', 'file', 'tài liệu', 'nội dung', 'tác giả' hay người viết trong đoạn abstract.
    # Chủ đề abstract: "{file_name}"
    '''

    user_prompt = f"Tạo abstract cho tệp có tên: {file_name}"


    responses = {}
    responses["gpt4omini"] = get_gpt4omini_response(system_prompt, user_prompt)
    responses["gemini"] = get_gemini_response(system_prompt, user_prompt)
    responses["deepseek"] = get_deepseek_response(system_prompt, user_prompt)
    responses["llama"] = get_llama_response(system_prompt, user_prompt)
    
    return responses

def load_records(input_file):
    records = []
    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records.append(record)
    return records

def main():
    mapping_details = {
        "gpt4omini": "GPT-4o-mini",
        "gemini": "Gemini 2.0",
        "deepseek": "Deepseek",
        "llama": "Llama"
    }

    input_file_2 = r"E:\Career\Research\TNM's thesis\authscan\data\human\graduation_thesis\abstract_graduation_thesis\data\vietnamese_cleaned.jsonl"
    records_2 = load_records(input_file_2)

    temporary_folder_2 = r"E:\Career\Research\TNM's thesis\authscan\data\temporary_thesis"
    os.makedirs(temporary_folder_2, exist_ok=True)

    with open("abstract_thesis_ai_gen.jsonl", "w", encoding="utf-8") as fout:
        for record in records_2:
            file_name = record.get("file_name")
            if not file_name:
                continue

            responses = generate_abstract(file_name)
            original_id = record.get("ID", "")
            for model, abstract in responses.items():
                new_item = {
                    "oid": original_id,
                    "id": f"{original_id}_thesis_{model}",
                    "text": abstract,
                    "label": "AI",
                    "label_detailed": mapping_details.get(model, model)
                }
                
                fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")
                
                temp_file_name = os.path.join(temporary_folder_2, f"{new_item['id']}.jsonl")
                with open(temp_file_name, "w", encoding="utf-8") as temp_out:
                    temp_out.write(json.dumps(new_item, ensure_ascii=False) + "\n")
                    
                print(new_item)
    


if __name__ == "__main__":
    main()
