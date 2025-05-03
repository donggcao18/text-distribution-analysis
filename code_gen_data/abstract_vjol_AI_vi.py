import json
import os
from data-generation.model_gen import get_deepseek_response, get_gemini_response, get_llama_response, get_gpt4omini_response

def generate_abstract(file_name):
    system_prompt = '''
    # Bạn là một nhà báo, có kỹ năng viết tóm tắt súc tích.
    # Hãy viết một đoạn văn ngắn KHOẢNG 150 - 200 CHỮ dựa trên chủ đề được thể hiện qua tên tệp được cung cấp.  
    # ĐOẠN VĂN PHẢI ĐƯỢC VIẾT LIỀN MẠCH, KHÔNG XUỐNG DÒNG, KHÔNG CHIA ĐOẠN.  
    # KHÔNG ĐƯỢC bắt đầu bằng các cụm từ như "Dựa trên", "Theo", "Nội dung tệp", "Tài liệu này" hoặc bất cứ cụm từ tương tự nào. Ngay lập tức đi thẳng vào chủ đề, trình bày rõ ràng, khách quan, đúng phong cách một bản tin ngắn gọn.  
    # Phải viết HOÀN TOÀN BẰNG TIẾNG VIỆT, tuyệt đối KHÔNG sử dụng tiếng Anh. Viết đúng chính tả, câu từ dễ đọc, dễ hiểu.  
    # KHÔNG ĐƯỢC đề cập đến cụm từ 'tên tệp', 'tập tin', 'file', 'tài liệu', 'nội dung', 'tác giả' hay người viết trong đoạn văn.  
    # Tên chủ đề: "{file_name}"
    '''
    

    user_prompt = f"Tạo tóm tắt cho tệp có tên: {file_name}"




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
    input_file_1 = r"E:\Career\Research\TNM's thesis\authscan\data\human\vjol_abstract\data\vjol_abstract_cleaned.jsonl"
    records_1 = load_records(input_file_1)
    mapping_details = {
        "gpt4omini": "GPT-4o-mini",
        "gemini": "Gemini 2.0",
        "deepseek": "Deepseek",
        "llama": "Llama"
    }
    temporary_folder_1 = r"E:\Career\Research\TNM's thesis\authscan\data\temporary_vjol"
    os.makedirs(temporary_folder_1, exist_ok=True)
    
    with open("abstract_vjol_ai_gen.jsonl", "w", encoding="utf-8") as fout:
        for record in records_1:
            file_name = record.get("file_name")
            id = record.get("ID")
            if not file_name:
                continue
            responses = generate_abstract(file_name)
            original_id = record.get("ID", "")
            for model, abstract in responses.items():
                new_item = {
                    "oid": original_id,
                    "id": f"{original_id}_vjol_{model}",
                    "text": abstract,
                    "label": "AI",
                    "label_detailed": mapping_details.get(model, model)
                }
                
                fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")
                
                temp_file_name = os.path.join(temporary_folder_1, f"{new_item['id']}.jsonl")
                with open(temp_file_name, "w", encoding="utf-8") as temp_out:
                    temp_out.write(json.dumps(new_item, ensure_ascii=False) + "\n")
                    
                print(new_item)

    


if __name__ == "__main__":
    main()
