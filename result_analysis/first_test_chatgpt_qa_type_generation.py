import pandas as pd
from openai import OpenAI
import json
import time

def categorize_question(prompt, openKey):
    client = OpenAI( api_key=openKey)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    response = response.choices[0].message.content
    # Extract the Question Type from the response
    question_type = response.split("Question Type: ")[-1].strip()
    
    return question_type

def main():
    with open('/home/ngoc/githubs/chatgpt-hie-vqa/chatgpt_key/dunghuynh110496.txt', 'r') as file:
        for line in file:
            openKey = line
            
    
    df = pd.read_csv('/home/ngoc/githubs/chatgpt-hie-vqa/result_analysis/new_qa_pairs_for_evaluation.csv')
    
    sub_df = df[:30000]
    for idx, row in enumerate(sub_df.to_dict("records")):
        print(f'[{idx}/{len(sub_df)}]')
        
        while True:
            try:
                question_str = row["question"]
                answer_str = row["answer"]
                prompt = row["prompt"]
                chatgpt_question_type = categorize_question(prompt, openKey)

                # Create a dictionary with the current key-value pair
                data = {"question": question_str,
                        "answer": answer_str,
                        "question_answer_type":chatgpt_question_type}

                file_name = "new_4.json"
                with open(file_name, 'a') as json_file:
                    json.dump(data, json_file)
                    json_file.write('\n')

                break  # Exit the retry loop if successful
            except Exception as e:
                print(f'Error: {e}')
                print('Retrying...')
                time.sleep(2)

if __name__ == "__main__":
    main()