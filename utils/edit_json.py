import json

# read the json file and remove "id" from each question
with open('./benchmark_questions/AFM_LLM_Combined_50_Questions.json', 'r') as f:
    data = json.load(f)

for question in data:
    question.pop('id')

# save the json file with proper formatting
with open('./benchmark_questions/AFM_LLM_Combined_50_Questions_no_id.json', 'w') as f:
    json.dump(data, f, indent=4)
