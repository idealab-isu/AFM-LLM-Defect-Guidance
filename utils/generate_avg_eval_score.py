import os
import json

# json_path = '../evaluations_by_AFM_experts/AFM-Image-Defect-Claude-Responses-Nabila.json'
# json_path = '../evaluations_by_AFM_experts/AFM-Image-Defect-GPTs-Responses-Hasib.json'
json_path = '../evaluations_by_AFM_experts/AFM-Image-Defect-GPTs-Responses-DrAnwesha.json'


with open(json_path, 'r') as f:
    evaluations = json.load(f)

reasoning_scores = []
non_reasoning_scores = []

for idx,eval in enumerate(evaluations):
    score = int(eval['annotations'][0]['result'][0]['value']['choices'][0].strip().split()[0])
    if idx < 50: # first 50 responses are from non-reasoning model
        non_reasoning_scores.append(score)
    else: # last 50 responses are from reasoning model
        reasoning_scores.append(score)

print(f'Average score for non-reasoning model: {sum(non_reasoning_scores) / len(non_reasoning_scores)}')
print(f'Average score for reasoning model: {sum(reasoning_scores) / len(reasoning_scores)}')

count1, count2, count3 = 0, 0, 0
count2_question_ids = []
for i in range(len(reasoning_scores)):
    if reasoning_scores[i] < non_reasoning_scores[i]:
        count1 += 1
    elif reasoning_scores[i] > non_reasoning_scores[i]:
        count2 += 1
        count2_question_ids.append(i)
    else:
        count3 += 1

print(f'Count of questions where non-reasoning score is greater than reasoning score (count1): {count1}')
print(f'Count of questions where reasoning score is greater than non-reasoning score (count2): {count2}')
print(f'Count of questions where reasoning score is equal to non-reasoning score (count3): {count3}')

# read question json file
with open('../benchmark_questions/AFM_LLM_Combined_50_Questions.json', 'r') as f:
    questions = json.load(f)

for i in count2_question_ids:
      print('#'*25)
      print(f'Question {i}: {questions[i]["question"]}')
      print(f'Reasoning score: {reasoning_scores[i]}')
      print(f'Non-reasoning score: {non_reasoning_scores[i]}')
      