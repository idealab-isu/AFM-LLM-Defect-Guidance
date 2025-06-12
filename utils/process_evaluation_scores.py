import os
import json

json_path_1 = '../evaluations_by_AFM_experts/Question_Set_1_Expert_1.json'
json_path_2 = '../evaluations_by_AFM_experts/Question_Set_2_Expert_2.json'
json_path_3 = '../evaluations_by_AFM_experts/Question_Set_3_Expert_3.json'
json_path_4 = '../evaluations_by_AFM_experts/Question_Set_4_Expert_4.json'


with open(json_path_1, 'r', encoding='utf-8') as f:
	evaluations_1 = json.load(f)

with open(json_path_2, 'r', encoding='utf-8') as f:
	evaluations_2 = json.load(f)

with open(json_path_3, 'r', encoding='utf-8') as f:
	evaluations_3 = json.load(f)

with open(json_path_4, 'r', encoding='utf-8') as f:
	evaluations_4 = json.load(f)

# combine all the evaluations
evaluations = evaluations_1 + evaluations_2 + evaluations_3 + evaluations_4

print(len(evaluations))
print(evaluations[0].keys())

results = []
models = {'gpt4o', 'gpto3', 'claude35', 'claude37'}
metrics = ['relevance', 'completeness', 'correctness', 'clarity', 'overall', 'comments']

for idx, eval in enumerate(evaluations):
	assert idx == eval['idx'], f"Error: idx: {idx} != eval['idx']: {eval['idx']}"

	result = {}
	result['question_id'] = idx
	result['question'] = eval['Question']
	# first 10 responses are from expert 1, next 10 are from expert 2, next 15 are from expert 3, next 15 are from expert 4
	result['expert_id'] = 1 if idx < 10 else 2 if idx < 20 else 3 if idx < 35 else 4 
	
	for model in models:
		result[model] = {}
		for metric in metrics:
			if metric == 'comments':
				if f'{model}_{metric}' not in eval:
					result[model][metric] = 'No comments'
				else:
					result[model][metric] = eval[f'{model}_{metric}']
			else:
				result[model][metric] = int(eval[f'{model}_{metric}'])

	results.append(result)

print(len(results))
# save the results to a json file
with open('../evaluations_by_AFM_experts/results.json', 'w', encoding='utf-8') as f:
	json.dump(results, f, ensure_ascii=False, indent=4)
