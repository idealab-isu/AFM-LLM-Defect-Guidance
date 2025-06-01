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
	if idx == 6:
		continue # Expert 1's response is not complete for model 3 and 4 for question 7
	if idx == 22 or idx == 27 or idx == 29 or idx == 34:
		continue # Expert 3's response is not complete for gpto3_clarity question 3 # Expert 3's response is not complete for gpto3_relevance question 8 
        # Expert 3's response is not complete for claude35_completeness question 10 # Expert 3's response is not complete for claude35_overall question 15
	# if idx == 35 or idx == 38 or idx == 42:
	# 	continue # Expert 4's response is not complete for gpto3_completeness for question 1 # Expert 4's response is not complete for claude37_relevance for question 4
	#     # Expert 4's response is not complete for claude37_completeness for question 8
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

# for eval in evaluations:
#     relevance_scores.append(eval['annotations'][0]['result'][0]['value']['choices'][0].strip().split()[0])
#     completeness_scores.append(eval['annotations'][1]['result'][0]['value']['choices'][0].strip().split()[0])
#     correctness_scores.append(eval['annotations'][2]['result'][0]['value']['choices'][0].strip().split()[0])
#     clarity_scores.append(eval['annotations'][3]['result'][0]['value']['choices'][0].strip().split()[0])
#     overall_quality_scores.append(eval['annotations'][4]['result'][0]['value']['choices'][0].strip().split()[0])


# {'question_id': 1, 'question': '...', 'expert_id': 1, 'gpt-4o':{'relevance_score': 3, 'completeness_score': 3, 'correctness_score': 3, 'clarity_score': 3, 'overall_quality_score': 3, 'comments': '...'},
#  'gpt-o3-mini':{'relevance_score': 3, 'completeness_score': 3, 'correctness_score': 3, 'clarity_score': 3, 'overall_quality_score': 3, 'comments': '...'},
#  'claude3-5-sonnet':{'relevance_score': 3, 'completeness_score': 3, 'correctness_score': 3, 'clarity_score': 3, 'overall_quality_score': 3, 'comments': '...'},
#  'claude3-7-sonnet':{'relevance_score': 3, 'completeness_score': 3, 'correctness_score': 3, 'clarity_score': 3, 'overall_quality_score': 3, 'comments': '...'}}

