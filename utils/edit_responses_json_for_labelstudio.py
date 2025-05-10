import os
import json
import numpy as np

# json_responses = [ 'responses_from_claude3-5-sonnet.json',
#                     'responses_from_claude3-7-sonnet.json',
#                     'responses_from_gpt-4o.json',
#                     'responses_from_gpt-o3-mini.json',
#                     ]
# for json_response in json_responses:
#     with open(f'./llm_responses/{json_response}', "r") as f:
#         data = json.load(f)

#     for d in data:
#         recommendations = d["Recommendations"]
#         d["Recommendations"] = "\n".join(f"{i+1}. {rec}" for i, rec in enumerate(recommendations))

#     with open(f'./llm_responses_labelstudio_format/{json_response}', "w") as f:
#         json.dump(data, f, indent=4)



# Create 4 json files with 10, 10, 15, 15 questions and responses from all json files for those questions

questions = []
question_json = '../benchmark_questions/AFM_LLM_Combined_50_Questions.json'
with open(question_json, "r") as f:
    data = json.load(f)


models = [  'gpt-4o',
            'gpt-o3-mini',
            'claude3-5-sonnet',
            'claude3-7-sonnet',
            ]

model_responses = {}
for model in models:
    model_responses[model] = []
    with open(f'../llm_responses_labelstudio_format_old/responses_from_{model}.json', "r") as f:
        data = json.load(f)
    for d in data:
        model_responses[model].append(d)

print(len(model_responses['gpt-4o']), len(model_responses['gpt-o3-mini']), len(model_responses['claude3-5-sonnet']), len(model_responses['claude3-7-sonnet']))

# for each question, find the response with the same idx and add it to the new json file
# for each question create a dict with the the sub-dicts of the responses from all models
# [{ "idx": 0, "gpt-4o":{answer:"", recommendations:""}, "gpt-o3-mini":{answer:"", recommendations:""}, "claude3-5-sonnet":{answer:"", recommendations:""}, "claude3-7-sonnet":{answer:"", recommendations:""}},
#  { "idx": 1, "gpt-4o":{answer:"", recommendations:""}, "gpt-o3-mini":{answer:"", recommendations:""}, "claude3-5-sonnet":{answer:"", recommendations:""}, "claude3-7-sonnet":{answer:"", recommendations:""}},
#  ...]
question_sets = {'set1': [int(x) for x in np.arange(0, 10)],
                'set2': [int(x) for x in np.arange(10, 20)],
                'set3': [int(x) for x in np.arange(20, 35)],
                'set4': [int(x) for x in np.arange(35, 50)],
                }

# for xml labelstudio format
new_model_keys = {'gpt-4o':'gpt_4o', 'gpt-o3-mini':'gpt_o3_mini', 'claude3-5-sonnet':'claude_3_5_sonnet', 'claude3-7-sonnet':'claude_3_7_sonnet'}

for question_set in question_sets:
    responses_to_question_set = []
    for idx in question_sets[question_set]:
        md = {}
        md['idx'] = idx
        md['Question'] = data[idx]['Question']
        for model in model_responses:
            md[new_model_keys[model]+'_Answers'] = model_responses[model][idx]['Answer']
            md[new_model_keys[model]+'_Recommendations'] = model_responses[model][idx]['Recommendations']
        responses_to_question_set.append(md)

    with open(f'../llm_responses_labelstudio_format/responses_to_{question_set}.json', "w") as f:
        json.dump(responses_to_question_set, f, indent=4)