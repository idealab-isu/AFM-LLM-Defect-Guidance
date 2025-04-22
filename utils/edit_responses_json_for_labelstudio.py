import os
import json

json_responses = { 'responses_from_claude3-5-sonnet.json',
                    'responses_from_claude3-7-sonnet.json',
                    'responses_from_gpt-4o.json',
                    'responses_from_gpt-o3-mini.json',
                    }
for json_response in json_responses:
    with open(f'./llm_responses/{json_response}', "r") as f:
        data = json.load(f)

    for d in data:
        recommendations = d["Recommendations"]
        d["Recommendations"] = "\n".join(f"{i+1}. {rec}" for i, rec in enumerate(recommendations))

    with open(f'./llm_responses_labelstudio_format/{json_response}', "w") as f:
        json.dump(data, f, indent=4)

