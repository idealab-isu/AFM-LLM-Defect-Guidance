import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_and_process_results(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    print(len(results))
    # Initialize dictionaries to store scores
    model_scores = {
        'gpt4o': {'relevance': [], 'completeness': [], 'correctness': [], 'clarity': [], 'overall': []},
        'gpto3': {'relevance': [], 'completeness': [], 'correctness': [], 'clarity': [], 'overall': []},
        'claude35': {'relevance': [], 'completeness': [], 'correctness': [], 'clarity': [], 'overall': []},
        'claude37': {'relevance': [], 'completeness': [], 'correctness': [], 'clarity': [], 'overall': []}
    }
    incorrect_results = {'gpt4o':0, 'gpto3':0, 'claude35':0, 'claude37':0}
    # Collect scores for each model and metric
    for result in results:
        for model in model_scores:
            for metric in ['relevance', 'completeness', 'correctness', 'clarity', 'overall']:
                score = float(result[model][metric])  # Assuming metrics are stored in lowercase
                model_scores[model][metric].append(score)
                if metric == 'correctness':
                    if float(result[model][metric]) == -1:
                        incorrect_results[model] += 1
    # Calculate average scores
    models_keys = list(model_scores.keys())
    models = {'gpt4o':'GPT-4o', 'gpto3':'o3-mini', 'claude35':'Claude 3.5', 'claude37':'Claude 3.7'}
    metrics = ['Relevance', 'Completeness', 'Correctness', 'Clarity', 'Overall']
    scores = {}
    
    for model in models_keys:
        print('-'*50)
        print(model)
        model_avg_scores = {metric: 0 for metric in metrics}
        for metric in metrics:
            avg_score = np.mean(model_scores[model][metric.lower()])
            print(f"{metric}: {round(avg_score, 2)}")
            model_avg_scores[metric] = round(avg_score, 2)
        scores[model] = model_avg_scores
    
    return models.values(), metrics, scores, incorrect_results

# plot separate radar chart for each metric
def plot_radar_chart_for_metric(models, metric, metrics_scores, output_path="llm_afm_evaluation_radar_matplotlib.png"):
    # Number of variables
    num_models = len(models)
    angles = np.linspace(0, 2 * np.pi, num_models, endpoint=False).tolist()
    angles += angles[:1]  # Loop
    model_keys = list(metrics_scores[metric[0]].keys())
    for i, metric in enumerate(metrics):
        # Set up radar chart
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(polar=True))
        metric_scores = metrics_scores[metric]
        values = list(metric_scores.values()) + list(metric_scores.values())[:1]  # Complete loop
        ax.plot(angles, values, label=metric, linewidth=2)
        ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(model_keys, fontsize=10)
        ax.tick_params(axis='both',which='major', pad=10)

        ax.set_ylim(0, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=10)

        ax.set_title(metric, fontsize=10, pad=10)
        ax.legend(loc='lower center', fontsize=10, bbox_to_anchor=(0.5, -0.15), ncol=len(models))

        # Save high-res figure
        plt.tight_layout()
        plt.savefig(output_path.replace('.png', f'_{metric}.png'), dpi=300, bbox_inches='tight')
        print(f"Radar chart saved to {output_path.replace('.png', f'_{metric}.png')}")
        plt.close()
        

def plot_radar_chart(models, metrics, scores, output_path="llm_afm_evaluation_radar_matplotlib.png"):
    '''
    scores is a dictionary of model names and dictionaries of their scores for each metric
    '''
    # Number of variables
    num_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Loop

    # Set up radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    # Plot each model
    for i, model in enumerate(scores.keys()):
        model_scores = scores[model]
        values = list(model_scores.values()) + list(model_scores.values())[:1]  # Complete loop
        ax.plot(angles, values, label=model, linewidth=2)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=15)
    ax.tick_params(axis='both',which='major', pad=30)
    # # Configure angles and label appearance
    # ax.set_theta_offset(np.pi / 2)
    # ax.set_theta_direction(-1)
    # ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=10)

    # # Rotate labels to avoid overlap
    # for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
    #     label.set_rotation(np.degrees(angle) - 90)
    #     label.set_horizontalalignment('center')

    # Set radial limits
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=15)

    # Add title and legend
    # ax.set_title("LLM Evaluation by AFM Experts", fontsize=16, pad=20)
    ax.legend(loc='lower center', fontsize=15, bbox_to_anchor=(0.5, -0.15), ncol=len(models))

    # Save high-res figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Radar chart saved to {output_path}")
    plt.close()


# Load and process the data
json_path = '../evaluations_by_AFM_experts/results.json'  # Update this path
models, metrics, scores, incorrect_results = load_and_process_results(json_path)
print(len(models), len(metrics), len(scores))
print(scores)
print(models)
print(metrics)
print('-'*50)
print(incorrect_results)
# scores is a dictionary of model names and dictionaries of their scores for each metric
# create a dictionary of metrics and their scores for each model
metrics_scores = {metric: {model: scores[model][metric] for model in scores.keys()} for metric in metrics}
print(metrics_scores)

plot_radar_chart(models, metrics, scores, 'llm_afm_evaluation_radar_matplotlib.png')
plot_radar_chart_for_metric(models, metrics, metrics_scores, 'llm_afm_evaluation_radar_matplotlib_for_each_metric.png')

