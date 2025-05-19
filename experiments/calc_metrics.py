import jieba
import json
import pandas as pd 
from tqdm import tqdm
from nltk.translate.meteor_score import meteor_score
from bert_score import score
from rouge import Rouge


def rouge(reference, candidate):
    rouge = Rouge()
    reference_tokenized = ' '.join(jieba.cut(reference))
    candidate_tokenized = ' '.join(jieba.cut(candidate))
    score = rouge.get_scores(candidate_tokenized, reference_tokenized, avg=True)
    result = {key: value['f'] for key, value in score.items()}
    return result


def meteor(reference, candidate):
    reference_tokenized = ' '.join(jieba.cut(reference)).split()
    candidate_tokenized = ' '.join(jieba.cut(candidate)).split()
    
    score = round(meteor_score([reference_tokenized], candidate_tokenized), 4)
    return {'meteor': score}



def machine_metric(generate_refs, generate_preds):
    results = []
    avg_res = {} 
    try:
        for idx,(ref, pred) in enumerate(zip(generate_refs, generate_preds)):
            if len(pred) == 0:
                continue
            rouge_score = rouge(ref, pred)
            meteor_score = meteor(ref, pred)
            results.append({**rouge_score, **meteor_score})
    except Exception as e:
        print(idx, e)
        if "maximum recursion depth exceeded in comparison" in str(e):
            results.append({"rouge-1":0.0, "rouge-2":0.0, "rouge-l":0.0, "meteor":0.0})

    for key in results[0].keys():
        avg_res[key] = round(sum([item[key] for item in results]) / len(results) * 100, 2)
    
    _, _, b_score = score(
        generate_preds,generate_refs,
        model_type='chinese-xlnet-base',
        num_layers=5,
        batch_size=8,
        lang="zh",
        return_hash=False,  
        idf=False 
    )
    avg_res.update({'bert_score': round(b_score.mean().item()* 100, 2)})

    return avg_res


def main():
    test_file = 'path/to/test_examples.jsonl'
    pred_dir = ['path/to/pred_file1.josnl', 'path/to/pred_file2.jsonl', '...']

    with open(test_file, 'r') as f:
        data = [json.loads(line) for line in f]
    generate_refs = [item['doctor_text'] for item in data]

    metrics = []
    for pred_file in tqdm(pred_dir, total=len(pred_dir), desc='Calculating metrics...'):
        with open(pred_file, 'r', encoding='utf-8') as f:
            results = [json.loads(line) for line in f]
            
        generate_preds = [item['pred_answer'] for item in results]
        metric = machine_metric(generate_refs, generate_preds)
        metrics.append(metric)

    # 获取模型名称
    index_names = [file.split('/')[-1].split('generate_')[-1].split('.jsonl')[0] for file in pred_dir]
    # 保存结果
    df = pd.DataFrame(metrics, index=index_names)
    df.to_excel('path/to/save_result.xlsx')
