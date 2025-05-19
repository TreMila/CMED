import argparse
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import json
from tqdm import tqdm


def call_soulchat(model, tokenizer, text):
    prompt = "用户：" + text + "\n心理咨询师："
    response, _ = model.chat(
        tokenizer, 
        query=prompt, 
        history=None, 
        max_length=2048, 
        num_beams=1, 
        do_sample=True, 
        top_p=0.75, 
        temperature=0.95, 
        logits_processor=None
    )
    return response


def call_mechat(model, tokenizer, text):
    prompt = f'''现在你扮演一位专业的心理咨询师，你具备丰富的心理学和心理健康知识。你擅长运用多种心理咨询技巧，例如认知行为疗法原则、动机访谈技巧和解决问题导向的短期疗法。以温暖亲切的语气，展现出共情和对来访者感受的深刻理解。以自然的方式与来访者进行对话，避免过长或过短的回应，确保回应流畅且类似人类的对话。提供深层次的指导和洞察，使用具体的心理概念和例子帮助来访者更深入地探索思想和感受。避免教导式的回应，更注重共情和尊重来访者的感受。根据来访者的反馈调整回应，确保回应贴合来访者的情境和需求。请为以下的对话生成一个回复。

对话：
来访者：{text}
咨询师：'''
    response, _ = model.chat(
        tokenizer, 
        prompt, 
        history=[], 
        temperature=0.8, 
        top_p=0.8
    )
    return response


def call_psychat(model, tokenizer, text):
    prompt = f'''现在你扮演一位专业的心理咨询师，你具备丰富的心理学和心理健康知识。你擅长运用多种心理咨询技巧，例如认知行为疗法原则、动机访谈技巧和解决问题导向的短期疗法。以温暖亲切的语气，展现出共情和对来访者感受的深刻理解。以自然的方式与来访者进行对话，避免过长或过短的回应，确保回应流畅且类似人类的对话。提供深层次的指导和洞察，使用具体的心理概念和例子帮助来访者更深入地探索思想和感受。避免教导式的回应，更注重共情和尊重来访者的感受。根据来访者的反馈调整回应，确保回应贴合来访者的情境和需求。请为以下的对话生成一个回复。

对话：
来访者：{text}
咨询师：'''
    response, _ = model.chat(
        tokenizer, 
        prompt, 
        history=[], 
        temperature=0.8, 
        top_p=0.8
    )
    return response 


def call_mindchat(model, tokenizer, text):
    response, _ = model.chat(
        tokenizer, 
        query = text, 
        history=None
    )
    return response


MODEL_CLASSES = {
    "soulchat": (AutoTokenizer, AutoModel, call_soulchat),
    "mechat": (AutoTokenizer, AutoModel, call_mechat),
    "psychat": (AutoTokenizer, AutoModel, call_psychat),
    "mindchat": (AutoTokenizer, AutoModelForCausalLM, call_mindchat)
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--data_file', default=None, type=str,)
    parser.add_argument('--output_file', default=None, type=str)

    args = parser.parse_args()
    
    
    # 加载模型
    tokenizer_class, model_class, call_func = MODEL_CLASSES[args.model_name]
    tokenizer = tokenizer_class.from_pretrained(args.base_model, trust_remote_code=True)
    model = model_class.from_pretrained(args.base_model, device_map="auto", trust_remote_code=True).half()
    model.generation_config = GenerationConfig.from_pretrained(args.base_model, trust_remote_code=True)

    model = model.eval()
    
    # 加载测试集
    with open(args.data_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    examples = [item['patient_text'] for item in data]

    print("Start inference.")
   
    results = []
    for example in tqdm(examples, desc='Generating response...'):
        pred_answer = call_func(model, tokenizer, example)
        results.append({
            "text": example,
            "pred_answer": pred_answer
        })
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
if __name__ == '__main__':
    main()