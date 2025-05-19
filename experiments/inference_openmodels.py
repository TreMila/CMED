import argparse
import json
import os
import gc
import torch
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)

def call_qwen(model, tokenizer, prompt, model_path, max_new_tokens, do_sample, temperature):
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature = temperature,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response  
    
    
def call_internlm(model, tokenizer, prompt, model_path, max_new_tokens, do_sample, temperature):
    response,_ = model.chat(
        tokenizer, 
        prompt, 
        history=[], 
        do_sample=do_sample, 
        temperature = temperature,
        max_new_tokens=max_new_tokens)
    
    return response
   

def call_yi(model, tokenizer, prompt, model_path, max_new_tokens, do_sample, temperature):
    messages = [
        {"role": "user", "content": prompt}
    ]
    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
    output_ids = model.generate(
        input_ids.to('cuda'), 
        eos_token_id=tokenizer.eos_token_id, 
        max_new_tokens=max_new_tokens,
        do_sample=do_sample, 
        temperature = temperature,
        )
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    return response
    

def call_deepseek(model, tokenizer, prompt, model_path, max_new_tokens, do_sample, temperature):
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    messages = [
        {"role": "user", "content": prompt}
    ]
    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    outputs = model.generate(
        input_tensor.to(model.device), 
        max_new_tokens=max_new_tokens, 
        do_sample=do_sample,  
        temperature = temperature,
        )
    response = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    return response
    

def prompt_emo_cls(emoiton_list, text):
    return f"请根据患者的陈述，推断患者目前的情绪状态，并确保该情绪属于下列情绪词汇列表之一：{emoiton_list}。直接输出情绪词。\n患者：{text}"


# 不用情绪预测结果
def prompt_stra_cls(strategy_list,text):
    return f"请根据提供的患者的话语，推测心理咨询师应当采用的回应策略，该策略需从以下策略词列表中选择：{strategy_list}。直接输出回应策略。\n患者：{text}"


# 用预测情绪预测结果
def prompt_stra_cls_2(strategy_list, text, emotion):
    return f"请根据提供的患者的言辞以及患者的情绪，推测心理咨询师应当采用的回应策略，该策略需从以下策略词列表中选择：{strategy_list}。直接输出回应策略。\n患者：{text}\n患者情绪：{emotion}"


# baseline
def prompt_base():
    return "你是一位专业的心理咨询师，具备丰富的心理学知识和咨询技巧，请你根据以下患者的话语，生成合适的回复，展现共情能力。\n"


# 1、with emotion only
def prompt_emotion_only(text, emotion):
    return f'''请你根据我提供的患者的话语，考虑患者当前的情绪，生成一句咨询师的回复。直接生成回复即可。\n患者：{text}\n患者情绪：{emotion}\n'''


# 2、with strategy only
def prompt_strategy_only(text, strategy):
    return f'''请你根据我提供的患者的话语，运用心理咨询师即将采用的回应策略，生成一句咨询师的回复。直接生成回复即可。\n患者：{text}\n心理咨询师回应策略：{strategy}'''


# 3、with both emotion and strategy
def prompt_both(text, emotion, strategy):
    return f'''请你根据我提供的患者的话语，考虑患者当前的情绪，运用心理咨询师即将采用的回应策略，生成一句咨询师的回复。直接生成回复即可。\n患者：{text}\n患者情绪：{emotion}\n心理咨询师回应策略：{strategy}'''


MODEL_CLASSES = {
    "qwen":(AutoModelForCausalLM, AutoTokenizer, call_qwen),
    "internlm": (AutoModelForCausalLM, AutoTokenizer, call_internlm),
    "yi": (AutoModelForCausalLM, AutoTokenizer, call_yi),
    "deepseek": (AutoModelForCausalLM, AutoTokenizer, call_deepseek),
}


emotion_list = [
    ['喜悦','愤怒','抑郁','焦虑','羞耻','悲伤','恐惧','孤独','厌恶','无'],
    {
        '喜悦': '患者感到快乐、满足，常见于咨询进展顺利时',
        '愤怒': '患者对问题表现出强烈不满',
        '抑郁': '患者情绪低落、无助',
        '焦虑': '患者感到紧张、不安',
        '羞耻': '患者因自身行为感到羞愧',
        '悲伤': '患者感到伤心和失落',
        '恐惧': '患者对潜在危险感到害怕',
        '孤独': '患者感到被疏离或孤立',
        '厌恶': '患者对某事件表现出强烈排斥或反感',
        '无': '患者当下情绪中性，无明显波动'
    }
]

strategy_list = [
    ['认可','提问','重述','挑战','解释','自我表露','提供信息','直接指导','其他'],
    {
        '认可': '提供情感支持、安慰、鼓励和强化',
        '提问': '获取患者特定/非特定的信息或数据',
        '重述': '用不同方式重复或澄清患者表达',
        '挑战': '指出患者未觉察的问题或不合理信念',
        '解释': '从更深层提供行为或情感的理解',
        '自我表露': '分享咨询师自身相关的个人信息',
        '提供信息': '传达事实、观点或资源',
        '直接指导': '给予建议或行动指导',
        '其他': '不属于上述任一策略的无关陈述'
    }
]

model_dict = {
    'qwen': {
        'v1': 'path_v1',
        'v2': 'path_v2',
        'v3': 'path_v3',
    },
    'internlm': {
        'v1': 'path_v1',
        'v2': 'path_v2',
        'v3': 'path_v3',
    },
    'yi': {
        'v1': 'path_v1',
        'v2': 'path_v2',
        'v3': 'path_v3',
    },
    'deepseek': {
        'v1': 'path_v1',
        'v2': 'path_v2',
        'v3': 'path_v3',
    },
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type_list', default='auto', type=str)
    parser.add_argument('--data_file', default=None, type=str, help="A file that contains instructions (one instruction per line)")
    parser.add_argument('--output_dir', default=None, type=str)
    parser.add_argument('--do_sample', action='store_true', help='Whether to use sampling')
    parser.add_argument('--model_version_list', default=None, type=str)
    parser.add_argument('--temperature', default=0.7, type=float)
    parser.add_argument('--skip_cls', action='store_true')
    parser.add_argument('--skip_gen', action='store_true')
    
    args = parser.parse_args()
    print(args)

    model_type_list = list(map(str, args.model_type_list.split(',')))
    model_version_list = list(map(str, args.model_version_list.split(',')))

    for model_type in tqdm(model_type_list):
        for model_version in tqdm(model_version_list):
            base_model = model_dict[model_type][model_version]
            print(base_model)
    
            # 加载模型
            model_class, tokenizer_class, call_func = MODEL_CLASSES[model_type]
            tokenizer = tokenizer_class.from_pretrained(base_model, trust_remote_code=True)
            tokenizer.padding_side = 'left'
            model = model_class.from_pretrained(
                base_model, 
                torch_dtype=torch.float16,
                device_map="auto", 
                trust_remote_code=True
                )
            model.eval()
            print("Start inference.")

            
            # # 分类
            with open(args.data_file, 'r') as f:
                data = [json.loads(line) for line in f]
            examples = [item['patient_text'] for item in data]

            # v1模型，表示训练时，策略分类和情绪分类任务独立，且不包含标签解释
            # v2模型，表示训练时，策略分类依赖情绪分类结果，且不包含标签解释
            # v3模型，表示训练时，策略分类和情绪分类任务独立，且包含标签解释
            
            # 跳过分类任务，使用已有的分类结果
            if not args.skip_cls:
                args.do_sample = False
                temperature = 0.0
                output_file_cls = os.path.join(args.output_dir, f"result_finetunemodel_{model_type}_{model_version}_cls.jsonl")
                
                os.makedirs(os.path.dirname(output_file_cls), exist_ok=True)
                if os.path.exists(output_file_cls):
                    os.remove(output_file_cls)
                
                for example in tqdm(examples, desc="Step1: Predicting..."):
                    # Step1: 情绪分类
                    # 当使用v3模型，情绪分类时需要标签解释
                    if model_version != 'v3':
                        pred_emotion = call_func(model, tokenizer, prompt_emo_cls(emotion_list[0], example), base_model, 10, args.do_sample, temperature)
                    else:
                        pred_emotion = call_func(model, tokenizer, prompt_emo_cls(emotion_list[1], example), base_model, 10, args.do_sample, temperature)
                    
                    # Step2: 策略分类
                    # 当使用v1模型，在推理策略分类时不使用情绪预测结果，且不包含标签解释
                    if model_version == 'v1':
                        pred_strategy = call_func(model, tokenizer, prompt_stra_cls(strategy_list[0], example), base_model, 10, args.do_sample, temperature)
                    
                    # 当使用v2模型，在推理策略分类时使用情绪预测结果，且不包含标签解释
                    elif model_version == 'v2':
                        pred_strategy = call_func(model, tokenizer, prompt_stra_cls_2(strategy_list[0], example, pred_emotion), base_model, 10, args.do_sample, temperature)
                    
                    elif model_version == 'v3':
                        pred_strategy = call_func(model, tokenizer, prompt_stra_cls_2(strategy_list[1], example, pred_emotion), base_model, 10, args.do_sample, temperature)
                    
                    else:
                        raise ValueError(f"Invalid model version: {model_version}")
                    
                    # 记录分类任务结果
                    result = {
                        "pred_emotion": pred_emotion,
                        "pred_strategy": pred_strategy,
                        "pred_answer_1": "",
                        "pred_answer_2": "",
                        "pred_answer_3": "",
                        "text": example,
                    }
                    with open(output_file_cls, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')

            if not args.skip_gen:
                args.do_sample = True
                temperature = args.temperature
                # Step3: 回复生成
                data_file = os.path.join(args.output_dir, f"result_finetunemodel_{model_type}_{model_version}_cls.jsonl")

                with open(data_file, 'r') as f:
                    data = [json.loads(line) for line in f]
                
                output_file = os.path.join(args.output_dir, f"result_finetunemodel_{model_type}_{model_version}_{temperature}_gen.jsonl")
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                if os.path.exists(output_file):
                    os.remove(output_file)
                    
                with open(output_file, 'a', encoding='utf-8') as f:
                    for item in tqdm(data, desc="Step2: Predicting..."):
                        example = item['text']
                        pred_emotion = item['pred_emotion']
                        pred_strategy = item['pred_strategy']
                        pred_answer_1 = call_func(model, tokenizer, prompt_emotion_only(example, pred_emotion), base_model, 300, args.do_sample, temperature)
                        pred_answer_2 = call_func(model, tokenizer, prompt_strategy_only(example, pred_strategy), base_model, 300, args.do_sample, temperature)
                        pred_answer_3 = call_func(model, tokenizer, prompt_both(example, pred_emotion, pred_strategy), base_model, 300, args.do_sample, temperature)
                        
                        result = {
                            "pred_emotion": pred_emotion,
                            "pred_strategy": pred_strategy,
                            "pred_answer_1": pred_answer_1,
                            "pred_answer_2": pred_answer_2,
                            "pred_answer_3": pred_answer_3,
                            "text": example,
                        }
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            del model
            del tokenizer
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == '__main__':
    main()