import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)


emotion_list = {
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


strategy_list = {
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


def prompt_emo_cls(text):
    return f"请根据患者的陈述，推断患者目前的情绪状态，并确保该情绪属于下列情绪词汇列表之一：{emotion_list}。直接输出情绪词。\n患者：{text}"


def prompt_stra_cls(text, emotion):
    return f"请根据提供的患者的言辞以及患者的情绪，推测心理咨询师应当采用的回应策略，该策略需从以下策略词列表中选择：{strategy_list}。直接输出回应策略。\n患者：{text}\n患者情绪：{emotion}"


def prompt_rsp(text, emotion, strategy):
    return f'''请你根据我提供的患者的话语，考虑患者当前的情绪，运用心理咨询师即将采用的回应策略，生成一句咨询师的回复。直接生成回复即可。\n患者：{text}\n患者情绪：{emotion}\n心理咨询师回应策略：{strategy}'''


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
    

MODEL_CLASSES = {
    "qwen":(AutoModelForCausalLM, AutoTokenizer, call_qwen),
    "internlm": (AutoModelForCausalLM, AutoTokenizer, call_internlm),
    "yi": (AutoModelForCausalLM, AutoTokenizer, call_yi),
    "deepseek": (AutoModelForCausalLM, AutoTokenizer, call_deepseek),
}


def main():
    model_type = "deepseek"
    model_path = "path/to/model"
    query = "your_query"
    
    do_sample = True
    temperature = 0.5
    
    # 加载模型
    model_class, tokenizer_class, call_func = MODEL_CLASSES[model_type]
    tokenizer = tokenizer_class.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    model = model_class.from_pretrained(
        model_path, 
        torch_dtype=torch.float16,
        device_map="auto", 
        trust_remote_code=True
        )

    model.eval()
    print("Start inference.")

    pred_emotion = call_func(model, tokenizer, prompt_emo_cls(query), model_path, 10, do_sample, temperature)
    pred_strategy = call_func(model, tokenizer, prompt_stra_cls(query, pred_emotion), model_path, 10, do_sample, temperature)
    pred_response = call_func(model, tokenizer, prompt_rsp(query, pred_emotion, pred_strategy), model_path, 300, do_sample, temperature)
    
    print(f"Emotion: {pred_emotion}")
    print(f"Strategy: {pred_strategy}")
    print(f"Response: {pred_response}")
        

if __name__ == '__main__':
    main()