import argparse
import json
from tqdm import tqdm
from openai import OpenAI
from openai import RateLimitError, APITimeoutError
import time
import os
import requests


ERROR_MESSAGE = "模型调用出错，请稍后再试。"
emoiton_list = ['喜悦','愤怒','抑郁','焦虑','羞耻','悲伤','恐惧','孤独','厌恶','其他']
strategy_list = ['认可','提问','重述','挑战','解释','自我表露','提供信息','直接指导','其他']

def get_sys_prompt():
    return(f'''当前你是一个专业的心理咨询师。你的任务是根据患者说的话，给出回复。''')


def get_user_prompt(patiect_text):
    return(f'''请你在回答时请按照以下步骤进行思考：
1. 首先根据患者的话语，判断患者当前的情绪，情绪分为：{emoiton_list}，你只能选择一种情绪。
2. 其次根据患者的话语和情绪，选择合适的回应策略，回应策略分为：{strategy_list}，你只能选择一种策略。
3. 最后根据患者的话语、患者的情绪和你选择的回应策略，给出回复。
你需要按照以下格式进行输出：
```
患者情绪: '你判断的情绪'
回应策略: '你选择的回应策略'
咨询师回复: '你最终的回复'
```

患者: {patiect_text}''')


def call_ernie(text, api_key, api_base, model, timeout=None, retry=None):
    url = "https://qianfan.baidubce.com/v2/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer "+ api_key
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": get_sys_prompt()+'\n'+get_user_prompt(text)
            }
        ],
        "max_output_tokens": 300,
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']


def call_gpt(text, api_key, api_base, model, timeout=60, retry=3):
    client = OpenAI(api_key=api_key,base_url=api_base)
    attempt = 0
    while attempt < retry:
        try:
            completion = client.chat.completions.create(
                model=model,
                top_p=0.6,
                messages=[
                    {"role": "system", "content": get_sys_prompt()},
                    {"role": "user", "content": get_user_prompt(text)},
                ],
                timeout=timeout
            )
            return completion.choices[0].message.content
        
        except RateLimitError as e:
            print(f"等待30s...模型过载或达到速率限制，错误信息: {e}")
            attempt += 1
            time.sleep(30)
        except APITimeoutError as e:
            print(f"模型超时，错误信息: {e}")
            attempt += 1
        except Exception as e:
            print(f"调用API出现其他错误: {e}")
            attempt += 1
    
    return ERROR_MESSAGE


def call_o1(text, api_key, api_base, model, timeout=60, retry=3):
    client = OpenAI(api_key=api_key,base_url=api_base)
    attempt = 0
    while attempt < retry:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": get_sys_prompt()+'\n'+get_user_prompt(text)},
                ],
                timeout=timeout
            )
            return completion.choices[0].message.content
        
        except RateLimitError as e:
            print(f"等待30s...模型过载或达到速率限制，错误信息: {e}")
            time.sleep(30)
            attempt += 1
        except APITimeoutError as e:
            print(f"模型超时，错误信息: {e}")
            time.sleep(30)
            attempt += 1
        except Exception as e:
            print(f"调用API出现其他错误: {e}")
            time.sleep(30)
            attempt += 1
    
    return ERROR_MESSAGE


def call_yi(text, api_key, api_base, model, timeout=60, retry=3):
    client = OpenAI(api_key=api_key,base_url=api_base)
    attempt = 0
    while attempt < retry:
        try:
            completion = client.chat.completions.create(
                model=model,
                top_p=0.6,
                messages=[
                    {"role": "system", "content": get_sys_prompt()},
                    {"role": "user", "content": get_user_prompt(text)},
                ],
                timeout=timeout
            )
            return completion.choices[0].message.content
        
        except RateLimitError as e:
            print(f"等待30s...模型过载或达到速率限制，错误信息: {e}")
            attempt += 1
            time.sleep(30)
        except APITimeoutError as e:
            print(f"模型超时，错误信息: {e}")
            attempt += 1
        except Exception as e:
            print(f"调用API出现其他错误: {e}")
            attempt += 1
    
    return ERROR_MESSAGE


MODEL_CLASS = {
    "gpt" : {
        "api_base": "API_BASE",
        "api_key": "API_KEY",
        "call_func": call_gpt,
    },
    "o1": {
        "api_base": "API_BASE",
        "api_key": "API_KEY",
        "call_func": call_o1,
    },
    "qwen": {
        "api_base": "API_BASE",
        "api_key": "API_KEY",
        "call_func": call_gpt,
    },
    "ernie":{
        "api_base": "API_BASE",
        "api_key": "API_KEY",
        "call_func": call_ernie,
    },
    "yi": {
        "api_base": "API_BASE",
        "api_key": "API_KEY",
        "call_func": call_yi,
    },
    "deepseek": {
        "api_base": "API_BASE",
        "api_key": "API_KEY",
        "call_func": call_gpt,
    }
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default=None, type=str)
    parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--data_file', default=None, type=str)
    parser.add_argument('--output_dir', default=None, type=str)
    args = parser.parse_args()

    api_base = MODEL_CLASS[args.model_type]['api_base']
    api_key = MODEL_CLASS[args.model_type]['api_key']
    call_func = MODEL_CLASS[args.model_type]['call_func']
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 加载测试集
    with open(args.data_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    examples = [item['patient_text'] for item in data]
    output_file = os.path.join(args.output_dir, f'result_{args.model_name}.jsonl')

    
    print("Start inference.")

    
    with open(output_file, 'a', encoding='utf-8') as f:
        for example in tqdm(examples, desc='Generating response...'):
            pred_answer = call_func(example, api_key, api_base, args.model_name)
            assert pred_answer != ERROR_MESSAGE, ERROR_MESSAGE
            item = {
                "text": example,
                "pred_answer": pred_answer
            }
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

            
if __name__ == '__main__':
    main()