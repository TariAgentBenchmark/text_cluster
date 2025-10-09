import json
import os
import re
from typing import Dict, List

import dotenv
import pandas as pd
import requests
from tqdm import tqdm

dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_MODEL_ID = os.getenv("OPENAI_MODEL_ID")


discover_prompt = """
你是一个教育评价分析专家。请分析以下教案评价内容，挖掘出主要的观点类别。

教案评价内容：
{texts}

请根据这些评价内容，总结出主要的观点类别和具体观点。要求：
1. 识别出评价内容中反复出现的主题和观点
2. 将相似观点归类到同一类别下
3. 类别名称要简洁明确
4. 每个类别下列出具体的观点

输出格式为JSON：
{{
    "类别1": ["观点1", "观点2", ...],
    "类别2": ["观点3", "观点4", ...],
    ...
}}

请直接输出JSON格式的分类结果。
"""


summary_prompt = """
你是一个教育评价分析专家。现在有多次分析得到的观点类别体系，请将它们汇总整合成一个统一的类别体系。

各次分析结果：
{all_results}

请整合这些结果，要求：
1. 合并相同或相似的类别
2. 去除重复的观点
3. 保持类别体系的完整性和逻辑性
4. 每个类别下的观点要具体且有代表性

输出格式为JSON：
{{
    "类别1": ["观点1", "观点2", ...],
    "类别2": ["观点3", "观点4", ...],
    ...
}}

请直接输出最终整合后的类别体系JSON。
"""


def call_llm(prompt: str) -> str:
    """调用大模型API"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

    data = {
        "model": OPENAI_MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": "你是一个教育评价分析专家，擅长从文本中提取和归纳观点类别。",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 2000,
    }

    response = requests.post(
        f"{OPENAI_API_BASE}/chat/completions", headers=headers, json=data, timeout=60
    )

    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    else:
        raise Exception(f"API调用失败: {response.status_code}, {response.text}")


def extract_json(text: str) -> Dict:
    """从文本中提取JSON"""
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        json_str = json_match.group()
        return json.loads(json_str)
    else:
        raise ValueError("无法从响应中提取JSON格式")


def discover_categories(
    csv_path: str = "data/test.csv",
    sample_size: int = 100,
    num_iterations: int = 10,
    text_column: str = "ocr",
    output_path: str = "discovered_categories.json",
):
    """
    从CSV文件中挖掘观点类别体系
    
    参数:
        csv_path: CSV文件路径
        sample_size: 每次采样的数量
        num_iterations: 采样次数
        text_column: 文本内容所在的列名
        output_path: 输出文件路径
    """
    print(f"正在读取数据文件: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"数据文件共有 {len(df)} 条记录")
    print(f"将进行 {num_iterations} 次采样，每次采样 {sample_size} 条")
    
    # 确保采样总数不超过数据集大小
    total_samples_needed = sample_size * num_iterations
    if total_samples_needed > len(df):
        print(f"警告: 需要 {total_samples_needed} 条样本，但数据集只有 {len(df)} 条")
        print(f"将使用不放回采样")
    
    # 随机采样，避免重复
    sampled_indices = set()
    all_results = []
    
    for i in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"第 {i+1}/{num_iterations} 次采样和分析")
        print(f"{'='*60}")
        
        # 获取可用的索引（排除已采样的）
        available_indices = list(set(range(len(df))) - sampled_indices)
        
        if len(available_indices) < sample_size:
            print(f"警告: 可用样本不足 {sample_size} 条，只有 {len(available_indices)} 条")
            if len(available_indices) == 0:
                print("没有更多可用样本，停止采样")
                break
            current_sample_size = len(available_indices)
        else:
            current_sample_size = sample_size
        
        # 随机选择索引
        selected_indices = pd.Series(available_indices).sample(
            n=current_sample_size, random_state=i
        ).tolist()
        
        # 更新已采样的索引
        sampled_indices.update(selected_indices)
        
        # 获取样本数据
        sample_df = df.iloc[selected_indices]
        sample_texts = sample_df[text_column].dropna().tolist()
        
        print(f"采样了 {len(sample_texts)} 条有效文本")
        
        # 将样本文本组合成一个字符串（限制长度）
        combined_text = "\n\n---\n\n".join(sample_texts[:sample_size])
        
        # 如果文本太长，截断
        max_length = 20000  # 大约8000 tokens
        if len(combined_text) > max_length:
            combined_text = combined_text[:max_length] + "\n...(文本过长已截断)"
        
        print(f"组合文本长度: {len(combined_text)} 字符")
        
        # 调用大模型进行分析
        print("正在调用大模型进行类别挖掘...")
        prompt = discover_prompt.format(texts=combined_text)
        response = call_llm(prompt)
        
        # 解析结果
        try:
            categories = extract_json(response)
            print(f"成功提取类别: {list(categories.keys())}")
            all_results.append(categories)
        except Exception as e:
            print(f"解析结果失败: {e}")
            print(f"原始响应: {response}")
            continue
    
    print(f"\n{'='*60}")
    print("所有采样完成，开始汇总类别体系")
    print(f"{'='*60}")
    
    # 将所有结果格式化为字符串
    results_text = ""
    for i, result in enumerate(all_results):
        results_text += f"\n\n第{i+1}次分析结果：\n"
        results_text += json.dumps(result, ensure_ascii=False, indent=2)
    
    print(f"共收集到 {len(all_results)} 次有效分析结果")
    print("正在调用大模型进行最终汇总...")
    
    # 调用大模型进行汇总
    summary_prompt_filled = summary_prompt.format(all_results=results_text)
    final_response = call_llm(summary_prompt_filled)
    
    # 解析最终结果
    final_categories = extract_json(final_response)
    
    print(f"\n{'='*60}")
    print("最终类别体系:")
    print(f"{'='*60}")
    print(json.dumps(final_categories, ensure_ascii=False, indent=2))
    
    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_categories, f, ensure_ascii=False, indent=2)
    
    print(f"\n类别体系已保存到: {output_path}")
    
    # 同时保存中间结果
    intermediate_path = output_path.replace(".json", "_intermediate.json")
    with open(intermediate_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"中间结果已保存到: {intermediate_path}")
    
    return final_categories


if __name__ == "__main__":
    # 可以通过修改这些参数来调整采样策略
    discover_categories(
        csv_path="data/test.csv",
        sample_size=100,        # 每次采样数量
        num_iterations=10,      # 采样次数
        text_column="ocr",      # 文本列名
        output_path="discovered_categories.json"
    )

