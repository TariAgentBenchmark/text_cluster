import json
import os
import re
import time
from typing import Dict, List, Tuple

import dotenv
import pandas as pd
import requests
from joblib import Parallel, delayed
from tqdm import tqdm

dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_MODEL_ID = os.getenv("OPENAI_MODEL_ID")


seed_prompt = """
1. 教师行为问题类

备课不充分：没有充分考虑学生个体差异，缺乏突发情况预案
缺乏教育机智：面对课堂失控时未能及时制止和引导
课堂管理失当：没有维护课堂纪律，放任学生起哄
教学方法不当：用学生体重做对比，选择了不合适的教学材料

2. 学生权益保护类

自尊心受损：壮壮同学因体重被嘲笑，心理受到伤害
隐私权侵犯：体重属于个人隐私，不应公开讨论
缺乏尊重：教师和同学都没有尊重壮壮的感受
个体差异忽视：没有考虑学生间的身体差异

3. 教育理念违背类

新课改理念偏离：从"关注学科"到"关注人"未能体现
学生观问题：违背了"学生是独特的人""学生是发展的人"等理念
教师观缺失：未能体现尊重、赞赏学生的要求
素质教育缺位：没有面向全体学生，促进全面发展

4. 课堂教学问题类

注意力转移：学生兴趣从学习内容转向同学外貌
教学目标偏离：原本学习"千克认识"变成讨论学生体重
师生关系失衡：教师失去课堂主导地位
教学效果不佳：没有达到预期的教学目标

5. 备课改进建议类
关注学生的多个方面：

心理健康：了解学生心理承受能力和敏感点
个体差异：因材施教，尊重每个学生的独特性
身心发展：符合学生年龄特点和发展规律
学习能力：考虑学生的接受程度和理解能力
家庭环境：了解学生的家庭背景和成长环境

6. 教学设计优化：

预案准备：提前考虑可能出现的突发情况
材料选择：避免使用可能伤害学生的教学素材
方法改进：采用更合适的教学方式和举例
目标明确：确保教学活动服务于学习目标

7. 德育教育缺失类

品德教育不足：学生缺乏尊重他人的意识
价值观引导缺位：没有及时纠正学生的不当言行
集体意识薄弱：班级没有形成良好的氛围和规范

8. 其他
...

输出格式及要求：
{{
    "类别1": [观点1...],
    "类别2": [观点2...],
    ...
}}
> 只抽取教案评价中包含的观点，不要抽取教案设计中没有的观点
> 如果教案评价有其他观点，请归类到其他类别
> 宁缺毋滥，不要抽取教案设计中没有的观点

<example>
输入：
1)从同学们的话语中看到了对壮壮胖的嘲笑,对壮壮的自尊心有很大的打打击★壮壮与同学们争论起来而使场面积控.★在张老师方面从选最减的打到选最胖的壮壮从而形成了对照组★内容星★小时候并没有意识★的制止★(2)从学生的兴趣方面,要有足够吸引学生眼球的内容让学生注意力集中★行为举止,向学生感兴趣的方向引导★

输出：
{{
  "教师行为问题类": ["缺乏教育机智"],
  "学生权益保护类": ["自尊心受损"],
  "课堂教学问题类": ["注意力转移", "课堂管理失当"]
}}
</example>


请你判断下面这段教案设计评价中包含上述哪些观点，以json格式返回，json格式如下：
教案评价内容：
{text}

{{
    "类别1": [观点1...],
    "类别2": [观点2...],
    ...
}}
> 只抽取教案评价中包含的观点，不要抽取教案设计中没有的观点
> 如果教案评价有其他观点，请归类到其他类别
> 宁缺毋滥，不要抽取教案设计中没有的观点
"""


def classify_text(text: str) -> Dict[str, List[str]]:
    """使用OpenAI API对文本进行分类"""
    prompt = seed_prompt.format(text=text)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

    data = {
        "model": OPENAI_MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": "你是一个教育评价分析专家，请根据给定的观点分类体系，分析教案评价内容。",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 1000,
    }

    response = requests.post(
        f"{OPENAI_API_BASE}/chat/completions", headers=headers, json=data, timeout=30
    )

    if response.status_code == 200:
        result = response.json()
        result_text = result["choices"][0]["message"]["content"].strip()

        # 提取JSON部分
        json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        else:
            return {"其他": ["无法解析JSON格式"]}


def classify_text_with_metadata(
    bmh: str, text: str, cleaned_ocr: str, cluster: int, idx: int, total: int
) -> Tuple[Dict, int]:
    """处理单个文本并返回结果和索引"""
    print(f"正在处理第 {idx+1}/{total} 条文本...")

    # 调用API进行分类
    classification = classify_text(text)

    # 添加延迟避免API限制
    time.sleep(0.5)

    return {
        "bmh": bmh,
        "ocr": text,
        "cleaned_ocr": cleaned_ocr,
        "cluster": cluster,
        "classification": json.dumps(classification, ensure_ascii=False),
    }, idx


def process_texts(n_jobs: int = 10):
    """使用joblib并行处理所有文本并进行分类"""
    df = pd.read_csv("test2.csv")

    print(f"开始处理 {len(df)} 条文本，使用 {n_jobs} 个并行进程...")

    # 准备所有任务
    tasks = []
    for idx, row in df.iterrows():
        tasks.append(
            (row["bmh"], row["ocr"], row["cleaned_ocr"], row["cluster"], idx, len(df))
        )

    # 并行处理
    results_with_indices = Parallel(n_jobs=n_jobs)(
        delayed(classify_text_with_metadata)(*task)
        for task in tqdm(tasks, desc="并行处理文本")
    )

    # 按原始顺序排序结果
    results_with_indices.sort(key=lambda x: x[1])
    results = [result for result, _ in results_with_indices]

    # 创建新的DataFrame
    result_df = pd.DataFrame(results)

    # 保存结果
    result_df.to_csv("test2_classified.csv", index=False, encoding="utf-8-sig")
    print(f"分类完成！结果已保存到 test2_classified.csv")

    return result_df


if __name__ == "__main__":
    process_texts()