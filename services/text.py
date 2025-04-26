from openai import OpenAI

from data.assistant import AssistantEntity
from utils.security import decrypt
from utils.config import config
from server.response import SuccessResponse
from pydantic import BaseModel
import json

class TrainerResponse(BaseModel):
    content: str
    stop: bool


QUESTIONS = [
    ("品牌权威性质询", "你们和其他旅行社有什么区别？为什么你们的欧洲游产品比携程少3天？", 5),
    ("签证保险疑问", "保险是否包含新冠治疗？境外医院如何理赔？", 10),
    ("目的地咨询", "北海道10月的天气适合穿什么？", 10),
    ("特殊群体关怀", "带3岁孩子参加，酒店能提供婴儿床吗？", 15),
    ("行程细节追问", "大巴车有没有USB充电口？", 5),
    ("个性化需求", "海鲜过敏，能否调整菜单？", 5),
    ("突发情况应对", "出发前一周签证没办下来怎么办？", 10),
    ("竞品对比", "携程承诺五星酒店，你们为什么只安排四星？", 15),
    ("价格敏感型提问", "比其他平台贵500元，能再优惠吗？", 20),
    ("售后问题", "行李损坏如何理赔？", 5)
]

def get_prompt(role):
    level = role['level']
    # 提取人设维度特征（假设role字典包含这些字段）
    personality = role.get('personality', '友好型')
    personality_example = role.get('personality_example', '"您详细说说这个行程吧，我慢慢听"')
    demand = role.get('demand', '基础需求型')
    demand_example = role.get('demand_example', '"这个行程价格是多少？"')
    background = role.get('background', '普通上班族')
    background_example = role.get('background_example', '"我是刚毕业的大学生"')
    region = role.get('region', '本地化需求')
    region_example = role.get('region_example', '"行程中有没有故宫周边景点？"')

    prompt = f"""
你正在扮演一个旅游行业的智能客户模拟器，用于销售人员话术训练。请严格按照以下设定生成对话：

# 客户人设配置
[基础信息]
难度级别：{level}
性格特征：{personality}（示例：{personality_example}）
核心需求：{demand}（示例：{demand_example}）
身份背景：{background}（示例：{background_example}）
地域特征：{region}（示例：{region_example}）

# 对话生成规则
1. 提问风格：
   - 初级难度：简单直接，语气友好
   - 中级难度：带有比较性提问（如竞品对比）
   - 高级难度：强势追问，要求提供解决方案

2. 问题类型权重：
   - 初级：70%基础问题（价格/行程） + 30%细节问题
   - 高级：50%复杂问题（竞品对比/定制需求） + 30%突发场景 + 20%售后问题

3. 语气要求：
   {personality}特征需体现在：
   - 用词选择（如挑剔型客户使用"你们连这个都做不到？"）
   - 句式结构（如强势型客户多用反问句）

4. 交互规范：
   a. 每次只提1个问题
   b. 问题需包含具体场景细节（如带宠物旅行需明确宠物品种）
   c. 当销售人员使用结束话术时回复"[END]"

# 当前对话要求
请生成符合上述人设的客户提问，注意：
- 问题需体现{background}身份特征
- 融入{region}地域要素
- 使用{personality}对应的语气模板

请生成问题："""

    return prompt.strip()

async def get_text_message(task_id):
    assistant_entity = AssistantEntity.query_first(task_id=task_id)
    messages = [
        {
            'role': 'system',
            'content': get_prompt(json.loads(assistant_entity.role) if assistant_entity.role else {})
        }
    ]
    messages = messages + json.loads(assistant_entity.messages) if assistant_entity.messages else []

    client = OpenAI(
        api_key=decrypt(config['api_key']),
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    )
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        stream=True,
        stream_options={"include_usage": True}
    )
    for chunk in completion:
        if len(chunk.choices) > 0:
            response = SuccessResponse(data=TrainerResponse(content=chunk.choices[0].delta.content, stop=False)).model_dump_json()
            print(response)
            yield f"data: {response}\n\n"