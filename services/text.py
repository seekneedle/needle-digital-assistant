from openai import AsyncOpenAI

from data.assistant import AssistantEntity
from utils.security import decrypt
from utils.config import config
from server.response import SuccessResponse
from pydantic import BaseModel
import json
from utils.log import log


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

# 当前对话要求
请生成符合上述人设的客户提问，注意：
- 问题需体现{background}身份特征
- 融入{region}地域要素
- 使用{personality}对应的语气模板
- 请模拟一个真实的客户的对话
- 要结合旅行场景，不要生硬的介绍自己或者旅行需求

请生成问题："""

    return prompt.strip()


TERMINATION_PROMPT = """
请分析当前旅行咨询对话是否需要结束。判断依据：
1. 用户已获得所有必要信息（行程/费用/住宿）
2. 用户明确表示没有更多问题
3. 对话轮次已达20次
4. 问题重复率超过30%

只需返回true/false，不要解释。
当前对话（最近3轮）：
{recent_dialog}
"""


async def should_terminate(messages: list) -> bool:
    """简化版终止判断"""
    try:
        # 硬性条件：对话轮次>=30
        if len([m for m in messages if m["role"] == "user"]) >= 30:
            return True

        # 智能判断
        recent_dialog = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in messages[-6:]  # 最近3组问答
        )

        client = AsyncOpenAI(
            api_key=decrypt(config['api_key']),
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        )
        response = await client.chat.completions.create(
            model="qwen-max",
            messages=[{
                "role": "system",
                "content": TERMINATION_PROMPT.format(recent_dialog=recent_dialog)
            }],
            temperature=0.1
        )

        return response.choices[0].message.content.strip().lower() == "true"
    except Exception:
        return False


async def get_text_message(task_id):
    log.info(f'get_text_message query_task: {task_id}')
    assistant_entity = AssistantEntity.query_first(task_id=task_id)
    log.info(f'get_text_message query_task: {task_id}, role: {assistant_entity.role}, message: {assistant_entity.messages}')
    messages = [
        {
            'role': 'system',
            'content': get_prompt(json.loads(assistant_entity.role) if assistant_entity.role else {})
        }
    ]
    messages = messages + json.loads(assistant_entity.messages) if assistant_entity.messages else []

    full_response = []
    try:
        client = AsyncOpenAI(
            api_key=decrypt(config['api_key']),
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        )

        completion = await client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            stream=True,
            stream_options={"include_usage": True}
        )

        # 流式传输
        async for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                log.info(f'get_text_message response text: {content}')
                full_response.append(content)
                yield content

        # 完整响应后判断终止
        if full_response:
            assistant_response = "".join(full_response)
            messages.append({"role": "assistant", "content": assistant_response})

            if await should_terminate(messages):
                yield "[END]"

    finally:
        # 不强制发送结束标记
        pass