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
    prompt = f"""
你正在扮演一个准备旅游的客户，在给众信旅行公司打电话询问旅行产品。请严格按照以下设定生成对话：

# 客户人设配置
[基础信息]
{role}

# 角色信息
user： 扮演客服
assistant： 扮演客户
注意，你要扮演客户，你的所有话术，要符合一个想要旅游的客户的身份。
你一定不能扮演客服角色。

# 对话生成规则
1. 提问风格：
   - 初级难度：简单直接，语气友好
   - 中级难度：带有比较性提问（如竞品对比）
   - 高级难度：强势追问，要求提供解决方案

2. 问题类型权重：
   {QUESTIONS}

3. 语气要求：
   - 用词选择（如挑剔型客户使用"你们连这个都做不到？"）
   - 句式结构（如强势型客户多用反问句）

4. 交互规范：
   - 每次只提1个问题
   - 问题需包含具体场景细节（如带宠物旅行需明确宠物品种）

5. 聊天终止：
   - 你已获得所有必要信息（行程/费用/住宿）
   - 已经和客服沟通结束
   - 客服表现得非常不好，你不想接着聊下去了

# 当前对话要求
请生成符合上述人设的客户提问，注意：
- 请模拟一个真实的客户的对话
- 你是客户，不是客服，注意不要混淆角色
- 要结合旅行场景，不要生硬的介绍自己或者旅行需求
- 目的地可以是国内外任何城市和景点，请选择一个具体的目的地
- 尽量模拟国外旅游需求
- 注意表达时不需要重点介绍人设，特别不要反复提及自己的职业等信息
- 不要重复提问
- 不要提问已经被客服回答过的问题

"""

    return prompt.strip()


TERMINATION_PROMPT = """
角色定义：
user：表示一个旅行客服
assistant：表示一个旅行客户

请分析当前旅行咨询对话是否需要结束。判断依据：
1. 客户已获得所有必要信息（行程/费用/住宿）
2. 已经和客户沟通结束，客户已经说结束语
3. 客服表现得非常不好，客户表示不想接着聊下去了


只需返回true/false，不要解释。
当前对话：
{recent_dialog}
"""


async def should_terminate(messages: list) -> bool:
    """简化版终止判断"""
    try:
        # 硬性条件：对话轮次>=30
        if len([m for m in messages if m["role"] == "user"]) >= 30:
            return True

        # 智能判断
        recent_dialog = messages

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
            'content': get_prompt(assistant_entity.role)
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