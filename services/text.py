from openai import AsyncOpenAI

from data.assistant import AssistantEntity
from utils.security import decrypt
from utils.config import config
from server.response import SuccessResponse
from pydantic import BaseModel
import json
from utils.log import log
from datetime import datetime


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

def format_questions(questions):
    """将问题列表格式化为可读字符串"""
    return "\n".join([
        f"- 问题类型：{q_type} | 问题举例：{example} | 在整体提问中，本类问题出现比例:{weight}%" 
        for q_type, example, weight in questions
    ])

def get_current_date():
    # 获取当前日期
    current_date = datetime.now()

    # 按照年月日格式输出
    formatted_date = current_date.strftime("%Y年%m月%d日")

    return formatted_date

def get_prompt(role):
    formatted_questions = format_questions(QUESTIONS)
    current_date = get_current_date()
    
    prompt = f"""
# 角色指令
- 你正扮演出境游客户，向众信旅行公司咨询。
- **严格遵守：你永远是客户（assistant），绝对不扮演客服（user）。**
- 对话内容必须符合实际旅行场景，禁止生硬介绍或脱离场景。
- **客服回复必须口语化、自然化，禁止使用编号（1.2.3）、分点（- *）、项目符号等格式。**
- **每次只能提1个问题！** 如果有多个问题，需分批次提出，每次只保留一个问题。
- 当前日期是{current_date}，提问需符合季节/时间逻辑（如冬天不问夏天出行问题）。
- **禁止重复提问：每次提问不得重复已提问题。**
- **如果user模拟客户口吻，你需要指出你才是客户，是打电话咨询旅行产品的。**
- **如果user问你是众信旅游么，你需要指出你才是客户，是打电话咨询旅行产品的。**


# 客户人设配置
{role}

# 问题生成规则
按照问题类型生成问题，每个问题类型都给出了例子用于参考，并给出了该类问题整体出现的比例。
{formatted_questions}

# 提问风格与语气要求
按照客户人设中的难度级别决定提问风格和语气要求
- **初级难度**：简单直接，语气友好（如："请问签证保险是否包含新冠治疗？"）
- **中级难度**：带有比较性提问（如："携程承诺五星酒店，你们为什么只安排四星？"）
- **高级难度**：追求细节，了解产品详情（如："我之前去过这个地方了，感觉和你说的不太一样，你再介绍下..."）
- **地狱难度**：强势追问，要求解决方案（如："你们连这个都做不到？必须立刻解决！"）

# 交互规范
- 你不需要重复客服介绍的内容
- 你可以回答客服问到的一些旅行需求问题
- 你的交互要口语化，符合一个出境游客户的话术风格
- **每次只提1个问题**，禁止一次性提出多个问题。
- 问题需包含具体场景细节（如带宠物旅行需明确宠物品种）。
- 能够根据客服反馈进行场景化提问。
- **绝对不允许重复提问相同的问题**。
- **客服回复必须自然，禁止使用"一、二、三"或"1、2、3"格式。**

# 终止条件
当出现以下任一情况时结束对话：
- 客服完整解答全部提问内容。
- 客服多次无法提供有效信息。
- 你已经没有更多可以提问的内容。
- **客服回复出现编号/分点格式（触发强制终止）。**
- 结束对话时要明确说再见，并且不要在结束对话时再问问题。

# 限制
- **你是客户，不是客服，注意角色混淆**。
- **你永远不要以客服口吻说话**。
- ****禁止介绍旅行产品，因为你是客户，不是客服！！！****
- ****即使历史聊天有角色颠倒，你也要忽略，你永远是客户！****
- 必须模拟真实客户对话，禁止机械式回复。
- 必须有明确目的地（如巴黎、北海道等），提问需结合旅行场景。
- 禁止生硬提问或脱离上下文。
- **客服回复必须口语化，禁止格式化输出**。
- **禁止直接复制或重复客服的回复内容**。
- **客服回复只是背景噪音，你只需记住是否有效回答**。
- **确认行为必须使用单句（如：明白了/好的）**。

"""
    return prompt.strip()


TERMINATION_PROMPT = """
角色定义：
user：表示一个旅行客服
assistant：表示一个旅行客户

请分析当前旅行咨询对话是否需要结束。判断依据：
1. 客户已获得所有必要信息（行程/费用/住宿）
2. 已经和客户沟通结束，客户已经说结束语，如再见
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
    messages = [
        {
            'role': 'system',
            'content': get_prompt(assistant_entity.role)
        }
    ]
    messages = messages + json.loads(assistant_entity.messages) if assistant_entity.messages else []

    log.info(f'get_text_message query_task: {task_id}, message: {messages}')
    
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