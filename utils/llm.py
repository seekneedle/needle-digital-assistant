from openai import AsyncOpenAI, APIError
from utils.security import decrypt
from utils.config import config
from utils.log import log
import multiprocessing
import asyncio
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import traceback

client = AsyncOpenAI(
        api_key=decrypt(config['api_key']),
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    )

async def qwen_call(messages, return_type: str, task_id: str, job_name: str, model_name: str):
    # task_id 和 job_name 只用于 logging 目的
    log.info(f'{task_id} {model_name} {job_name} begins')
    t0 = datetime.now()
    try:
        completion = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_format={'type': return_type}
        )
        log.info(f'{task_id} {model_name} {job_name} done, cost {datetime.now() - t0}')
        log.info(f'{task_id} {model_name} {job_name} request_id: {completion.id}, usage: {completion.usage}')
        return completion.choices[0].message.content
    except APIError as e:
        log.info(f'{task_id} {model_name} {job_name} APIError: {e.code}, {e.message}')
        return ''
    except Exception as e:  # 其他异常（如网络问题）
        log.info(f'{task_id} {model_name} {job_name} api Exception: {str(e)}')
        return ''


if __name__ == '__main__':
    pass
