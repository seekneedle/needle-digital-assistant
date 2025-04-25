from openai import OpenAI, APIError
from utils.security import decrypt
from utils.config import config
from utils.log import log
import multiprocessing
import asyncio
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import traceback

client = OpenAI(
        api_key=decrypt(config['api_key']),
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    )

def qwen_call(messages, return_type: str, task_id: str, job_name: str, model_name: str):
    # task_id 和 job_name 只用于 logging 目的
    log.info(f'{task_id} {model_name} {job_name} begins')
    t0 = datetime.now()
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_format={'type': return_type}
        )
        log.info(f'{task_id} {model_name} {job_name} done, cost {datetime.now() - t0}')
        log.info(f'{task_id} {model_name} {job_name} request_id: {completion.id}, usage: {completion.usage}')
        return completion.choices[0].message.content
    except APIError as e:
        log.info(f'{task_id} {model_name} {job_name} APIError: {e.status_code}, {e.code}, {e.message}')
        return ''
    except Exception as e:  # 其他异常（如网络问题）
        log.info(f'{task_id} {model_name} {job_name} api Exception: {str(e)}')
        return ''

def qwen_stream_call(messages, queue, model_name: str):
    # task_id 和 job_name 只用于 logging 目的
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=True,
        stream_options={'include_usage': True} # 得到 token 使用情况统计
    ) # 貌似是第一个 chunk 返回时才返回
    for chunk in completion:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content is not None: # 真正的回复
            queue.put(f'data: {delta.content}\n\n')
    queue.put(None)

# wrapper for qwen_stream_call()
async def stream_generate_ex(messages, task_id: str, job_name: str, model_name: str):
    # log.info(f'stream_call {task_id} {model_name} {job_name} WRAPPER before calling qwen')
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=qwen_stream_call, args=(messages, queue, model_name))
    # log.info(f'stream_call {task_id} {model_name} {job_name} WRAPPER before process.start()')
    process.start()
    # log.info(f'stream_call {task_id} {model_name} {job_name} WRAPPER after process.start()')

    try:
        cnt = 0
        while True:
            # 异步监听队列（避免阻塞事件循环）
            data = await asyncio.get_event_loop().run_in_executor(
                None,  # 使用默认线程池
                queue.get  # 阻塞调用，但通过线程池转为异步
            )
            if cnt == 0:
                log.info(f'stream_call {task_id} {model_name} {job_name} WRAPPER first chunk received')
            cnt += 1
            if data is None: # 结束信号
                break
            yield data  # 返回 SSE 数据
    finally:
        process.join()  # 确保进程退出

if __name__ == '__main__':
    pass
