import json
import uuid
from typing import Optional

import aiohttp
from pydantic import BaseModel
from server.response import SuccessResponse, FailResponse
from utils.log import log
from utils.security import decrypt
from utils.config import config
from aiohttp import ClientError, ClientPayloadError, ClientResponseError
import asyncio

class TtsRequest(BaseModel):
    voice_type: Optional[str] = None  # 允许空值
    emotion: Optional[str] = None
    language: Optional[str] = None
    speed_ratio: float = 1.0
    text: str

class TtsResponse(BaseModel):
    data: str # based64 encoded

appid = '1976574484'
access_token = decrypt(config['tts_token'])
cluster = 'volcano_tts'
language = 'cn'
voice_type = 'BV700_V2_streaming'
emotion = 'professional'
host = 'openspeech.bytedance.com'

# 配置常量
MAX_RETRIES = 3  # 最大重试次数
RETRY_DELAY = 1.0  # 初始重试延迟(秒)，每次重试会指数增加
TIMEOUT = 30  # 请求超时时间(秒)

async def to_speech(tts_request: TtsRequest):
    text = tts_request.text

    final_voice = tts_request.voice_type.strip() if tts_request.voice_type else voice_type
    final_emotion = tts_request.emotion.strip() if tts_request.emotion else emotion
    final_language = tts_request.language.strip() if tts_request.language else language

    api_url = f'https://{host}/api/v1/tts'
    header = {'Authorization': f'Bearer;{access_token}'}
    reqid = str(uuid.uuid4())
    request_json = {
        'app': {
            'appid': appid,
            'token': 'access_token',
            'cluster': cluster
        },
        'user': {
            'uid': '388808087185088'
        },
        'audio': {
            'voice_type': final_voice,
            'emotion': final_emotion,
            'encoding': 'mp3',
            'speed_ratio': tts_request.speed_ratio,
            'volume_ratio': 1.0,
            'pitch_ratio': 1.0,
            'language': final_language
        },
        'request': {
            'reqid': reqid,
            'text': text,
            'text_type': 'plain',
            'operation': 'query',
            'with_frontend': 1,
            'frontend_type': 'unitTson'
        }
    }
    log.info(f'to_speech id: {reqid}. Request: {tts_request}')

    # 重试机制
    for attempt in range(MAX_RETRIES + 1):  # 尝试次数 = 重试次数 + 1
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as session:
                async with session.post(api_url, headers=header, json=request_json) as response:
                    # 先读取原始响应，以便错误时记录
                    raw_response = await response.read()

                    try:
                        # 尝试解析JSON
                        response_data = json.loads(raw_response)
                    except json.JSONDecodeError:
                        # JSON解析失败时记录原始响应
                        log.error(f'TTS {reqid} Invalid JSON response: {raw_response[:200]}...')
                        response_data = {"error": "Invalid JSON response"}

                    if response.status == 200:
                        if 'data' in response_data:
                            log.info(f'TTS {reqid} success')
                            return SuccessResponse(data=TtsResponse(data=response_data['data']))
                        else:
                            log.warning(f'TTS {reqid} missing data field: {response_data}')

                    # 非200状态码或缺少data字段
                    log.error(f'TTS {reqid} API error (status {response.status}): {response_data}')
                    return FailResponse(error=json.dumps(response_data))

        except (ClientError, ClientPayloadError, asyncio.TimeoutError) as e:
            # 可重试的网络错误
            if attempt < MAX_RETRIES:
                wait_time = RETRY_DELAY * (2 ** attempt)  # 指数退避
                log.warning(
                    f'TTS {reqid} attempt {attempt + 1}/{MAX_RETRIES} failed ({type(e).__name__}), retrying in {wait_time:.1f}s...')
                await asyncio.sleep(wait_time)
            else:
                error_msg = f'TTS {reqid} failed after {MAX_RETRIES} attempts: {str(e)}'
                log.error(error_msg)
                return FailResponse(error=error_msg)

        except Exception as e:
            # 不可预知的错误，不重试
            error_msg = f'TTS {reqid} unexpected error: {str(e)}'
            log.error(error_msg, exc_info=True)
            return FailResponse(error=error_msg)

    # 理论上不会执行到这里，但为了完整性
    return FailResponse(error=f'TTS {reqid} failed after {MAX_RETRIES} attempts')
