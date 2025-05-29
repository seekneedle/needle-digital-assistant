import json
import uuid
from typing import Optional

import aiohttp
from pydantic import BaseModel
from server.response import SuccessResponse, FailResponse
from utils.log import log
from utils.security import decrypt
from utils.config import config

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

    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, headers=header, json=request_json) as response:
            response_data = await response.json()
            if response.status == 200:
                if 'data' in response_data:
                    log.info(f'to_speech {reqid} success.')
                    return SuccessResponse(data=TtsResponse(data=response_data['data']))

            log.info(f'openspeech.bytedance tts {reqid} api wrong. {response_data}')
            return FailResponse(error=json.dumps(response_data))
