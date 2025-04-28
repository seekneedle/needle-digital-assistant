import json
import uuid
import aiohttp
from pydantic import BaseModel
from server.response import SuccessResponse, FailResponse
from utils.log import log

class TtsRequest(BaseModel):
    text: str

class TtsResponse(BaseModel):
    data: str # based64 encoded

appid = '4823798285'
access_token = 'GmakIY6Um9DZgCQl7Rr0RJNfDukCJ8RB'
cluster = 'volcano_tts'
language = 'cn'
voice_type = 'BV700_V2_streaming'
emotion = 'professional'
host = 'openspeech.bytedance.com'


async def to_speech(tts_request: TtsRequest):
    text = tts_request.text

    api_url = f'https://{host}/api/v1/tts'
    header = {'Authorization': f'Bearer;{access_token}'}
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
            'voice_type': voice_type,
            'encoding': 'mp3',
            'speed_ratio': 1.0,
            'volume_ratio': 1.0,
            'pitch_ratio': 1.0,
        },
        'request': {
            'reqid': str(uuid.uuid4()),
            'text': text,
            'text_type': 'plain',
            'operation': 'query',
            'with_frontend': 1,
            'frontend_type': 'unitTson'
        }
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, headers=header, json=request_json) as response:
            response_data = await response.json()
            if response.status == 200:
                if 'data' in response_data:
                    return SuccessResponse(data=TtsResponse(data=response_data['data']))

            log.info(f'openspeech.bytedance tts api wrong. {response_data}')
            return FailResponse(error=json.dumps(response_data))
