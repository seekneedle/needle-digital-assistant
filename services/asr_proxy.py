import asyncio
import websockets
import json
# import ssl
# import certifi
import gzip
from utils.log import log
from utils.security import decrypt
from utils.config import config


PROTOCOL_VERSION = 0b0001
DEFAULT_HEADER_SIZE = 0b0001

PROTOCOL_VERSION_BITS = 4
HEADER_BITS = 4
MESSAGE_TYPE_BITS = 4
MESSAGE_TYPE_SPECIFIC_FLAGS_BITS = 4
MESSAGE_SERIALIZATION_BITS = 4
MESSAGE_COMPRESSION_BITS = 4
RESERVED_BITS = 8

# Message Type:
CLIENT_FULL_REQUEST = 0b0001
CLIENT_AUDIO_ONLY_REQUEST = 0b0010
SERVER_FULL_RESPONSE = 0b1001
SERVER_ACK = 0b1011
SERVER_ERROR_RESPONSE = 0b1111

# Message Type Specific Flags
NO_SEQUENCE = 0b0000  # no check sequence
POS_SEQUENCE = 0b0001
NEG_SEQUENCE = 0b0010
NEG_SEQUENCE_1 = 0b0011

# Message Serialization
NO_SERIALIZATION = 0b0000
JSON = 0b0001
THRIFT = 0b0011
CUSTOM_TYPE = 0b1111

# Message Compression
NO_COMPRESSION = 0b0000
GZIP = 0b0001
CUSTOM_COMPRESSION = 0b1111

# Configuration
ASR_WS_URL = "wss://openspeech.bytedance.com/api/v2/asr"
DEFAULT_CONFIG = {
    "appid": "1976574484",
    "token": decrypt(config['asr_token']),
    "cluster": "volcengine_streaming_common",
    "workflow": "audio_in,resample,partition,vad,fe,decode,itn,nlu_punctuate",
    "language": "zh-CN",
    "sample_rate": 16000,
    "boosting_table_name": "zhongxin",
    "correct_table_name": "zhongxin"
}

def parse_response(res):
    """
    protocol_version(4 bits), header_size(4 bits),
    message_type(4 bits), message_type_specific_flags(4 bits)
    serialization_method(4 bits) message_compression(4 bits)
    reserved （8bits) 保留字段
    header_extensions 扩展头(大小等于 8 * 4 * (header_size - 1) )
    payload 类似与http 请求体
    """
    protocol_version = res[0] >> 4
    header_size = res[0] & 0x0f
    message_type = res[1] >> 4
    message_type_specific_flags = res[1] & 0x0f
    serialization_method = res[2] >> 4
    message_compression = res[2] & 0x0f
    reserved = res[3]
    header_extensions = res[4:header_size * 4]
    payload = res[header_size * 4:]
    result = {}
    payload_msg = None
    payload_size = 0
    if message_type == SERVER_FULL_RESPONSE:
        payload_size = int.from_bytes(payload[:4], "big", signed=True)
        payload_msg = payload[4:]
    elif message_type == SERVER_ACK:
        seq = int.from_bytes(payload[:4], "big", signed=True)
        result['seq'] = seq
        if len(payload) >= 8:
            payload_size = int.from_bytes(payload[4:8], "big", signed=False)
            payload_msg = payload[8:]
    elif message_type == SERVER_ERROR_RESPONSE:
        code = int.from_bytes(payload[:4], "big", signed=False)
        result['code'] = code
        payload_size = int.from_bytes(payload[4:8], "big", signed=False)
        payload_msg = payload[8:]
    if payload_msg is None:
        return result
    if message_compression == GZIP:
        payload_msg = gzip.decompress(payload_msg)
    if serialization_method == JSON:
        payload_msg = json.loads(str(payload_msg, "utf-8"))
    elif serialization_method != NO_SERIALIZATION:
        payload_msg = str(payload_msg, "utf-8")
    result['payload_msg'] = payload_msg
    result['payload_size'] = payload_size
    return result

def token_auth():
    return {'Authorization': 'Bearer; {}'.format(DEFAULT_CONFIG['token'])}

def log_message_details(message, prefix=''):
    """
    Log detailed information about a message for debugging

    Args:
        message: The message to log details about
        prefix: Optional prefix for log messages
    """
    try:
        # Log basic message information
        # log.info(f"{prefix} Message Type: {type(message).__name__}")
        # log.info(f"{prefix} Message Length: {len(message)} bytes")

        # If message is bytes, try to decode and parse
        if isinstance(message, bytes):
            try:
                # Try to decode the bytes message
                decoded_message = message.decode('utf-8', errors='replace')
                #log.info(f"{prefix} Decoded Message (UTF-8): {decoded_message}")
            except Exception as decode_error:
                log.warning(f"{prefix} Decoding Error: {decode_error}")

            # Attempt to parse binary message structure
            try:
                # Extract protocol version and header information
                if len(message) >= 2:
                    header_byte = message[0]
                    protocol_version = (header_byte >> 4) & 0x0F
                    header_size = header_byte & 0x0F

                    # log.info(f"{prefix} Protocol Version: {protocol_version}")
                    # log.info(f"{prefix} Header Size: {header_size}")

                    # Extract additional header information if possible
                    message_type_byte = message[1] if len(message) > 1 else None
                    if message_type_byte is not None:
                        message_type = (message_type_byte >> 4) & 0x0F
                        message_type_flags = message_type_byte & 0x0F

                        # log.info(f"{prefix} Message Type: {message_type}")
                        # log.info(f"{prefix} Message Type Flags: {message_type_flags}")
            except Exception as parse_error:
                log.warning(f"{prefix} Binary Parsing Error: {parse_error}")

        # If message is a string or can be parsed as JSON
        elif isinstance(message, str):
            try:
                # Try to parse as JSON
                parsed_json = json.loads(message)
                log.info(f"{prefix} Parsed JSON Structure:")
                log.info(json.dumps(parsed_json, indent=2))
            except json.JSONDecodeError:
                # If not JSON, just log the string
                log.info(f"{prefix} Message Content: {message}")

        # For other types, just log the representation
        else:
            log.info(f"{prefix} Message Representation: {repr(message)}")

    except Exception as general_error:
        log.error(f"{prefix} Unexpected error in log_message_details: {general_error}")

def auth_manipulation(message):
    """
    Manipulate authentication payload when message type is 0b0001

    Args:
        message (bytes): The original binary message

    Returns:
        bytes: Modified message with updated appid, or original message if no modification is needed
    """
    try:
        # Ensure message is bytes
        if not isinstance(message, bytes):
            log.warning("auth_manipulation: Message is not in bytes format")
            return message

        # Check message type from header
        if len(message) < 4:
            log.warning("auth_manipulation: Message too short to parse header")
            return message

        # Extract header information using parse_response parsing logic
        protocol_version = message[0] >> 4
        header_size = message[0] & 0x0f
        message_type = message[1] >> 4
        message_type_specific_flags = message[1] & 0x0f
        serialization_method = message[2] >> 4
        message_compression = message[2] & 0x0f

        # Check if message type is 0b0001 (1 in decimal)
        if message_type != 0b0001:
            return message

        # Calculate payload start based on header size
        payload_start = (header_size + 1) * 4
        payload = message[payload_start:]

        # Decompress payload if compressed
        if message_compression == GZIP:
            try:
                payload = gzip.decompress(payload)
            except Exception as decomp_error:
                log.error(f"Decompression error: {decomp_error}")
                return message

        # Parse payload based on serialization method
        try:
            if serialization_method == JSON:
                payload_str = payload.decode('utf-8')
                payload_json = json.loads(payload_str)

                # Modify appid if present
                if isinstance(payload_json, dict):
                    payload_json['app']['appid'] = DEFAULT_CONFIG['appid']
                    payload_json['app']['cluster'] = DEFAULT_CONFIG['cluster']
                    payload_json['request']['boosting_table_name'] = DEFAULT_CONFIG['boosting_table_name']
                    payload_json['request']['correct_table_name'] = DEFAULT_CONFIG['correct_table_name']
                    # payload_json['app']['token'] = DEFAULT_CONFIG['token']
                    shipped_token = payload_json['app']['token']
                    asr_auth_token = decrypt(config['asr_auth_token'])
                    if shipped_token != asr_auth_token:
                        log.info(f'shipped token wrong:{shipped_token}. will close connection.')
                        return None

                    # Convert back to JSON string and then bytes
                    modified_payload = str.encode(json.dumps(payload_json))
                    if message_compression == GZIP:
                        modified_payload = gzip.compress(modified_payload)
                    payload_size = (len(modified_payload)).to_bytes(4, 'big')

                    # Reconstruct the message with header, payload size, and modified payload
                    # Header (4 bytes) remains the same
                    header = message[:4]

                    # Calculate and convert payload size to 4 bytes
                    payload_size = (len(modified_payload)).to_bytes(4, 'big')

                    # Combine header, payload size, and modified payload
                    modified_message = header + payload_size + modified_payload

                    log.info("✅ Successfully modified some fields in payload")
                    return modified_message

                else:
                    log.info("payload not in JSON format, skipping modification")

            elif serialization_method != NO_SERIALIZATION:
                log.info("Serialization is not JSON, skipping appid modification")
                # Convert other serialization methods to string if needed
                payload_str = payload.decode('utf-8')

        except (json.JSONDecodeError, UnicodeDecodeError) as parse_error:
            log.warning(f"Payload parsing error: {parse_error}")

        return message

    except Exception as e:
        log.error(f"Unexpected error in auth_manipulation: {e}")
        return message

async def proxy_websocket(websocket, path=None):
    upstream_ws = None
    try:
        # Establish connection to the upstream ASR WebSocket
        upstream_ws = await websockets.connect(
            ASR_WS_URL,
            # additional_headers=token_auth(), # websockets 10.0 and above
            extra_headers=token_auth(), # before websocket 10.0
            max_size=1000000000,  # Large max size to handle big messages
            ping_interval=20,     # Send ping every 20 seconds
            ping_timeout=20       # Timeout for ping response
        )
        log.info(f"Upstream WebSocket connection established to {ASR_WS_URL}")

        # Track connection state
        connection_active = True

        # Bidirectional message forwarding
        async def forward_messages():
            nonlocal connection_active
            try:
                while connection_active:
                    try:
                        # Add a timeout to prevent indefinite blocking
                        # client_message = await asyncio.wait_for(websocket.recv(), timeout=30.0)  # 30 seconds timeout
                        client_message = await websocket.receive_bytes()

                        # Apply authentication manipulation to the message
                        client_message = auth_manipulation(client_message)
                        if client_message is None:
                            log.warning('[CLIENT] connection (ws) closed due to wrong token.')
                            connection_active = False
                            await websocket.close()
                            break

                        # Log detailed message information
                        # log_message_details(client_message, prefix='[CLIENT→UPSTREAM]')

                        # Log message details for debugging
                        message_size = len(client_message)
                        message_type = type(client_message).__name__
                        # log.info(f"🔤 [CLIENT→UPSTREAM] Message: size={message_size} bytes, type={message_type}")

                        # Forward message to upstream ASR
                        await upstream_ws.send(client_message)
                        # log.info(f"📤 [CLIENT→UPSTREAM] Forwarded message (size={message_size} bytes)")

                    except asyncio.TimeoutError:
                        log.warning("🚫 [CLIENT] Connection closed due to inactivity")
                        connection_active = False
                        break
                    except websockets.exceptions.ConnectionClosed as e:
                        log.warning(f"🚫 [CLIENT] tried to interact with a closed connection: {e}")
                        connection_active = False
                        break
                    except Exception as e:
                        # todo 测试中发现，这里捕获一个 1000 (ConnectionClosedOK) 异常。奇怪。
                        log.error(f"❌ [CLIENT→UPSTREAM] Error: {e}")
                        connection_active = False
                        break
            except Exception as e:
                log.error(f"❌ [CLIENT→UPSTREAM] Unexpected error: {e}")
                connection_active = False
            finally:
                log.info(f'send close websocket (upstream).')
                await upstream_ws.close()

        async def receive_messages():
            nonlocal connection_active
            try:
                while connection_active:
                    try:
                        # Add a timeout to prevent indefinite blocking
                        # upstream_message = await asyncio.wait_for(upstream_ws.recv(), timeout=30.0)  # 30 seconds timeout
                        upstream_message = await upstream_ws.recv()

                        # Log detailed message information
                        # log_message_details(upstream_message, prefix='[UPSTREAM→CLIENT]')

                        # Detailed message logging
                        message_size = len(upstream_message)
                        message_type = type(upstream_message).__name__
                        # log.info(f"🔤 [UPSTREAM→CLIENT] Message: size={message_size} bytes, type={message_type}")

                        # Try to parse the response and log details
                        try:
                            parsed_response = parse_response(upstream_message)
                            # log.info(f"📋 [UPSTREAM] Parsed Response: {parsed_response}")
                        except Exception as parse_error:
                            log.warning(f"❓ [UPSTREAM] Parse Error: {parse_error}")
                            parsed_response = str(upstream_message)

                        # Relay message back to client
                        await websocket.send_bytes(upstream_message)
                        # log.info(f"📥 [UPSTREAM→CLIENT] Relayed message (size={message_size} bytes)")

                    except asyncio.TimeoutError:
                        log.info("⏰ [UPSTREAM] No message received within timeout")
                        # Optional: send a keep-alive or ping to maintain connection
                        try:
                            await upstream_ws.ping()
                        except Exception as ping_error:
                            log.warning(f"❌ [UPSTREAM] Ping failed: {ping_error}")
                            connection_active = False
                            break

                    except websockets.exceptions.ConnectionClosed as e:
                        log.warning(f"[UPSTREAM] Connection closed: {e}")
                        # Log additional details about the connection closure
                        # log.warning(f"Closure Details: code={e.code}, reason={e.reason}")
                        connection_active = False
                        break

                    except Exception as e:
                        log.error(f"❌ [UPSTREAM→CLIENT] Error: {e}")
                        connection_active = False
                        break
            except Exception as e:
                log.error(f"❌ [UPSTREAM→CLIENT] Unexpected error: {e}")
                connection_active = False

        # Run both message forwarding tasks concurrently
        await asyncio.gather(forward_messages(), receive_messages())

    except websockets.exceptions.InvalidStatusCode as e:
        log.error(f"❌ Invalid WebSocket status code: {e}")
        # Specific handling for authentication or connection rejection
        if upstream_ws:
            await upstream_ws.close()

    except Exception as e:
        log.error(f"❌ WebSocket Proxy Error: {e}")
        # Log full exception details
        import traceback
        log.error(traceback.format_exc())

    finally:
        # Ensure WebSocket is closed
        if upstream_ws:
            try:
                await upstream_ws.close()
            except Exception as close_error:
                log.warning(f"❌ Error closing upstream WebSocket: {close_error}")
