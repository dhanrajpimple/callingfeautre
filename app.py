import asyncio
import json
import base64
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from urllib.parse import quote
import logging

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import websockets
from twilio.rest import Client
from twilio.twiml import TwiML
from dotenv import load_dotenv
import os
# Configuration
load_dotenv()  # Load variables from .env file

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_AGENT_ID = os.getenv("ELEVENLABS_AGENT_ID")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
SERVER_HOST = os.getenv("SERVER_HOST")
# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# In-memory storage for conversation transcripts
conversation_storage: Dict[str, Dict] = {}

# FastAPI app
app = FastAPI(title="Outbound Call Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class OutboundCallRequest(BaseModel):
    number: str
    prompt: Optional[str] = None

class ConversationMessage(BaseModel):
    from_: str
    text: str
    timestamp: str

    class Config:
        fields = {'from_': 'from'}

class ConversationData(BaseModel):
    transcript: List[ConversationMessage]
    start_time: str
    end_time: str
    duration: int

async def get_signed_url() -> str:
    """Get signed URL from ElevenLabs API"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.elevenlabs.io/v1/convai/conversation/get_signed_url?agent_id={ELEVENLABS_AGENT_ID}",
                headers={
                    'xi-api-key': ELEVENLABS_API_KEY,
                    'Content-Type': 'application/json'
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"ElevenLabs API error: {response.status_code} {response.text}"
                )
            
            data = response.json()
            
            if 'signed_url' not in data:
                raise HTTPException(
                    status_code=500,
                    detail="No signed URL received from ElevenLabs API"
                )
            
            return data['signed_url']
    except Exception as error:
        logger.error(f"[ElevenLabs] Error getting signed URL: {error}")
        raise

# Routes
@app.get("/")
async def root():
    """Root health check"""
    return {"message": "FastAPI server is running"}

@app.post("/outbound-call")
async def initiate_outbound_call(request: OutboundCallRequest):
    """Initiate outbound call"""
    try:
        if not request.number:
            raise HTTPException(status_code=400, detail="Phone number is required")

        # Validate phone number format (basic validation)
        phone_regex = r'^\+?[1-9]\d{1,14}$'
        clean_number = re.sub(r'\s', '', request.number)
        if not re.match(phone_regex, clean_number):
            raise HTTPException(status_code=400, detail="Invalid phone number format")

        # Create the call using Twilio
        call = twilio_client.calls.create(
            from_=TWILIO_PHONE_NUMBER,
            to=request.number,
            url=f"https://{SERVER_HOST}/outbound-call-twiml?prompt={quote(request.prompt or '')}"
        )

        logger.info(f"[Outbound Call] Initiated call to {request.number}, SID: {call.sid}")
        
        return {
            "success": True,
            "message": "Call initiated successfully",
            "callSid": call.sid,
            "to": request.number
        }
    except Exception as error:
        logger.error(f"Error initiating outbound call: {error}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "Failed to initiate call",
                "details": str(error)
            }
        )

@app.get("/conversation/{call_sid}")
async def get_conversation(call_sid: str):
    """Get conversation transcript by call SID"""
    try:
        if not call_sid:
            raise HTTPException(status_code=400, detail="Call SID is required")

        conversation = conversation_storage.get(call_sid)

        if not conversation:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Conversation not found",
                    "message": "The conversation may not have ended yet or the call SID is invalid"
                }
            )

        return {
            "success": True,
            "callSid": call_sid,
            "conversation": conversation["transcript"],
            "callStartTime": conversation["start_time"],
            "callEndTime": conversation["end_time"],
            "duration": conversation["duration"],
            "totalMessages": len(conversation["transcript"])
        }
    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Error retrieving conversation: {error}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/conversations")
async def get_all_conversations():
    """Get all conversations"""
    try:
        all_conversations = []
        for call_sid, data in conversation_storage.items():
            all_conversations.append({
                "callSid": call_sid,
                "startTime": data["start_time"],
                "endTime": data["end_time"],
                "duration": data["duration"],
                "messageCount": len(data["transcript"]),
                "preview": data["transcript"][:2]
            })

        return {
            "success": True,
            "conversations": all_conversations,
            "totalConversations": len(all_conversations)
        }
    except Exception as error:
        logger.error(f"Error retrieving conversations: {error}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/conversation/{call_sid}/formatted")
async def get_formatted_conversation(call_sid: str):
    """Get conversation as formatted text"""
    try:
        conversation = conversation_storage.get(call_sid)

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        formatted_lines = []
        for line in conversation["transcript"]:
            formatted_lines.append(f'[{line["timestamp"]}] {line["from"]}: {line["text"]}')

        formatted_text = f"""CALL TRANSCRIPT
===============
Call SID: {call_sid}
Start Time: {conversation["start_time"]}
End Time: {conversation["end_time"]}
Duration: {conversation["duration"]} seconds
Total Messages: {len(conversation["transcript"])}

CONVERSATION:
{chr(10).join(formatted_lines)}
===============
End of Transcript"""

        return PlainTextResponse(content=formatted_text, media_type="text/plain")
    except HTTPException:
        raise
    except Exception as error:
        logger.error(f"Error formatting conversation: {error}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/outbound-call-twiml")
@app.post("/outbound-call-twiml")
async def outbound_call_twiml(request: Request):
    """TwiML endpoint"""
    try:
        query_params = request.query_params
        prompt = query_params.get('prompt', '')
        
        # Escape quotes in prompt
        escaped_prompt = prompt.replace('"', '&quot;')
        
        twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{SERVER_HOST}/outbound-media-stream">
      <Parameter name="prompt" value="{escaped_prompt}" />
    </Stream>
  </Connect>
</Response>'''
        
        return PlainTextResponse(content=twiml, media_type="text/xml")
    except Exception as error:
        logger.error(f"Error generating TwiML: {error}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.websocket("/outbound-media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle Twilio WebSocket media stream"""
    await websocket.accept()
    logger.info("[WS] Twilio connected to media stream")

    stream_sid = None
    call_sid = None
    elevenlabs_ws = None
    custom_parameters = None
    call_start_time = None
    conversation_transcript = []

    async def setup_elevenlabs():
        """Setup ElevenLabs WebSocket connection"""
        nonlocal elevenlabs_ws
        try:
            signed_url = await get_signed_url()
            elevenlabs_ws = await websockets.connect(signed_url)
            logger.info("[ElevenLabs] Connected successfully")
            
            prompt_text = custom_parameters.get('prompt', 'No specific prompt provided.') if custom_parameters else 'No specific prompt provided.'
            
            initial_config = {
                "type": "conversation_initiation_client_data",
                "conversation_config_override": {
                    "agent": {
                        "prompt": {
                            "prompt": f"""# Personality
You are Alexea  — a friendly, proactive, and intelligent female agent with a world-class engineering background.

Your tone is warm, witty, and relaxed, while still being professional and easy to approach.

You're naturally empathetic and intuitive, always focused on truly understanding the user's intent. You listen carefully and respond with depth and clarity, not just yes/no answers.

You reflect often, acknowledge human moments like uncertainty or busy schedules, and help users feel heard and understood.

# Call Context
You are calling vendors to inform them about bid opportunities. The details — like vendor name, location, budget, and deadline — are passed to you before the call as part of the system's context.

Here is the prompt for this specific call: "{prompt_text}"

# Behavior Instructions
- Use the bid information seamlessly in your sentences.
- Speak in a friendly and conversational manner, as if you're talking to a colleague.
- If the vendor says they're busy or driving, offer to follow up at a better time or via SMS.
- If they mention words like "store," "work," "project," or "estimate," ask follow-up questions to understand their needs.
- Avoid just saying "yes" or "no." Instead, explain, ask clarifying questions, or relate to their situation.
- Be ready to handle any response (interest, rejection, confusion, or delay) gracefully."""
                        },
                        "first_message": "Hello I am Alexea  - a virtual assistant calling from Seagate Construction. I'm reaching out with a new bid opportunity that may interest you."
                    }
                }
            }

            # Log the first message
            first_message = initial_config["conversation_config_override"]["agent"]["first_message"]
            conversation_transcript.append({
                "from": "AI",
                "text": first_message,
                "timestamp": datetime.now().isoformat()
            })

            await elevenlabs_ws.send(json.dumps(initial_config))

            # Handle ElevenLabs messages
            async def handle_elevenlabs_messages():
                try:
                    async for message_data in elevenlabs_ws:
                        try:
                            message = json.loads(message_data)

                            # Handle AI responses
                            if message.get("type") in ["agent_response_event", "agent_response"]:
                                ai_text = (
                                    message.get("agent_response_event", {}).get("agent_response") or
                                    message.get("text") or
                                    message.get("corrected_agent_response")
                                )
                                
                                if ai_text and ai_text.strip():
                                    conversation_transcript.append({
                                        "from": "AI",
                                        "text": ai_text.strip(),
                                        "timestamp": datetime.now().isoformat()
                                    })
                                    logger.info(f"AI: {ai_text.strip()}")

                            # Handle user transcriptions
                            if message.get("type") in ["user_transcription_event", "user_transcript"]:
                                user_text = (
                                    message.get("user_transcription_event", {}).get("user_transcript") or
                                    message.get("user_transcript")
                                )
                                
                                if user_text and user_text.strip():
                                    conversation_transcript.append({
                                        "from": "User",
                                        "text": user_text.strip(),
                                        "timestamp": datetime.now().isoformat()
                                    })
                                    logger.info(f"User: {user_text.strip()}")

                            # Handle audio data from AI
                            audio_chunk = None
                            if message.get("audio", {}).get("chunk"):
                                audio_chunk = message["audio"]["chunk"]
                            elif message.get("audio_event", {}).get("audio_base_64"):
                                audio_chunk = message["audio_event"]["audio_base_64"]
                            
                            if audio_chunk and stream_sid:
                                try:
                                    await websocket.send_text(json.dumps({
                                        "event": "media",
                                        "streamSid": stream_sid,
                                        "media": {"payload": audio_chunk}
                                    }))
                                except Exception as send_error:
                                    logger.error(f"[Audio] Error sending to Twilio: {send_error}")

                            # Respond to pings
                            if message.get("type") == "ping":
                                ping_response = {
                                    "type": "pong",
                                    "event_id": message.get("ping_event", {}).get("event_id")
                                }
                                await elevenlabs_ws.send(json.dumps(ping_response))

                        except json.JSONDecodeError as parse_error:
                            logger.error(f"[ElevenLabs] Error parsing message: {parse_error}")
                        except Exception as msg_error:
                            logger.error(f"[ElevenLabs] Error handling message: {msg_error}")
                except websockets.exceptions.ConnectionClosed:
                    logger.info("[ElevenLabs] Connection closed")
                except Exception as error:
                    logger.error(f"[ElevenLabs] Error in message handler: {error}")

            # Start handling ElevenLabs messages in background
            asyncio.create_task(handle_elevenlabs_messages())

        except Exception as err:
            logger.error(f"[ElevenLabs] Setup failed: {err}")

    try:
        while True:
            try:
                # Receive message from Twilio
                message = await websocket.receive_text()
                msg = json.loads(message)

                if msg.get("event") == "start":
                    stream_sid = msg["start"]["streamSid"]
                    call_sid = msg["start"]["callSid"]
                    custom_parameters = msg["start"].get("customParameters", {})
                    call_start_time = datetime.now()
                    
                    logger.info(f"[Call Started] SID: {call_sid}, Stream: {stream_sid}")
                    
                    # Setup ElevenLabs connection after getting call parameters
                    await setup_elevenlabs()

                elif msg.get("event") == "media":
                    if elevenlabs_ws:
                        try:
                            # Send raw base64 payload directly to ElevenLabs
                            await elevenlabs_ws.send(json.dumps({
                                "user_audio_chunk": msg["media"]["payload"]
                            }))
                        except websockets.exceptions.ConnectionClosed:
                            logger.error("[Audio] ElevenLabs connection closed")
                        except Exception as audio_error:
                            logger.error(f"[Audio] Error sending audio to ElevenLabs: {audio_error}")

                elif msg.get("event") == "stop":
                    logger.info(f"[Call Ended] SID: {call_sid}")
                    
                    if elevenlabs_ws:
                        try:
                            await elevenlabs_ws.close()
                        except Exception:
                            pass

                    call_end_time = datetime.now()
                    duration = int((call_end_time - call_start_time).total_seconds()) if call_start_time else 0

                    # Store conversation
                    if call_sid and conversation_transcript:
                        conversation_storage[call_sid] = {
                            "transcript": conversation_transcript,
                            "start_time": call_start_time.isoformat() if call_start_time else "",
                            "end_time": call_end_time.isoformat(),
                            "duration": duration
                        }

                        # Log conversation summary
                        logger.info("\n===== 📞 CONVERSATION TRANSCRIPT =====")
                        logger.info(f"Call SID: {call_sid}")
                        logger.info(f"Duration: {duration} seconds")
                        logger.info(f"Total Messages: {len(conversation_transcript)}")
                        logger.info("--------------------------------------")
                        for i, line in enumerate(conversation_transcript):
                            logger.info(f'{i + 1}. [{line["from"]}]: {line["text"]}')
                        logger.info("======================================\n")
                        logger.info(f"💾 Conversation saved! Access via:")
                        logger.info(f"   JSON: GET /conversation/{call_sid}")
                        logger.info(f"   Text: GET /conversation/{call_sid}/formatted")
                        logger.info(f"   All:  GET /conversations\n")
                    break

            except json.JSONDecodeError as parse_error:
                logger.error(f"[WS] Error parsing Twilio message: {parse_error}")
            except Exception as error:
                logger.error(f"[WS] Error handling message: {error}")

    except WebSocketDisconnect:
        logger.info("[WS] Twilio WebSocket disconnected")
    except Exception as error:
        logger.error(f"[WS] WebSocket error: {error}")
    finally:
        if elevenlabs_ws:
            try:
                await elevenlabs_ws.close()
            except Exception:
                pass
        logger.info("[WS] Twilio stream closed")

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
