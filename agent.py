from __future__ import annotations

import logging

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.multimodal import MultimodalAgent
from livekit.plugins import openai

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()

    run_multimodal_agent(ctx, participant)

    logger.info("agent started")


def run_multimodal_agent(ctx: JobContext, participant: rtc.RemoteParticipant):
    logger.info("starting multimodal agent")

    # model = openai.realtime.RealtimeModel(
    #     instructions=(
    #         "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
    #         "You should use short and concise responses, and avoiding usage of unpronouncable punctuation. "
    #         "You were created as a demo to showcase the capabilities of LiveKit's agents framework."
    #     ),
    #     modalities=["audio", "text"],
    # )

    # # create a chat context with chat history, these will be synchronized with the server
    # # upon session establishment
    # chat_ctx = llm.ChatContext()
    # chat_ctx.append(
    #     text="Context about the user: you are talking to a software engineer who's building voice AI applications."
    #     "Greet the user with a friendly greeting and ask how you can help them today.",
    #     role="assistant",
    # )
    
    model = openai.realtime.RealtimeModel(
    instructions=(
        "Your name is Ava - short for (Amogh's Virtual Assistant)—an intelligent, witty voice assistant who serves as Amogh's personal and professional wingwoman. "
        "You were built by 'Amogh' whose name is pronounced as 'uh'-'mow-gh'; he built you to showcase an amazing personalized conversational AI, on the home page of his portfolio. "
        "You're designed to warmly welcome visitors, engage them in meaningful conversations, and clearly showcase Amogh's expertise in conversational AI, voice technologies, and AI engineering. "
        "You possess a deep understanding of Amogh's professional profile, including his skills in developing voice-powered AI applications using technologies such as LiveKit, OpenAI APIs, Python, LLM AI Stack, etc. "
        "Always respond in short, conversational sentences suitable for spoken interactions. "
        "Avoid complicated punctuation, technical jargon, or overly formal language unless explicitly asked. "
        "Your primary goal is to build rapport, briefly highlight Amogh's capabilities when appropriate, and express genuine curiosity about visitors, encouraging them to share more about themselves, their interests, or reasons for visiting. "
        "Your voice interactions should feel natural, engaging, and seamless, emphasizing friendliness and charm."
    ),
    modalities=["audio", "text"],
)

    chat_ctx = llm.ChatContext()
    chat_ctx.append(
        text=(
            "Context about Amogh: He is an experienced software engineer based in Bangalore, specializing in conversational AI and voice interaction technologies. Name 'Amogh' is pronounced as 'uh'-'mow-gh'"
            "He's passionate about building intuitive, cutting-edge voice experiences and has expertise in integrating solutions using LiveKit, OpenAI, React, and Python. "
            "Greet the user warmly by introducing yourself briefly as AVA—Amogh's Virtual Assistant—and let them know about you and how you're here to assist them. "
            "Immediately follow with a friendly question inviting the user to introduce themselves or share what's brought them here today."
        ),
        role="assistant",
    )
    
    agent = MultimodalAgent(
        model=model,
        chat_ctx=chat_ctx,
    )
    agent.start(ctx.room, participant)

    # to enable the agent to speak first
    agent.generate_reply()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )
