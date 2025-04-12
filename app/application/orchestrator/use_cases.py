import os
import json
import groq  # Assuming you are using Groq's API
import tiktoken
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from app.application.agents.ms_excel_agent import MSExcelAgent
from app.application.agents.ms_word_agent import MSWordAgent
from app.application.agents.gmail_agent import GmailAgent
from app.application.agents.medical_agent import MedicalAgent
from app.application.agents.outlook_agent import OutlookAgent
from app.application.agents.pdf_agent import PdfAgent
from app.application.agents.social_media_agent import SocialMediaAgent
from app.application.agents.calendar_agent import CalendarAgent
from app.application.agents.general_agent import GeneralAgent
from app.application.agents.web_agent import WebAgent
from app.application.agents.project_agent import ProjectAgent

# from app.application.agents.work_agent import WorkAgent
from app.application.agents.algo_work_agent import WorkAgent
from app.core.di import Container
from openai import OpenAI

load_dotenv()

# Tokenizer setup
TOKEN_LIMIT = 4000  # Keep buffer under Groq’s 5000-token limit
encoder = tiktoken.get_encoding("cl100k_base")  # Use the right tokenizer

# Conversation Memory Setup
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


def chunk_text(text: str, token_limit: int = TOKEN_LIMIT):
    """Splits text into chunks based on token limits."""
    if not isinstance(text, str):  # Ensure input is a string
        text = str(text)  # Convert non-string input to string

    if not text.strip():  # Avoid empty input
        return []

    tokens = encoder.encode(text)
    chunks = [
        encoder.decode(tokens[i : i + token_limit])
        for i in range(0, len(tokens), token_limit)
    ]

    return chunks


class Orchestrator:
    def __init__(self, userChatQuery: str):
        self.userChatQuery = userChatQuery
        self.chatHistory = memory.load_memory_variables({})["chat_history"]
        self.client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))

    async def route_query(self):
        """Routes user query to the correct agent while integrating Voice, Memory, and Sentiment Analysis."""

        # 🚀 Chunk input if it's too long
        query_chunks = chunk_text(self.userChatQuery)
        history_chunks = chunk_text(self.chatHistory)

        if not query_chunks:  # Ensure there's valid input
            return "Error: Empty query provided."

        responses = []

        for query_chunk in query_chunks:
            for history_chunk in history_chunks or [""]:  # Ensure history exists
                try:
                    response = self.client.chat.completions.create(
                        model="gemma2-9b-it",  # Use the best available Groq model
                        # model="mistralai/mistral-small-24b-instruct-2501:free",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a smart AI assistant that engages in **conversational** discussions. "
                                "Your goal is to understand user queries, recall past conversations, and provide responses in a **natural and interactive way**."
                                "\n\n"
                                "🔹 You must route queries to the correct agent while maintaining the conversation flow."
                                "🔹 Always consider the **user’s past messages** to provide relevant responses."
                                "🔹 If the user continues a previous topic, stay on track and do not repeat previous explanations."
                                "🔹 Your tone should be friendly, engaging, and interactive."
                                "🔹 You can ask questions to user for all the respones or when they tell someting incomplete."
                                "\n\n"
                                "Available agents and their roles:\n"
                                "📌 'project' → Handles business projects, strategies, execution, and risk management.\n"
                                "📌 'medical' → Provides health-related insights, symptom analysis, and medical recommendations.\n"
                                "📌 'social_media' → Assists with social media planning, branding, and content strategy.\n"
                                "📌 'calendar' → Manages meeting schedules and event organization.\n"
                                "📌 'general' → Covers all other casual or undefined queries.\n"
                                "📌 'web_agent' → (specializes in web scraping).\n "
                                "📌 'pdf_agent' → (specializes in PDF processing).\n "
                                "📌 'gmail' → (specializes in gmail management).\n "
                                "📌 'outlook' → (specializes in email management).\n "
                                "📌 'ms_excel_agent' → (specializes in microfoft excel processing).\n "
                                "📌 'ms_word_agent' → (specializes in microfoft word processing).\n "
                                "📌 'alog_worker' → (specializes in electrical work related or alog_worker).\n "
                                "\n\n"
                                "If the user’s message follows up on a past topic, assume continuity and respond accordingly.",
                            },
                            {
                                "role": "user",
                                "content": f"User Query: {query_chunk} \n"
                                f"Chat History: {history_chunk} \n"
                                "Which agent should handle this query? Return only the agent name without any explanation.",
                            },
                        ],
                        max_tokens=10,
                    )
                    decision = response.choices[0].message.content.strip().lower()
                    if decision:  # Ensure response is valid
                        responses.append(decision)

                except Exception as e:
                    print(f"Error in API request: {e}")  # Log error
                    continue  # Skip to next chunk if an error occurs

        if not responses:  # Prevent empty `max()` call
            return "Error: No valid response received from Groq API."

        # Use the most frequent decision
        decision = max(set(responses), key=responses.count)

        # 🔗 Map decision to corresponding agent
        agent_mapping = {
            "medical": MedicalAgent,
            "project": ProjectAgent,
            "social_media": SocialMediaAgent,
            "calendar": CalendarAgent,
            "general": GeneralAgent,
            "web_agent": WebAgent,
            "pdf_agent": PdfAgent,
            "gmail": GmailAgent,
            "outlook": OutlookAgent,
            "ms_excel_agent": MSExcelAgent,
            "ms_word_agent": MSWordAgent,
            "alog_worker": WorkAgent,
        }

        agent_class = agent_mapping.get(decision)

        if not agent_class:
            return "Error: Agent not found."

        # 📌 Instantiate the chosen agent
        agent = agent_class()

        # 🚀 Chunk user query & history for agent processing
        agent_responses = []
        for query_chunk in query_chunks:
            for history_chunk in history_chunks or [""]:
                agent_response = await agent.handle_query(query_chunk, history_chunk)
                agent_responses.append(agent_response)

        final_response = " ".join(
            json.dumps(resp) if isinstance(resp, dict) else str(resp)
            for resp in agent_responses
        )

        # 📝 Update Memory with user query and agent response
        memory.save_context({"input": self.userChatQuery}, {"output": final_response})

        return final_response  # Returns the full response, not just the agent name
