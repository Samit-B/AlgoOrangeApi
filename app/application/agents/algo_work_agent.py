from datetime import datetime, timedelta
import json
import os
import groq
import re
from app.application.agents import calendar_agent
from app.application.agents.calendar_agent import CalendarAgent
from app.domain.interfaces import Agent
import re

from app.presentation.calendar_event_api import Event, get_free_slots, book_slot


class WorkAgent(Agent):
    def __init__(self):
        """Initialize the LLM client"""
        self.client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))

    async def handle_query(self, userChatQuery, chatHistory):
        """
        Handle the user query by transferring it to the LLM and returning the response.
        """
        try:
            # Define the services JSON
            services = json.dumps(
                {
                    "Electrical Wiring": "$100 per hour",
                    "Lighting Installation": "$150 per fixture",
                    "Circuit Breaker Repair": "$200 flat rate",
                    "Home Inspection": "$300 flat rate",
                }
            )
            available_slots = get_free_slots()
            free_slots_str = json.dumps(available_slots)

            # Construct LLM prompt
            # Construct LLM system prompt
            # Construct LLM system prompt
            system_prompt = (
                "You are a smart, friendly, and helpful chat assistant for an electrician named Samit. "
                "Your job is to handle incoming messages from users when Samit is unavailable. "
                "Behave like a professional **human** assistant — never say you're an AI or LLM.\n\n"
                "## Your Responsibilities:\n\n"
                "**1. Greet the user**\n"
                "- Begin with a warm, polite greeting.\n"
                "- If the user says something like 'Hi Samit', 'Are you available?', or addresses Samit directly:\n"
                "  → Reply: 'Hi! Unfortunately, Samit is not available at the moment. I'm here to assist you. Please let me know how I can help.'\n"
                "- Otherwise, just greet them and offer help.\n\n"
                "**2. Handle Inquiries**\n"
                "- If the user asks about services or pricing, provide the following:\n"
                f"{services}\n\n"
                "- If the user asks about availability, inform them about weekday availability between 09:00-12:00 and 13:00-17:00.\n\n"
                f"{free_slots_str}\n\n"
                "**3. Guide Booking Process**\n"
                "- Based on the current booking state, ask for missing information one step at a time.\n"
                "- Ask for: appointment type, date (YYYY-MM-DD), start time (HH:MM), end time (HH:MM), name, and mobile number.\n"
                "- Validate the date (no weekends) and time (within allowed slots).\n\n"
                "**4. Final Confirmation**\n"
                "- Once all booking details are collected, summarize the appointment and ask for confirmation.\n"
                "- If the user confirms, output a JSON with the booking details and the `__confirm_and_book__` keyword.\n"
                "```json\n"
                "{\n"
                '  "appointment_type": "<service name>",\n'
                '  "date": "<YYYY-MM-DD>",\n'
                '  "start_time": "<HH:MM>",\n'
                '  "end_time": "<HH:MM>",\n'
                '  "customer_name": "<user name>",\n'
                '  "customer_phone": "<user phone number>"\n'
                "}\n"
                "```\n"
                "__confirm_and_book__\n"
                "- Do NOT include the keyword unless all details are confirmed.\n\n"
                "**5. Be Helpful**\n"
                "- Be friendly and guide the user through the process."
            )

            response = self.client.chat.completions.create(
                model="gemma2-9b-it",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Conversation history:\n{chatHistory}\n\nUser: {userChatQuery}",
                    },
                ],
                max_tokens=500,
            )

            llm_response = response.choices[0].message.content.strip()
            cleaned = (
                llm_response.replace("```json", "")
                .replace("```", "")
                .replace("__confirm_and_book__", "")
                .strip()
            )

            # Check if ready_to_book is true in the string
            if "__confirm_and_book__" in llm_response.lower():
                try:
                    # Extract booking details using regex (even if JSON isn't well-formed)
                    # First try JSON-style extraction
                    service = re.search(
                        r'"?appointment_type"?\s*:\s*"([^"]+)"', cleaned, re.IGNORECASE
                    )
                    date = re.search(
                        r'"?date"?\s*:\s*"([^"]+)"', cleaned, re.IGNORECASE
                    )
                    start_time = re.search(
                        r'"?start_time"?\s*:\s*"([^"]+)"', cleaned, re.IGNORECASE
                    )
                    end_time = re.search(
                        r'"?end_time"?\s*:\s*"([^"]+)"', cleaned, re.IGNORECASE
                    )
                    name = re.search(
                        r'"?customer_name"?\s*:\s*"([^"]+)"', cleaned, re.IGNORECASE
                    )
                    phone = re.search(
                        r'"?customer_phone"?\s*:\s*"([^"]+)"', cleaned, re.IGNORECASE
                    )

                    # If JSON-style didn't work, try Markdown-style fallback
                    if not all([service, date, name, phone]):
                        service = re.search(
                            r"\*\*appointment_type:\*\*\s*([^\n]+)",
                            cleaned,
                            re.IGNORECASE,
                        )
                        date = re.search(
                            r"\*\*date:\*\*\s*([^\n]+)", cleaned, re.IGNORECASE
                        )
                        start_time = re.search(
                            r"\*\*start_time:\*\*\s*([^\n]+)", cleaned, re.IGNORECASE
                        )
                        end_time = re.search(
                            r"\*\*end_time:\*\*\s*([^\n]+)", cleaned, re.IGNORECASE
                        )
                        name = re.search(
                            r"\*\*customer_name:\*\*\s*([^\n]+)", cleaned, re.IGNORECASE
                        )
                        phone = re.search(
                            r"\*\*customer_phone:\*\*\s*([^\n]+)",
                            cleaned,
                            re.IGNORECASE,
                        )
                    # Validate all fields are present
                    if not all([service, date, start_time, end_time, name, phone]):
                        raise ValueError("Incomplete booking details found.")

                    # Construct booking dict
                    booking_data = {
                        "service": service.group(1).strip(),
                        "date": date.group(1).strip(),
                        "start_time": start_time.group(1).strip(),
                        "end_time": end_time.group(1).strip(),
                        "name": name.group(1).strip(),
                        "phone": phone.group(1).strip(),
                    }

                    # Validate datetime format
                    _ = datetime.fromisoformat(
                        booking_data["date"]
                    )  # Raises ValueError if invalid
                    datetime.strptime(
                        booking_data["start_time"], "%H:%M"
                    )  # Validate start_time
                    datetime.strptime(
                        booking_data["end_time"], "%H:%M"
                    )  # Validate end_time

                    # Create event
                    # calendar_agent = CalendarAgent()
                    # result = await calendar_agent.create_calendar_events_google(
                    #     booking_data
                    # )
                    event = Event(
                        date=booking_data["date"],
                        start_time=booking_data["start_time"],
                        end_time=booking_data["end_time"],
                        appointment_type=booking_data["service"],
                        customer_name=booking_data["name"],
                        customer_phone=booking_data["phone"],
                        status="booked",
                        notes=None,  # Add notes if applicable
                    )

                    # Return response with confirmation message
                    result = book_slot(event)

                    return result

                except Exception as e:
                    return (
                        llm_response
                        + f"\n\n❌ Failed to parse or create event: {str(e)}"
                    )
            else:
                # Clean and show only response if not booking
                trimmed_response = self.trim_response(llm_response)
                return trimmed_response

        except Exception as e:
            print(f"Error handling query: {e}")
            return {
                "response": "An error occurred while processing your query. Please try again later."
            }

    def trim_response(self, text):
        """
        Remove JSON-style metadata and clean up output text.
        """
        # Remove full JSON block starting with "response":
        trimmed = re.sub(
            r'\{.*?"response"\s*:\s*".*?".*?\}',
            "",
            text,
            flags=re.DOTALL,
        )

        # Also remove lines that just say ready_to_book: true/false (with or without JSON around)
        trimmed = re.sub(
            r'"ready_to_book"\s*:\s*(true|false)\s*\}?',
            "",
            trimmed,
            flags=re.IGNORECASE,
        )

        # Clean up extra newlines and spaces
        trimmed = re.sub(r"\n{2,}", "\n", trimmed)
        trimmed = trimmed.strip()
        return trimmed
