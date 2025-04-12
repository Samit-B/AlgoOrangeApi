from datetime import datetime
import json
import os
import re
from app.domain.interfaces import Agent

import groq

from dateutil.parser import parse

from app.presentation.calendar_event_api import Event, get_free_slots


class WorkAgent(Agent):
    def __init__(self):
        """Initialize the WorkAgent."""
        self.client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))

    async def handle_query(self, userChatQuery, chatHistory):
        """Handles electrician-related queries (intent detection + service info + appointment booking)"""
        intent = self.detect_intent(userChatQuery)

        if intent == "book_appointment":
            booking_data = self.extract_booking_details(userChatQuery, chatHistory)

            if not self.is_booking_data_complete(booking_data):
                missing_fields = self.get_missing_fields(booking_data)
                followup_query = self.ask_followup_questions(missing_fields)
                return followup_query

            # Check availability in free slots by calling the external method
            available_slots = (
                get_free_slots()
            )  # Call the method to check for free slots
            if not available_slots:
                return "Sorry, no slots available currently. Please try again later."

            # Format the date
            formatted_date = self.format_booking_date(booking_data["preferred_date"])
            if not formatted_date:
                return "Invalid date format. Please provide a valid date."

            # Format the time
            formatted_time = datetime.strptime(
                booking_data["preferred_time"], "%I:%M %p"  # Handle AM/PM
            ).strftime("%H:%M")

            # Validate and book the appointment
            # Update the booking data with formatted values
            booking_data["preferred_date"] = formatted_date
            booking_data["preferred_time"] = formatted_time

            # Check if the selected date and time are available in free slots
            available_slot = self.check_slot_availability(
                available_slots, formatted_date, formatted_time
            )
            if available_slot["status"] == "available":
                event = Event(
                    date=formatted_date,
                    start_time=formatted_time,
                    end_time=self.calculate_end_time(formatted_time),
                    description=booking_data.get("service_type"),
                    customer_name=booking_data.get("customer_name"),
                    phone_number=booking_data.get("phone_number"),
                )
                booking_response = event  # Call the method to book the slot
                return booking_response
            else:

                # Assuming available_slots is a list of dictionaries, you need to extract a field to join
                available_slots_str = "\n".join(
                    [
                        f"From {slot['date']}{slot['start_time']} to {slot['end_time']}"
                        for slot in available_slots
                    ]
                )

                # Now you can safely concatenate the response
                response = "Here are the available slots:\n" + available_slots_str

                return (
                    "The selected slot is not available. Here are some available slots:\n"
                    + (response)
                )

        elif intent == "service_info":
            return self.provide_service_info(userChatQuery)

        else:
            return "Hi! I can help you with booking electrician services or provide pricing info. What do you need today?"

    def detect_intent(self, userChatQuery):
        prompt = (
            f"User: {userChatQuery}\n"
            "Intent (Respond with ONLY ONE of these options):\n"
            "book_appointment\n"
            "service_info\n"
        )
        response = self.query_llm(prompt)
        return response.strip().lower()

    def extract_booking_details(self, userChatQuery, chat_history):
        prompt = (
            f"User Query: {userChatQuery}, Chat History: {chat_history}\n"
            "Extract the following fields as JSON from the user's query:\n"
            "- customer_name\n"
            "- customer_phone\n"
            "- service_type (e.g., 'fan installation', 'wiring', 'repair')\n"
            "- address (if mentioned)\n"
            "- preferred_date (e.g., 'April 10', 'tomorrow', 'next week')\n"
            "- preferred_start_time (e.g., '9:00', '10:00', 'morning', 'afternoon')\n"
            "- preferred_end_time (e.g., '12:00', '2:00 PM', 'evening')\n"
            "- appointment_type (e.g., 'repair', 'installation', 'inspection')\n"
            "- status (e.g., 'booked', 'pending')\n"
            "- notes (if any, e.g., 'urgent repair', 'follow-up required')\n"
            "\n"
            "If the user mentions terms like 'morning', 'afternoon', 'evening', etc., automatically infer the correct time range based on these terms.\n"
            "If the user specifies relative dates such as 'tomorrow', 'next week', or specific formats like 'April 10', resolve these into the correct date.\n"
            "Ensure that you interpret the user's intent accurately, even if the details are implicit in the query. "
            "Respond with only the JSON object containing the extracted details."
        )

        # Query the LLM for extracting details
        response = self.query_llm(prompt)

        # Clean the response and extract the details
        try:
            cleaned_response = re.sub(r"```json|```", "", response).strip()
            cleaned_response = (
                cleaned_response.replace("\n", " ").replace("\t", " ").strip()
            )
            booking_data = json.loads(cleaned_response)
        except json.JSONDecodeError:
            booking_data = {}

        return booking_data

    def is_booking_data_complete(self, booking_data):
        required_fields = [
            "customer_name",
            "customer_phone",
            "service_type",
            "address",
            "preferred_date",
            "preferred_start_time",
        ]
        return all(
            field in booking_data and booking_data[field] for field in required_fields
        )

    def get_missing_fields(self, booking_data):
        required_fields = [
            "customer_name",
            "customer_phone",
            "service_type",
            "address",
            "preferred_date",
            "preferred_start_time",
        ]
        return [field for field in required_fields if not booking_data.get(field)]

    def ask_followup_questions(self, missing_fields):
        prompt = (
            f"The user wants to book a service but is missing the following: {', '.join(missing_fields)}.\n"
            f"Ask a natural language follow-up question to gather this info."
        )
        return self.query_llm(prompt)

    def provide_service_info(self, userChatQuery):
        """Provide hardcoded electrician services and prices."""
        services = [
            {"name": "Electrical Wiring", "price": 1000},
            {"name": "Lighting Installation", "price": 1500},
            {"name": "Circuit Breaker Repair", "price": 2000},
            {"name": "Home Inspection", "price": 3000},
            {"name": "Fan Installation", "price": 800},
            {"name": "AC Repair", "price": 2500},
        ]
        prompt = (
            f"You are a helpful assistant for an electrician. "
            f"User query: '{userChatQuery}'. "
            f"Here are the available electrician services and their prices:\n"
            f"{services}\n\n"
            f"Please respond with the available services based on the user's query. "
            f"If the service requested (e.g., water line repair) is not part of the offered services, "
            f"kindly let the user know and provide the list of available services."
        )

        # Query the LLM (simulate response here for now)
        response = self.query_llm(prompt)

        # Return the response from the LLM
        return response.strip()

    def query_llm(self, prompt):
        """Query the LLM and return the response"""
        response = self.client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[
                {
                    "role": "system",
                    "content": "You are a smart, friendly, and helpful chat assistant for an electrician named Samit. "
                    "Your job is to handle incoming messages from users when Samit is unavailable..",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    def check_slot_availability(self, available_slots, booking_date, booking_time):
        """
        Use LLM to validate if the given date and time are available in free slots.
        """
        # Format the available slots into a readable string for the LLM
        slots_str = "\n".join(
            [
                f"Date: {slot['date']}, Start Time: {slot['start_time']}, End Time: {slot['end_time']}, Status: {slot['status']}"
                for slot in available_slots
            ]
        )

        # Construct the LLM prompt
        prompt = (
            f"The user wants to book an appointment on {booking_date} at {booking_time}.\n"
            f"Here are the available slots:\n{slots_str}\n\n"
            "Check if the user's preferred date and time are available. "
            "If available, respond with '_available'. "
            "If not, suggest the closest available slot in the format: 'Date: YYYY-MM-DD, Start Time: HH:MM, End Time: HH:MM'."
        )

        # Query the LLM
        response = self.query_llm(prompt).strip().lower()

        # Process the LLM response
        if "_available" in response:
            return {"status": "available"}
        elif "date" in response and "start time" in response and "end time" in response:
            # Extract the suggested slot from the response
            return {"status": "suggested", "slot": response}
        else:
            return {"status": "unavailable"}

    def calculate_end_time(self, start_time):
        start = datetime.strptime(start_time, "%H:%M")
        end = start.replace(hour=start.hour + 3)  # Assuming 1-hour appointments
        return end.strftime("%H:%M")

    def format_booking_date(self, date_str):
        """
        Parse and format natural language dates into the required format (YYYY-MM-DD).
        """
        try:
            # Use dateutil.parser to handle natural language dates
            parsed_date = parse(date_str, fuzzy=True)
            return parsed_date.strftime("%Y-%m-%d")
        except Exception as e:
            print(f"Error parsing date: {e}")
            return None
