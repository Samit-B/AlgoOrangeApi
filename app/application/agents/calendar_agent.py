from app.domain.interfaces import Agent
from app.infrastructure.services.calendar.calendar_service import CalendarService
import re
from datetime import datetime, time, timedelta
import groq  # Groq's API
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
import pytz
from dateutil.parser import isoparse

local_tz = pytz.timezone("Asia/Kolkata")


SERVICE_ACCOUNT_FILE = r"C:\Algo Orange\algoorangeapi\credentials.json"
SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]
calendar_service: CalendarService
WORKING_HOURS_START = 9
WORKING_HOURS_END = 17
SLOT_DURATION_MINUTES = 240


class CalendarAgent(Agent):

    async def handle_query(self, userChatQuery: str, userChatHistory: str) -> str:

        client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))
        # Query Groq LLM to determine which agent to call
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=(
                [
                    {
                        "role": "system",
                        "content": "You are an AI assistant that determines which calendar action to perform.",
                    },
                    {
                        "role": "user",
                        "content": f"Query: {userChatQuery},chathistoy: {userChatHistory} Identify if the query is related to one of the following actions:"
                        "schedule → If the query is about creating or setting up a new event/meeting,"
                        "get → If the query asks to retrieve, view, or list calendar events, including queries like:"
                        "Get my upcoming meetings"
                        "Get meeting"
                        "Get my meeting"
                        "Show my meetings"
                        "Show me the tomorrow meeting"
                        "Check my meetings on [specific date],"
                        "cancel → If the query involves canceling or deleting an event,"
                        "reschedule → If the query is about modifying or changing the time of an existing event"
                        "Respond with only one word from the list: schedule, get, cancel, or reschedule. Do not provide explanations.",
                    },
                ]
            ),
            max_tokens=10,
        )

        query_lower = response.choices[0].message.content.strip().lower()

        # Determine the type of query (schedule, get, cancel, reschedule)

        if query_lower == "schedule":
            return await self.create_calendar_event_google(userChatQuery)
        elif query_lower == "get":
            return await self.get_events(userChatQuery)
        elif query_lower == "cancel":
            return await self.cancel_event(userChatQuery)
        elif query_lower == "reschedule":
            return await self.reschedule_event(userChatQuery)
        else:

            return "Unknown query type."

    def extract_date_from_query(self, userChatQuery: str):
        # Convert query to lowercase for easier processing
        query_lower = userChatQuery.lower()

        # Extract a specific date if mentioned (e.g., "March 15", "next Monday")

        date_pattern = re.search(
            r"\b(\d{4}-\d{2}-\d{2}|\b\w+\s\d{1,2}(?:st|nd|rd|th)?)\b", query_lower
        )

        if "tomorrow" in query_lower:
            extracted_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        elif "today" in query_lower:
            extracted_date = datetime.now().strftime("%Y-%m-%d")
        elif date_pattern:
            extracted_date = date_pattern.group(1)
        # Optional: Convert extracted text-based dates (e.g., "March 15") to a proper YYYY-MM-DD format
        # try:
        #     extracted_date = datetime.strptime(extracted_date, "%B %d").replace(year=datetime.now().year).strftime("%Y-%m-%d")
        # except ValueError:
        #     pass  # If conversion fails, keep the extracted value as-is
        else:
            extracted_date = None  # No specific date mentioned, assume "upcoming"

        return extracted_date

    async def get_events(self, userChatQuery: str) -> str:
        # Extract date range from the query
        extracted_dates = self.extract_date_from_query(userChatQuery)
        if extracted_dates:
            from_date = datetime.strptime(extracted_dates, "%Y-%m-%d").isoformat() + "Z"
            to_date = (
                datetime.strptime(extracted_dates, "%Y-%m-%d") + timedelta(days=1)
            ).isoformat() + "Z"
        else:
            today = datetime.now()
            from_date = (
                today.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
                + "Z"
            )
            to_date = (
                today.replace(
                    hour=23, minute=59, second=59, microsecond=999999
                ).isoformat()
                + "Z"
            )

        try:
            creds = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=SCOPES
            )
            service = build("calendar", "v3", credentials=creds)
            events_result = (
                service.events()
                .list(
                    calendarId="sachinbfrnd@gmail.com",
                    timeMin=from_date,
                    timeMax=to_date,
                    maxResults=10,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
            events = events_result.get("items", [])

            if not events:
                return "No upcoming events found."

            event_list = []
            for event in events:
                start = event["start"].get("dateTime", event["start"].get("date"))
                event_list.append(f"{start} - {event['summary']}")

            return "\n".join(event_list)
        except Exception as e:
            print(f"Error fetching events: {str(e)}")
            return f"Error fetching events: {str(e)}"

    async def cancel_event(self, userChatQuery: str) -> str:
        from_date, to_date = self.extracted_date_from_query(userChatQuery)

        if not from_date:
            return "Could not determine the date or time from your request."

        event_ids = await self.get_event_ids(from_date, to_date)

        if not event_ids:
            return f"No events found to cancel on {from_date}."

        canceled_count = 0
        for event_id in event_ids:
            try:
                creds = service_account.Credentials.from_service_account_file(
                    SERVICE_ACCOUNT_FILE, scopes=SCOPES
                )
                service = build("calendar", "v3", credentials=creds)
                service.events().delete(
                    calendarId="sachinbfrnd@gmail.com", eventId=event_id
                ).execute()
                canceled_count += 1
                print(f"Successfully deleted event: {event_id}")
            except Exception as e:
                print(f"Failed to cancel event {event_id}: {str(e)}")

        return f"Successfully canceled {canceled_count} event(s) on {from_date}."

    def extracted_date_from_query(self, userChatQuery: str):
        query_lower = userChatQuery.lower()

        # Handling specific date (e.g., "on March 15")
        date_pattern = re.search(r"on (\d{4}-\d{2}-\d{2})", query_lower)

        if "tomorrow" in query_lower:
            extracted_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        elif "today" in query_lower:
            extracted_date = datetime.now().strftime("%Y-%m-%d")
        elif "next week" in query_lower:
            start_of_week = datetime.now() + timedelta(
                days=(7 - datetime.now().weekday())
            )
            end_of_week = start_of_week + timedelta(days=6)
            return start_of_week.strftime("%Y-%m-%d"), end_of_week.strftime("%Y-%m-%d")
        elif "this week" in query_lower:
            start_of_week = datetime.now() - timedelta(days=datetime.now().weekday())
            end_of_week = start_of_week + timedelta(days=6)
            return start_of_week.strftime("%Y-%m-%d"), end_of_week.strftime("%Y-%m-%d")
        elif "next month" in query_lower:
            next_month = datetime.now().replace(day=1) + timedelta(days=31)
            start_of_month = next_month.replace(day=1)
            end_of_month = (start_of_month + timedelta(days=32)).replace(
                day=1
            ) - timedelta(days=1)
            return start_of_month.strftime("%Y-%m-%d"), end_of_month.strftime(
                "%Y-%m-%d"
            )
        elif date_pattern:
            extracted_date = date_pattern.group(1)
        else:
            extracted_date = None

        return extracted_date, extracted_date  # Same date for single-day events

    async def get_event_ids(self, from_date: str, to_date: str):
        try:
            from_date_iso = (
                datetime.strptime(from_date, "%Y-%m-%d")
                .replace(hour=0, minute=0, second=0)
                .isoformat()
                + "Z"
            )
            to_date_iso = (
                datetime.strptime(to_date, "%Y-%m-%d")
                .replace(hour=23, minute=59, second=59)
                .isoformat()
                + "Z"
            )

            creds = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=SCOPES
            )

            service = build("calendar", "v3", credentials=creds)
            events_result = (
                service.events()
                .list(
                    calendarId="sachinbfrnd@gmail.com",
                    timeMin=from_date_iso,
                    timeMax=to_date_iso,
                    maxResults=50,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
            events = events_result.get("items", [])
            print(events)
            if not events:
                return []

            return [event["id"] for event in events]

        except Exception as e:
            print(f"Error fetching events: {str(e)}")
            return []

    async def reschedule_event(self, userChatQuery: str) -> str:
        from_date, to_date = self.extracted_date_from_query(userChatQuery)
        if not from_date or not to_date:
            return "Could not parse date from the query."
        events_ids = await self.get_event_ids(from_date, to_date)
        if not events_ids:
            return "Could not parse event ID from the query."

        # Extract new date and time from the query
        new_date_pattern = re.search(r"to (\d{4}-\d{2}-\d{2})", userChatQuery)
        new_time_pattern = re.search(r"at (\d{1,2}:\d{2})", userChatQuery)

        new_date = new_date_pattern.group(1) if new_date_pattern else None
        new_time = new_time_pattern.group(1) if new_time_pattern else None

        if not new_date and not new_time:
            return "Could not parse new date or time from the query."

        # Load credentials
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )
        service = build("calendar", "v3", credentials=creds)

        try:
            # Fetch existing event details to avoid overwriting other fields
            eventss = events_ids[0]
            event = (
                service.events()
                .get(calendarId="sachinbfrnd@gmail.com", eventId=eventss)
                .execute()
            )

            # Update event date and/or time
            if new_date:
                start_date = datetime.strptime(new_date, "%Y-%m-%d").isoformat()
                end_date = (
                    datetime.strptime(new_date, "%Y-%m-%d") + timedelta(hours=1)
                ).isoformat()
                event["start"]["dateTime"] = start_date
                event["end"]["dateTime"] = end_date

            if new_time:
                start_time = datetime.strptime(new_time, "%H:%M").time()
                end_time = (
                    datetime.combine(
                        datetime.strptime(
                            event["start"]["dateTime"], "%Y-%m-%dT%H:%M:%S"
                        ).date(),
                        start_time,
                    )
                    + timedelta(hours=1)
                ).time()
                event["start"]["dateTime"] = datetime.combine(
                    datetime.strptime(
                        event["start"]["dateTime"], "%Y-%m-%dT%H:%M:%S"
                    ).date(),
                    start_time,
                ).isoformat()
                event["end"]["dateTime"] = datetime.combine(
                    datetime.strptime(
                        event["end"]["dateTime"], "%Y-%m-%dT%H:%M:%S"
                    ).date(),
                    end_time,
                ).isoformat()

            updated_event = (
                service.events()
                .update(calendarId="sachinbfrnd@gmail.com", eventId=eventss, body=event)
                .execute()
            )

            return f"Event {eventss} rescheduled successfully to {event['start']['dateTime']} - {event['end']['dateTime']}"

        except Exception as e:
            return f"Error rescheduling event: {str(e)}"

    async def create_calendar_event_google(self, userchatquery: str) -> str:
        """Creates a Google Calendar event based on the user's query."""
        try:
            summary, start_time, end_time, description = extract_event_details(
                userchatquery
            )

            if not start_time or not end_time:
                return {
                    "status": "error",
                    "message": "Could not extract a valid date and time.",
                }

            creds = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=SCOPES
            )
            service = build("calendar", "v3", credentials=creds)

            event = {
                "summary": summary,
                "location": "Online",  # You can enhance this by extracting location
                "description": description,
                "start": {
                    "dateTime": start_time.isoformat(),
                    "timeZone": "Asia/Kolkata",  # Changed to Indian Standard Time
                },
                "end": {
                    "dateTime": end_time.isoformat(),
                    "timeZone": "Asia/Kolkata",  # Changed to Indian Standard Time
                },
                "reminders": {
                    "useDefault": True,
                },
            }
            print(event)

            event = (
                service.events()
                .insert(calendarId="sachinbfrnd@gmail.com", body=event)
                .execute()
            )

            # return {"status": "success", "event_link": event.get('htmlLink')}
            return "meeting created successfully"

        except Exception as e:

            return {"status": "error", "message": str(e)}

    async def get_multi_day_free_slots(self, days=5):
        """
        Returns all time slots across multiple days within working hours.
        Each slot is marked as either 'available' or 'booked'.
        """
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )
        service = build("calendar", "v3", credentials=creds)
        now = datetime.now(local_tz)
        free_slots = {}

        for i in range(days):
            date = now + timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")

            day_start = local_tz.localize(
                datetime.combine(date.date(), time(hour=WORKING_HOURS_START))
            )
            day_end = local_tz.localize(
                datetime.combine(date.date(), time(hour=WORKING_HOURS_END))
            )

            # Convert to UTC and format as ISO string with "Z"
            time_min = day_start.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            time_max = day_end.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            events_result = (
                service.events()
                .list(
                    calendarId="sachinbfrnd@gmail.com",
                    timeMin=time_min,
                    timeMax=time_max,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )

            events = []
            for event in events_result.get("items", []):
                start = event["start"].get("dateTime")
                end = event["end"].get("dateTime")
                if start and end:
                    events.append(
                        {
                            "start": isoparse(start)
                            .astimezone(local_tz)
                            .replace(tzinfo=None),
                            "end": isoparse(end)
                            .astimezone(local_tz)
                            .replace(tzinfo=None),
                        }
                    )

            slots = []
            current_time = day_start.replace(tzinfo=None)

            while current_time + timedelta(
                minutes=SLOT_DURATION_MINUTES
            ) <= day_end.replace(tzinfo=None):
                slot_end = current_time + timedelta(minutes=SLOT_DURATION_MINUTES)
                overlapping = any(
                    event["start"] < slot_end and event["end"] > current_time
                    for event in events
                )
                slots.append(
                    {
                        "start": current_time.strftime("%H:%M"),
                        "end": slot_end.strftime("%H:%M"),
                        "status": "booked" if overlapping else "available",
                    }
                )
                current_time += timedelta(minutes=SLOT_DURATION_MINUTES)

            free_slots[date_str] = slots

        return free_slots

    async def create_calendar_events_google(self, booking: dict) -> dict:
        """
        Creates a calendar event from the confirmed booking info.
        booking format:
        {
            "service": "Electrical work",
            "datetime": "2025-04-09T09:00:00",
            "name": "Jayaram",
            "phone": "9698132646"
        }
        """
        try:
            # Get calendar service
            creds = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=SCOPES
            )
            service = build("calendar", "v3", credentials=creds)

            # Parse start and end time
            start_dt = datetime.fromisoformat(booking["datetime"])
            end_dt = start_dt + timedelta(hours=1)  # 1-hour default duration
            start_dt = local_tz.localize(start_dt)
            end_dt = local_tz.localize(end_dt)

            # Event structure
            event = {
                "summary": f'{booking["service"]} - {booking["name"]}',
                "description": f'Service: {booking["service"]}\n'
                f'Customer: {booking["name"]}\n'
                f'Phone: {booking["phone"]}',
                "start": {
                    "dateTime": start_dt.isoformat(),
                    "timeZone": str(local_tz),
                },
                "end": {
                    "dateTime": end_dt.isoformat(),
                    "timeZone": str(local_tz),
                },
                "attendees": [],  # Optional
            }

            # Insert the event
            created_event = (
                service.events()
                .insert(calendarId="sachinbfrnd@gmail.com", body=event)
                .execute()
            )

            return {
                "success": True,
                "event_link": created_event.get("htmlLink"),
                "message": f"✅ Appointment booked successfully for {booking['name']} on {start_dt.strftime('%Y-%m-%d %H:%M')}!",
            }

        except Exception as e:
            return {"success": False, "message": f"❌ Failed to create event: {str(e)}"}


def extract_event_details(userchatquery):
    """Extracts meeting details (summary, date, time, description) from user query."""

    summary = "Meeting"
    description = "No description provided."
    start_time, end_time = None, None

    # Extract date
    match = re.search(r"on (\d{4}-\d{2}-\d{2})", userchatquery)
    if match:
        date_str = match.group(1)
        start_time = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=9, minute=0)
        end_time = start_time + timedelta(hours=1)  # Default 1-hour duration

    if "tomorrow" in userchatquery:
        start_time = datetime.now() + timedelta(days=1)
        start_time = start_time.replace(hour=9, minute=0)
        end_time = start_time + timedelta(hours=1)

    # Extract start time
    time_match = re.search(r"at (\d{1,2}:\d{2})", userchatquery)
    if time_match:
        time_parts = time_match.group(1).split(":")
        if start_time:
            start_time = start_time.replace(
                hour=int(time_parts[0]), minute=int(time_parts[1])
            )
        else:
            start_time = datetime.now().replace(
                hour=int(time_parts[0]), minute=int(time_parts[1])
            )

        end_time = start_time + timedelta(
            hours=1
        )  # Default duration if no end time provided

    # Extract end time explicitly
    end_time_match = re.search(r"to (\d{1,2}:\d{2})", userchatquery)
    if end_time_match:
        end_time_parts = end_time_match.group(1).split(":")
        if start_time:  # Ensure we have a start time before setting end time
            end_time = start_time.replace(
                hour=int(end_time_parts[0]), minute=int(end_time_parts[1])
            )

    # Extract description
    desc_match = re.search(r"about (.+)", userchatquery)
    if desc_match:
        description = desc_match.group(1)

    # Extract meeting summary
    title_match = re.search(r"for (.+)", userchatquery)
    if title_match:
        summary = title_match.group(1)

    return summary, start_time, end_time, description
