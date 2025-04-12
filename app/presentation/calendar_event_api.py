from fastapi import FastAPI, Query, APIRouter, HTTPException
from datetime import datetime, timedelta
from pydantic import BaseModel
from typing import Optional
from pymongo import MongoClient

app = FastAPI()
calendarRouter = APIRouter()

# MongoDB setup
client = MongoClient("mongodb://localhost:27017")
db = client["calendar_system"]
slots = db["slots"]  # Single collection for all slot types (free, booked, personal)


# Models
class Event(BaseModel):
    date: str
    start_time: str
    end_time: str
    status: str  # free | booked | personal
    appointment_type: Optional[str] = (
        None  # Type of appointment (general, repair, etc.)
    )
    customer_name: Optional[str] = None  # Name of the customer (if booked)
    customer_phone: Optional[str] = None  # Phone number of the customer (if booked)
    notes: Optional[str] = None  # Any additional notes for the appointment


# Helpers
def generate_custom_daily_slots(date: str):
    return [
        {
            "date": date,
            "start_time": "09:00",
            "end_time": "12:00",
            "status": "free",
            "appointment_type": None,
            "customer_name": None,
            "customer_phone": None,
            "notes": None,
        },
        {
            "date": date,
            "start_time": "13:00",
            "end_time": "17:00",
            "status": "free",
            "appointment_type": None,
            "customer_name": None,
            "customer_phone": None,
            "notes": None,
        },
    ]


def update_slot_status(
    date,
    start_time,
    end_time,
    status,
    appointment_type=None,
    customer_name=None,
    customer_phone=None,
    notes=None,
):
    result = slots.update_many(
        {
            "date": date,
            "start_time": {"$gte": start_time},
            "end_time": {"$lte": end_time},
            "status": "free",
        },
        {
            "$set": {
                "status": status,
                "appointment_type": appointment_type,
                "customer_name": customer_name,
                "customer_phone": customer_phone,
                "notes": notes,
            }
        },
    )
    if result.matched_count == 0:
        raise HTTPException(
            status_code=404, detail="No free slots found for the given time range."
        )


def is_valid_slot_time(start: str, end: str) -> bool:
    allowed_slots = [
        ("09:00", "12:00"),
        ("13:00", "17:00"),
    ]
    for slot_start, slot_end in allowed_slots:
        if start == slot_start and end == slot_end:
            return True
    return False


# Routes
@calendarRouter.post("/generate-slots/")
def generate_slots_for_days(
    days: int = Query(7, description="Number of working days to generate")
):
    today = datetime.today()
    generated_days = 0
    current_day_offset = 0

    while generated_days < days:
        date_obj = today + timedelta(days=current_day_offset)
        weekday = date_obj.weekday()

        if weekday < 5:  # Skip weekends
            date = date_obj.strftime("%Y-%m-%d")
            existing = list(slots.find({"date": date}))
            if not existing:
                new_slots = generate_custom_daily_slots(date)
                slots.insert_many(new_slots)
                generated_days += 1
        current_day_offset += 1

    return {
        "message": f"Generated free slots for {days} working days (excluding weekends)."
    }


@calendarRouter.post("/book-slot/")
def book_slot(event: Event):
    # Check if the day is a weekend (Saturday = 5, Sunday = 6)
    booking_date = datetime.strptime(event.date, "%Y-%m-%d")
    if booking_date.weekday() >= 5:
        return {"message": "Bookings are not allowed on weekends."}

    # Validate if the given time range is allowed
    if not is_valid_slot_time(event.start_time, event.end_time):
        return {
            "message": "Invalid time range. Allowed slots are 09:00–12:00 and 13:00–17:00."
        }

    # Update the free slot
    update_slot_status(
        event.date,
        event.start_time,
        event.end_time,
        status="booked",
        appointment_type=event.appointment_type,
        customer_name=event.customer_name,
        customer_phone=event.customer_phone,
        notes=event.notes,
    )

    return {"message": "Slot booked and free slot updated."}


@calendarRouter.post("/add-personal-slot/")
def add_personal_slot(event: Event):

    # Update the free slot status and type to personal
    update_slot_status(
        event.date,
        event.start_time,
        event.end_time,
        status="personal",
        appointment_type=event.appointment_type,
        customer_name=event.customer_name,
        customer_phone=event.customer_phone,
        notes=event.notes,
    )

    return {"message": "Personal slot added and free slot updated."}


@calendarRouter.get("/free-slots/")
def get_free_slots():
    # Fetch all slots where the status is 'free'
    return list(slots.find({"status": "free"}, {"_id": 0}))


@calendarRouter.get("/booked-slots/")
def get_booked_slots():
    # Fetch all slots where the status is 'booked'
    return list(slots.find({"status": "booked"}, {"_id": 0}))


@calendarRouter.get("/personal-slots/")
def get_personal_slots():
    # Fetch all slots where the status is 'personal'
    return list(slots.find({"status": "personal"}, {"_id": 0}))


# Attach the router to the main app
app.include_router(calendarRouter)
