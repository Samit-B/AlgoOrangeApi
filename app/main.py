from fastapi import FastAPI
from app.presentation.chat_api import chatRouter
from app.presentation.project_api import projectApiRouter
from app.presentation.browser_plugin_api import browserPluginApiRouter
from app.presentation.program_api import programApiRouter
from app.presentation.task_api import taskApiRouter
from app.application.agents.pdf_agent import pdfRouter
from app.presentation.excel_api import excelRouter
from app.presentation.calendar_event_api import calendarRouter
from app.presentation.chromadb_view_api import cromaRouter
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
app.include_router(chatRouter, prefix="/chat", tags=["ChatAPI"])
app.include_router(
    browserPluginApiRouter, prefix="/browserPlugin", tags=["BrowserPluginAPI"]
)
app.include_router(projectApiRouter, prefix="/project", tags=["ProjectAPI"])
app.include_router(programApiRouter, prefix="/program", tags=["ProgramAPI"])
app.include_router(taskApiRouter, prefix="/task", tags=["TaskAPI"])
app.include_router(pdfRouter, prefix="/upload", tags=["UploadAPI"])
app.include_router(excelRouter, prefix="/excel", tags=["ExcelAPI"])
app.include_router(
    calendarRouter, prefix="/calendar", tags=["CalendarAPI"]
)  # Include the calendar router
app.include_router(
    cromaRouter, prefix="/chromadb", tags=["ChromaDBAPI"]
)  # Include the ChromaDB router


@app.get("/")
def read_root():
    return {"message": "Welcome to the Student CRUD ddd API"}
