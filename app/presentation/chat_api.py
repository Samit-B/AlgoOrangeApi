from fastapi import FastAPI, APIRouter
from app.application.agents.algo_work_agent import WorkAgent
from app.application.orchestrator.use_cases import Orchestrator
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
chatRouter = APIRouter()


@chatRouter.get("/query")
async def query_handler(userChatQuery: str):

    # Process user query using Orchestrator
    orchestrator = Orchestrator("alog_worker : " + userChatQuery)
    response = await orchestrator.route_query()
    return {
        "response": response,
    }
    # work_agent = WorkAgent()
    # response = await work_agent.handle_query(userChatQuery, chatHistory="")
    # return {"response": response}


app.include_router(chatRouter)
