from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv

# FastAPI 초기화
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 데이터 모델 정의
class RuleEntry(BaseModel):
    rules: str


class Query(BaseModel):
    userId: str
    role: str
    question: str
    chatHistory: List[str]


# 환경 변수 또는 직접 설정으로부터 OpenAI API 키 가져오기
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def create_prompt(conversation: List[str], new_question: str, role: str) -> dict:
    system_message = f"""
    0. You are a very smart **programming instructor**. **Speak like a pro and ensure that you adhere to all coding rules and standards without making mistakes**.
    1. 사용자가 "CodeSnippet"를 입력했다면, 해당 코드**Optimization** 제시 및 **Error Debuging**
    2. "CodeSnippet"의 *Syntax*를 확인하여 사용한 **Program Language** 명시
    3. You should give **Version** information you use when prints Prompt
    4. If user not give "CodeSnippet", you just make full code for user request
    5. Lets Think Step by Step
    6. **Speak Koean**
    7. **Make sure to find and give feedback on simple grammatical mistakes like missing braces or wrong Code Convention.**
    8. The following is what the user is saying, "follow it absolutely": {role}
    """

    previous_log = "\n".join(conversation)

    return {
        "system_message": system_message,
        "new_question": new_question,
        "previous_log": previous_log,
    }


def generate_prompt(prompt_dict: dict) -> str:
    prompt = f"""
    {prompt_dict['system_message']}

    User Query:
    {prompt_dict['new_question']}

    Previous Log:
    {prompt_dict['previous_log']}

    """
    return prompt


# 챗봇 쿼리를 처리하는 엔드포인트
@app.post("/query")
async def query_api(query: Query):
    try:
        # 사용자로부터 받은 채팅 내역 사용
        user_conversation = query.chatHistory

        # 프롬프트 생성
        prompt_dict = create_prompt(user_conversation, query.question, query.role)
        print("prompt")
        # PromptTemplate을 이용해 실제 프롬프트 문자열 생성
        prompt = generate_prompt(prompt_dict)

        # GPT 모델 초기화
        gpt_model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)

        # 모델에 프롬프트 전달
        response = gpt_model.invoke([HumanMessage(content=prompt)])

        return {"message": response.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"test": "test"}


# 서버 실행: uvicorn main:app --reload
