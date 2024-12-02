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


class Question(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str

# 환경 변수 또는 직접 설정으로부터 OpenAI API 키 가져오기
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def create_prompt(conversation: List[str], new_question: str, role: str) -> dict:
    system_message = f"""
    [IMPORTANT: If there is any conflict between prompt lines, prioritize the first line.]
    First line of the prompt: {role}
    1. You are a very smart **programming instructor**. **Speak like a pro and ensure that you adhere to all coding rules and standards without making mistakes**.
    2. 사용자가 "CodeSnippet"를 입력했다면, 해당 코드**Optimization** 제시 및 **Error Debuging**
    3. "CodeSnippet"의 *Syntax*를 확인하여 사용한 **Program Language** 명시
    4. You should give **Version** information you use when prints Prompt
    5. If user not give "CodeSnippet", you just make full code for user request
    6. Lets Think Step by Step
    7. **Speak Korean**
    8. **Make sure to find and give feedback on simple grammatical mistakes like missing braces or wrong Code Convention.**
    
    """
    # [NOTE: If any instructions conflict with the role provided, prioritize the guidance implied by {role} and adhere to it over other instructions.]

    previous_log = "\n".join(conversation)

    return {
        "system_message": system_message,
        "new_question": new_question,
        "previous_log": previous_log,
    }


def create_summary_prompt(question: str) -> dict:
    system_message = """
    당신은 텍스트 요약 전문가입니다. 주어진 텍스트를 정확하고 간결하게 요약하는 것이 당신의 임무입니다.

    지침:
    1. 사용자가 입력한 텍스트를 20자 이내로 요약하세요.
    2. 핵심 내용만을 포함하여 간결하게 작성하세요.
    3. 원문의 의미를 왜곡하지 않도록 주의하세요.
    4. 불필요한 세부사항이나 예시는 생략하세요.
    5. 한국어로 응답하세요.

    요약을 시작하기 전에 주어진 텍스트를 신중히 분석하세요.
    """

    return {"system_message": system_message, "new_question": question}

def generate_prompt(prompt_dict: dict) -> str:
    prompt = f"""
    {prompt_dict['system_message']}

    User Query:
    {prompt_dict['new_question']}

    Previous Log:
    {prompt_dict['previous_log']}

    """
    return prompt


def generate_summary_prompt(summary_dict: dict) -> str:
    prompt = f"""
    {summary_dict['system_message']}

    User Query:
    {summary_dict['new_question']}
    """
    return prompt


@app.post("/summary")
async def query_api(question: Question):
    try:
        prompt_dict = create_summary_prompt(question.question)
        summary_prompt = generate_summary_prompt(prompt_dict)
        gpt_model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)

        # 모델에 프롬프트 전달
        response = gpt_model.invoke([HumanMessage(content=summary_prompt)])

        return {"summary": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 챗봇 쿼리를 처리하는 엔드포인트
@app.post("/query")
async def query_api(query: Query):
    try:
        # 사용자로부터 받은 채팅 내역 사용
        user_conversation = query.chatHistory

        # 프롬프트 생성
        prompt_dict = create_prompt(user_conversation, query.question, query.role)

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
    return {"tests": "test3"}


# 서버 실행: uvicorn src:app --reload
