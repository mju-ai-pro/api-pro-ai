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
    question: str
    chatHistory: List[str]


# 환경 변수 또는 직접 설정으로부터 OpenAI API 키 가져오기
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def create_prompt(conversation: List[str], new_question: str) -> dict:
    system_message = f"""
    1. 사용자가 "CodeSnippet"을 입력했다면, 해당 코드**Optimization** 제시 및 **Error Debuging**
    2. "CodeSnippet"의 *Syntax*를 확인하여 사용한 **Program Language** 명시
    3. You should give **Version** information you use when prints Prompt
    4. If Error information is not in user_input then request which error user have faced when you feel vague from "CodeSnippet" or user_input
    5. If user not give "CodeSnippet", you just make full code for user request
    6. IF user give "CodeSnippet", you need to ask whether you should give full codes or give partial codes which you have modified by asking "네/아니오"
    7. If user not give "CodeSnippet" and **No mention** about **Program Language**, you have to ask **which type of program language user want to use**
    8. Lets Think Step by Step
    """

    previous_log = "\n".join(conversation)

    # new_question에서 "CodeSnippet"이 있는지 검사
    if "CodeSnippet" in new_question:
        code = new_question  # new_question을 코드 스니펫으로 가정
        return {
            "system_message": system_message,
            "new_question": new_question,
            "previous_log": previous_log,
            "response": f"""
                *프롬프트 생성기*

                문제 설명:
                {new_question}

                코드:
                {code}

                버전 정보:
                - *프로그래밍 언어*: [언어 버전 명시]
                - *라이브러리/패키지*: [라이브러리 및 패키지 정보]

            """,
        }

    # "CodeSnippet"이 없을 경우, 프로그래밍 언어를 물어봄
    else:
        return {
            "system_message": system_message,
            "new_question": new_question,
            "previous_log": previous_log,
            "response": """
                프로그래밍 언어: 사용을 원하시는 언어를 기입해서 사용해주세요.
            """,
        }


def generate_prompt(prompt_dict: dict) -> str:
    prompt_template = f"""
    {prompt_dict['system_message']}

    User Query:
    {prompt_dict['new_question']}

    Previous Log:
    {prompt_dict['previous_log']}

    Assistant Response:
    {prompt_dict['response']}
    """
    return prompt_template


# 챗봇 쿼리를 처리하는 엔드포인트
@app.post("/query")
async def query_api(query: Query):
    try:
        # 사용자로부터 받은 채팅 내역 사용
        user_conversation = query.chatHistory

        # 새로운 질문을 대화 기록에 추가
        user_conversation.append(f"user: {query.question}")

        # 프롬프트 생성
        prompt_dict = create_prompt(user_conversation, query.question)

        # PromptTemplate을 이용해 실제 프롬프트 문자열 생성
        prompt_template = generate_prompt(prompt_dict)
        prompt = prompt_template.format(**prompt_dict)

        # GPT 모델 초기화
        gpt_model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)

        # 모델에 프롬프트 전달
        response = gpt_model.invoke([HumanMessage(content=prompt)])

        # 모델의 응답을 대화 기록에 추가
        user_conversation.append(f"assistant: {response.content}")
        return {"message": response.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"test": "test"}


# 서버 실행: uvicorn main:app --reload
