import os
import asyncio
from dotenv import load_dotenv
from jinja2 import Template

from typing import Any, List, Annotated, Sequence, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage
from langchain_core.prompts.chat import ChatPromptTemplate

from database.retriever import Retriever
from .base_agent import BaseAgent
from .prompts import *


class RagAgentState(TypedDict):
    input: str
    search_queries: List[str]
    documents: List[str]
    is_search_sufficient: bool
    number_of_search: int
    title: str
    final_answer: str

class RagAgentConfig(TypedDict):
    llm: Any
    retriever: Any
    max_search_limit: int


def format_document(document):
    return f"""DOC_ID : {document['metadata']['document_id']} / HEADING_ID : {document['metadata']['heading_id']}
    {document['content'].strip()}"""


async def search_database(state: RagAgentState, config):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents.
    """
    retriever = config['configurable'].get("retriever")

    async def _one(q: str):
        # 동기 리트리버를 스레드로 우회해 비블로킹
        res = await asyncio.to_thread(retriever.retrieve_small_to_big, q)
        return [format_document(x) for x in res]

    tasks = [asyncio.create_task(_one(q)) for q in state.get('search_queries', [])]
    batches = await asyncio.gather(*tasks) if tasks else []
    documents = [doc for batch in batches for doc in batch]
    return {
        "documents" : documents,
        "number_of_search" : state.get('number_of_search', 0) + 1
        }


class GenerateQueryFormat(BaseModel):
    search_queries: List[str] = Field(description="Possible queries for searching vectorstore database.")

async def generate_queries(state: RagAgentState, config):
    """
    Returns:
        state (dict): Generate available queries from 
    """
    llm = config['configurable'].get("llm")
    system_prompt = Template(GENERATE_QUERY_SYSTEM).render(
        search_queries=" / ".join(x for x in state.get('search_queries', [])),
        documents='\n\n'.join(x for x in state.get('documents', []))
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", '{input}')]
    )
    chain = prompt | llm.with_structured_output(GenerateQueryFormat)
    response = await chain.ainvoke({'input' : state['input']})
    return {"search_queries" : response.search_queries}


class EvaluateSearchFormat(BaseModel):
    is_sufficient: bool = Field(description="Check if search result is sufficient for answering user's question.")

async def evaluate_search_results(state: RagAgentState, config):
    llm = config['configurable'].get("llm")
    user_prompt = Template(EVALUATE_SEARCH_RESULT_USER).render(
        input=state['input'],
        documents=state['documents']
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", EVALUATE_SEARCH_RESULT_SYSTEM), ("human", user_prompt)]
    )
    chain = prompt | llm.with_structured_output(EvaluateSearchFormat)
    response = await chain.ainvoke({"input": state['input']})
    return {"is_search_sufficient": response.is_sufficient}


def is_search_results_sufficient(state: RagAgentState, config) -> bool:
    search_limit = config['configurable'].get("max_search_limit", 3)
    if state["number_of_search"] >= search_limit:
        return True
    return state['is_search_sufficient']


class RAGAnswerFormat(BaseModel):
    title: str = Field(description="Title for this conversation.")
    answer: str = Field(description="Answer for the user's question.")

async def generate_rag_answer(state: RagAgentState, config):
    llm = config['configurable'].get("llm")
    user_prompt = Template(GENERATE_RAG_ANSWER_USER).render(
        input=state['input'],
        documents=state['documents']
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", GENERATE_RAG_ANSWER_SYSTEM), ("human", user_prompt)]
    )
    chain = prompt | llm.with_structured_output(RAGAnswerFormat)
    response = await chain.ainvoke({"input" : state['input'], "documents" : state['documents']})
    return {
        "title" : response.title,
        "final_answer": response.answer
    }


class RagAgent(BaseAgent):
    def __init__(self, llm, index_paths):
        super().__init__()
        self.llm = llm
        self.retriever = Retriever(index_paths)

    def build_workflow(self):
        rag_workflow = StateGraph(RagAgentState, RagAgentConfig)
        rag_workflow.add_node("generate_queries", generate_queries)
        rag_workflow.add_node("search_database", search_database)
        rag_workflow.add_node("evaluate_search_results", evaluate_search_results)
        rag_workflow.add_node("generate_rag_answer", generate_rag_answer)

        rag_workflow.add_edge(START, "generate_queries")
        rag_workflow.add_edge("generate_queries", "search_database")
        rag_workflow.add_edge("search_database", "evaluate_search_results")
        rag_workflow.add_conditional_edges(
            "evaluate_search_results",
            is_search_results_sufficient,
            {
                True: "generate_rag_answer",
                False: "generate_queries"
            }
        )
        rag_workflow.add_edge("generate_rag_answer", END)
        self.workflow = rag_workflow

    async def async_run(self, user_input):
        if self.workflow is None:
            self.build_workflow()
        agent = self.workflow.compile()
        config = {
            "configurable" : {
                "llm" : self.llm,
                "retriever" : self.retriever,
                "max_search_limit" : 3
            }
        }
        return await agent.ainvoke({"input": user_input}, config=config)