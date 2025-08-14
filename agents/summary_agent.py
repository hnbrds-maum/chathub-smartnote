import os
from dotenv import load_dotenv
from jinja2 import Template

from typing import Any, List, Annotated, Sequence, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph
from langchain_core.prompts.chat import ChatPromptTemplate

from .base_agent import BaseAgent
from .prompts import *

BATCH_SIZE = 10


class SummaryAgentState(TypedDict):
    input: List[str]
    summary: str
    title: str

class SummaryAgentConfig(TypedDict):
    llm: Any


def _chunked(seq: List[str], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

async def batch_summarize(state, config):
    llm = config['configurable'].get("llm")
    prompt = ChatPromptTemplate.from_messages(
        [("system", SINGLE_SUMMARY_SYSTEM), ("human", "{input}")]
    )
    chain = prompt | llm

    summaries: List[str] = []
    for batch in _chunked(state["input"], BATCH_SIZE):
        results = await chain.abatch([{"input": x} for x in batch])
        for r in results:
            summaries.append(getattr(r, "content", str(r)))

    return {"input": summaries}


class SummaryFormat(BaseModel):
    title: str = Field(description="Title for this summary")
    summary: str = Field(description="A single paragraph summary for given documents")

async def merge_summarize(state, config):
    llm = config['configurable'].get("llm")
    prompt = ChatPromptTemplate.from_messages(
        [("system", MERGE_SUMMARY_SYSTEM), ("human", "{input}")]
    )
    chain = prompt | llm.with_structured_output(SummaryFormat)

    response = await chain.ainvoke({"input": "\n".join(x for x in state["input"])})
    return {
        "summary": response.summary,
        "title" : response.title
    }


class SummaryAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__()
        self.llm = llm

    def build_workflow(self):
        workflow = StateGraph(SummaryAgentState, SummaryAgentConfig)
        workflow.add_node("batch_summarize", batch_summarize)
        workflow.add_node("merge_summarize", merge_summarize)

        workflow.add_edge("batch_summarize", "merge_summarize")
        workflow.add_edge("merge_summarize", END)
        self.workflow = workflow

    
    async def async_run(self, user_input: List[str], single_document: bool = True):
        if self.workflow is None:
            self.build_workflow()

        if single_document:
            self.workflow.set_entry_point("batch_summarize")
        else:
            self.workflow.set_entry_point("merge_summarize")

        agent = self.workflow.compile()
        cfg = {"configurable": {"llm": self.llm}}

        out_state: SummaryAgentState = await agent.ainvoke({"input": user_input}, config=cfg)
        return out_state