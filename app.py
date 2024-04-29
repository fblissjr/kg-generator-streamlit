import streamlit as st
from typing import Any, Callable, Optional, TypeVar, Union, cast, overload, List
from typing_extensions import TypedDict
import os
from openai import OpenAI
from pydantic import BaseModel, Field
from graphviz import Digraph
import instructor

from braintrust import init_logger, traced
import ast

# Braintrust
# braintrust.login(api_key=os.environ.get("BRAINTRUST_API_KEY"))


# Initialize braintrust logger
# logger = init_logger(project="KG")


# native braintrust proxy, todo later
# client = wrap_openai(
#    OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "Your OPENAI_API_KEY here"))
# )

client = OpenAI(
    # base_url="https://braintrustproxy.com/v1", # only if using braintrust or other logging
    api_key=os.environ[
        "OPENAI_API_KEY"
    ],  # Can use Braintrust, Anthropic, etc. API keys here
)

# Patch the client with instructor
client = instructor.from_openai(client, mode=instructor.Mode.TOOLS)


class Node(BaseModel):
    id: int
    label: str
    color: str


class Edge(BaseModel):
    source: int
    target: int
    label: str
    color: str = "black"


class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(description="Nodes in the knowledge graph")
    edges: List[Edge] = Field(description="Edges in the knowledge graph")


class MessageDict(TypedDict):
    role: str
    content: str


def add_chunk_to_ai_message(chunk: str):
    st.session_state.messages.append({"role": "assistant", "content": chunk})


def display_knowledge_graph():
    if len(st.session_state.messages) > 0:
        if st.session_state.messages[-1]["role"] != "user":
            obj = st.session_state.messages[-1]["content"]
            if f"{obj}" != "":
                obj = ast.literal_eval(f"{obj}")
                dot = Digraph(comment="Knowledge Graph")
                if obj["nodes"] not in [None, []]:
                    if obj["nodes"][0]["label"] not in [None, ""]:
                        for i, node in enumerate(obj["nodes"]):
                            if obj["nodes"][i]["label"] not in [None, ""]:
                                dot.node(
                                    name=str(obj["nodes"][i]["id"]),
                                    label=obj["nodes"][i]["label"],
                                    color=obj["nodes"][i]["color"],
                                )
                if obj["edges"] not in [None, []]:
                    if obj["edges"][0]["label"] not in [None, ""]:
                        for i, edge in enumerate(obj["edges"]):
                            if (
                                obj["edges"][i]["source"] not in [None, ""]
                                and obj["edges"][i]["target"] not in [None, ""]
                                and obj["edges"][i]["label"] not in [None, ""]
                            ):
                                dot.edge(
                                    tail_name=str(obj["edges"][i]["source"]),
                                    head_name=str(obj["edges"][i]["target"]),
                                    label=obj["edges"][i]["label"],
                                    color=obj["edges"][i]["color"],
                                )
                st.graphviz_chart(dot)


def send_message():
    st.markdown("Generating graph, please wait.")
    st.session_state.messages.append(
        {"role": "user", "content": st.session_state.text_block}
    )


def response(message):
    extraction_stream = client.chat.completions.create_partial(
        model="gpt-3.5-turbo",
        response_model=KnowledgeGraph,
        messages=[
            {
                "role": "user",
                "content": f"Help me understand the following by describing it as small knowledge graph: {message}. It is important to add variety of colors in the nodes.",
            },
        ],
        temperature=0,
        stream=True,
    )
    for extraction in extraction_stream:
        obj = extraction.model_dump()
        if f"{obj}" != st.session_state.aux:
            add_chunk_to_ai_message(f"{obj}")
            st.session_state.aux = f"{obj}"


def main():
    st.set_page_config(page_title="Knowledge Graph Generator")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "aux" not in st.session_state:
        st.session_state.aux = ""

    st.title("Knowledge Graph Generator")
    st.markdown(
        "Input text description, out comes relationships of nodes and edges in a graph"
    )

    text_block = st.text_area(
        "Enter text:",
        value="Image of a toy car with red and yellow colors and big eyes. Designed for toddlers.",
        key="text_block",
        height=20,
    )
    st.button("Generate Knowledge Graph", on_click=send_message)

    if st.session_state.messages:
        if st.session_state.messages[-1]["role"] == "user":
            response(st.session_state.messages[-1]["content"])

    display_knowledge_graph()


if __name__ == "__main__":
    main()
