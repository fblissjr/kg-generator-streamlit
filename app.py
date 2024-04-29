import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from typing import Any, Callable, Optional, TypeVar, Union, cast, overload, List
from typing_extensions import TypedDict
import os
from openai import OpenAI
from pydantic import BaseModel, Field
from graphviz import Digraph
import instructor
import ast
from PIL import Image
import io
from knowledge_graph import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge

# Braintrust
# braintrust.login(api_key=os.environ.get("BRAINTRUST_API_KEY"))

# Initialize braintrust logger
# logger = init_logger(project="KG")

# native braintrust proxy, todo later
# client = wrap_openai(
#    OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "Your OPENAI_API_KEY here"))
# )

# color_scheme_config = apply_color_scheme("dark")

# Define color schemes
COLOR_SCHEMES = {
    "dark": {
        "bg_color": "#1f1f1f",
        "node_color": "#f0f0f0",
        "edge_color": "#8c8c8c",
        "text_color": "#f0f0f0",
    },
    "light": {
        "bg_color": "#f0f0f0",
        "node_color": "#1f1f1f",
        "edge_color": "#4c4c4c",
        "text_color": "#1f1f1f",
    },
}


# Function to apply the selected color scheme
def apply_color_scheme(color_scheme):

    # Apply the selected color scheme to the Streamlit theme
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {COLOR_SCHEMES[color_scheme]['bg_color']};
            color: {COLOR_SCHEMES[color_scheme]['text_color']};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Configure the Config instance with the selected color scheme
    config = Config(
        width=1024,
        height=800,
        directed=True,
        physics=True,
        hierarchical=True,
        node_color=COLOR_SCHEMES[color_scheme]["node_color"],
        edge_color=COLOR_SCHEMES[color_scheme]["edge_color"],
    )

    return config


client = OpenAI(
    # base_url="https://braintrustproxy.com/v1", # only if using braintrust or other logging
    api_key=os.environ[
        "OPENAI_API_KEY"
    ],  # Can use Braintrust, Anthropic, etc. API keys here
)

# Patch the client with instructor
client = instructor.from_openai(client, mode=instructor.Mode.TOOLS)


class MessageDict(TypedDict):
    role: str
    content: str


def add_chunk_to_ai_message(chunk: str):
    st.session_state.messages.append({"role": "assistant", "content": chunk})


def display_knowledge_graph(
    nodes: List[KnowledgeGraphNode], edges: List[KnowledgeGraphEdge]
):
    streamlit_nodes = []
    streamlit_edges = []

    for node in nodes:
        streamlit_node = Node(id=node.id, label=node.label, color=node.color)
        streamlit_nodes.append(streamlit_node)

    for edge in edges:
        streamlit_edge = Edge(
            source=edge.source,
            target=edge.target,
            label=edge.label,
            color=edge.color,
        )
        streamlit_edges.append(streamlit_edge)

    config = Config(
        width=800,
        height=800,
        directed=True,
        physics=False,
        hierarchical=True,
    )
    return_value = agraph(nodes=streamlit_nodes, edges=streamlit_edges, config=config)

    st.write(return_value)

    # Export to PNG
    if st.button("Export as PNG"):
        img_data = return_value.screenshot(driver="selenium", wait_time=1)
        buffered = io.BytesIO()
        img = Image.open(io.BytesIO(img_data))
        img.save(buffered, format="PNG")
        st.download_button(
            label="Download PNG",
            data=buffered.getvalue(),
            file_name="knowledge_graph.png",
            mime="image/png",
        )


def send_message():
    st.markdown("Generating graph, please wait.")
    st.session_state.messages.append(
        {"role": "user", "content": st.session_state.text_block}
    )


def response(message):
    knowledge_graph = KnowledgeGraph()
    nodes, edges = knowledge_graph.generate_from_text(message, client)

    display_knowledge_graph(nodes, edges)


def main():
    st.set_page_config(
        page_title="Knowledge Graph Generator",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

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


if __name__ == "__main__":
    main()
