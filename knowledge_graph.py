from typing import List, Dict, Tuple, Optional
from openai import OpenAI
from pydantic import BaseModel, Field


class KnowledgeGraphNode(BaseModel):
    id: int
    label: str
    color: Optional[str] = "blue"


class KnowledgeGraphEdge(BaseModel):
    source: int
    target: int
    label: str
    color: Optional[str] = "black"


class KnowledgeGraph(BaseModel):
    nodes: List[KnowledgeGraphNode] = []
    edges: List[KnowledgeGraphEdge] = []
    next_node_id: int = 1

    def add_node(self, node: KnowledgeGraphNode):
        node.id = self.next_node_id
        self.nodes.append(node)
        self.next_node_id += 1

    def add_edge(self, edge: KnowledgeGraphEdge):
        self.edges.append(edge)

    def generate_from_text(
        self, text: str, openai_client: OpenAI
    ) -> Tuple[List[KnowledgeGraphNode], List[KnowledgeGraphEdge]]:
        extraction_stream = openai_client.chat.completions.create_partial(
            model="gpt-3.5-turbo",
            response_model=KnowledgeGraph,
            messages=[
                {
                    "role": "user",
                    "content": f"Help me understand the following by describing it as a small knowledge graph: {text}. It is important to add a variety of colors in the nodes.",
                },
            ],
            temperature=0,
            stream=True,
        )

        knowledge_graph_dict = None
        for extraction in extraction_stream:
            obj = extraction.model_dump()
            if obj is not None:
                knowledge_graph_dict = obj

        if knowledge_graph_dict is not None:
            knowledge_graph = KnowledgeGraph(
                nodes=[
                    KnowledgeGraphNode(
                        id=node_dict["id"],
                        label=node_dict["label"],
                        color=node_dict.get("color", "blue"),
                    )
                    for node_dict in knowledge_graph_dict.get("nodes", [])
                ],
                edges=[
                    KnowledgeGraphEdge(
                        source=edge_dict["source"],
                        target=edge_dict["target"],
                        label=edge_dict["label"],
                        color=edge_dict.get("color", "black"),
                    )
                    for edge_dict in knowledge_graph_dict.get("edges", [])
                ],
            )
            return knowledge_graph.nodes, knowledge_graph.edges
        else:
            return [], []
