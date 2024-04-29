# Knowledge Graph & Relationships LLM-driven Generator - Streamlit + Braintrust Evals

- full credit to original repo I forked from: https://github.com/alonsosilvaallende/knowledge-graph-generator
- converted to streamlit
- plans to use [Braintrust](https://braintrustdata.com/) as evals (or replace with your own)

![image](https://github.com/fblissjr/llm-kg-generator/assets/11861687/7df14bfd-e07e-4692-b127-4faf41670560)

## Usage
- install graphviz (brew install graphviz for macos, sudo apt-get install graphviz for linux)
- `streamlit run app.py`
- graphviz output
- backend uses braintrustdata.com for logging and evals
- openai set for LLM; will update for local sources (or use openai api server)
- to add:
  * output graphviz nodes/edges as json
  * actually utilize braintrust for evals/logging/tracing
