import dotenv
import os
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, Settings
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.node_parser import SentenceSplitter

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")

llm = OpenAI(model="gpt-4o", temperature=0)

embed_model = OpenAIEmbedding(
    model="text-embedding-ada-002")

Settings.llm = llm
Settings.embed_model = embed_model

# Create some sample documents.
documents = SimpleWebPageReader(html_to_text=True).load_data(
    ["https://lilianweng.github.io/posts/2023-06-23-agent"]
)

parser = SentenceSplitter(chunk_size=100, chunk_overlap=20)
nodes = parser.get_nodes_from_documents(documents)

index = VectorStoreIndex(nodes)
retriever = index.as_retriever()

query = "What is TALM?"
nodes = retriever.retrieve(query)

# Extract text from the retrieved nodes.
context = "\n\n".join([node.get_text() for node in nodes])

# Build a prompt that includes the retrieved context and the original question.
question = query
prompt = f"Using the following context, please answer the question: '{
    question}'\n\nContext:\n{context}"

# Feed the prompt to the LLM and print the generated answer.
response_text = llm.complete(prompt=prompt)
print("LLM Response:\n", response_text)
