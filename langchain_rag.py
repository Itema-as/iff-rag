import bs4
from getpass import getpass
from langchain import hub
from langchain.document_loaders import WebBaseLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI

# Prompt the user for the OpenAI API key
OPENAI_API_KEY = getpass("Enter API key for OpenAI: ")

# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    },
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002", api_key=OPENAI_API_KEY
)
vector_store = FAISS.from_documents(all_splits, embedding_model)

# Define prompt for question-answering using LangChain Hub
prompt = hub.pull("rlm/rag-prompt")


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    messages = prompt.invoke({
        "question": state["question"],
        "context": docs_content
    })

    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile the application and test it
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

response = graph.invoke({"question": "What is TALM?"})
print(response["answer"])
