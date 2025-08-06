import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()

# -- App Setup --
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -- Document Loading & Chunking (Semantic with Overlap) --
loader = DirectoryLoader(
    "SAMA_rulebook_dataset",    # Relative path to your dataset folder
    glob="*.pdf",               # Loads only PDF files
    loader_cls=PyPDFLoader,     # Use PyPDFLoader for PDF compatibility
    show_progress=True,
    use_multithreading=True
)
raw_docs = loader.load()

#print(len(raw_docs))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100,
    separators=["\n\n", "\n", ".", "؟", "!", " "] # The **separators** parameter in `RecursiveCharacterTextSplitter` controls **where**The separators parameter in RecursiveCharacterTextSplitter controls where the text will be split to create chunks.
)
docs = text_splitter.split_documents(raw_docs) # docs is a list of document chunks.

# -- Embedding Vector Store (ChromaDB) --

embedding = OpenAIEmbeddings(
    model = 'text-embedding-3-large',
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

vectordb = Chroma.from_documents(
    docs,
    embedding=embedding,
    collection_metadata={"hnsw:space": "cosine"}
)

# -- Prompt Engineering --
prompt_template = """
أنت مساعد خبير في أنظمة وقواعد البنك المركزي السعودي.
اقرأ المقتطفات التالية وافهم محتواها، ثم جاوب على السؤال بناءً على ما فهمته منها.
إذا كانت المعلومة غير واضحة أو غير موجودة، قل "لا أعرف بناءً على البيانات المتوفرة".
إذا كانت الإجابة مبنية على مقتطف معيّن، اذكر اسم الملف كمصدر إن أمكن.

المقتطفات:
{context}

السؤال:
{question}

الإجابة:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


retriever = vectordb.as_retriever()
llm = ChatOpenAI(model="gpt-4-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# -- FastAPI Endpoint --
@app.get("/", response_class=None)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=None)
async def ask_question(request: Request, question: str = Form(...)):
    if not question.strip():
        return templates.TemplateResponse("index.html", {"request": request, "result": None, "question": question})

    result = qa_chain({"query": question})
    print("\n\n=========== المقتطفات المسترجعة من ChromaDB ===========\n")
    for i, doc in enumerate(result["source_documents"]):
      print(f"[{i+1}] المصدر: {doc.metadata.get('source', 'Unknown')}")
      print(doc.page_content)
      print("\n-------------------------------------------\n")
    answer = result["result"]
    sources = []
    for doc in result["source_documents"]:
        snippet = doc.page_content[:400]
        src = doc.metadata.get("source", "Unknown")
        sources.append({"source": src, "content": snippet})

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": answer,
            "sources": sources,
            "question": question
        }
    )

if __name__ == "_main_":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print("🚀 Running on port", port)

    uvicorn.run("main:app", host="0.0.0.0",port=port)
