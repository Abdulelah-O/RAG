import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# 1. List your PDF files
pdf_paths = [
    "SAMA_rulebook_dataset/اللائحة التنفيذية لنظام الايجار التمويلي.pdf",
    "SAMA_rulebook_dataset/اللائحة التنفيذية لنظام التمويل العقاري.pdf",
    "SAMA_rulebook_dataset/اللائحة التنفيذية لنظام المدفوعات وخدماتها.pdf",
    "SAMA_rulebook_dataset/اللائحة التنفيذية لنظام المعلومات الائتمانية.pdf",
    "SAMA_rulebook_dataset/اللائحة التنفيذية لنظام مكافحة جرائم الإرهاب وتمويله.pdf",
    "SAMA_rulebook_dataset/اللائحة التنفيذية لنظام مكافحة غسل الأموال.pdf",
    "SAMA_rulebook_dataset/قواعد تطبيق أحكام نظام مراقبة البنوك.pdf",
    "SAMA_rulebook_dataset/نظام الايجار التمويلي.pdf",
    "SAMA_rulebook_dataset/نظام البنك المركزي السعودي.pdf",
    "SAMA_rulebook_dataset/نظام التمويل العقاري.pdf",
    "SAMA_rulebook_dataset/نظام المدفوعات وخدماتها.pdf",
    "SAMA_rulebook_dataset/نظام المعلومات الائتمانية.pdf",
    "SAMA_rulebook_dataset/نظام مراقبة البنوك.pdf",
    "SAMA_rulebook_dataset/نظام مراقبة شركات التمويل.pdf",
    "SAMA_rulebook_dataset/نظام معالجة المنشآت المالية المهمّة.pdf",
    "SAMA_rulebook_dataset/نظام مكافحة غسل الأموال.pdf",
    "SAMA_rulebook_dataset/نظام مكافحة جرائم الإرهاب وتمويله.pdf",
    "SAMA_rulebook_dataset/نظام النقد العربي السعودي.pdf",
    "SAMA_rulebook_dataset/اللائحة التنفيذية لنظام مراقبة شركات التمويل.pdf",
    "SAMA_rulebook_dataset/لائحة تنظيم المقاصة النهائية وترتيبات الضمان المرتبطة بها.pdf"
]
# 2. Load all PDFs using PyPDFLoader


all_docs = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    all_docs.extend(loader.load())

# 3. Split into chunks


text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, chunk_overlap=50
)
splits = text_splitter.split_documents(all_docs)

# 4. Embed and index with Chroma


vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# 5. Build the RAG (retrieval + generation) chain


llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)

# 6. Ask a question and show answer and sources
query = "ما هي وحدة النقد العربي السعودي؟"
result = rag_chain(query)

print("Answer:\n", result['result'])
print("\nSources:")
for i, doc in enumerate(result['source_documents']):
    print(f"\nSource {i+1} (from {doc.metadata.get('source','unknown file')}):\n", doc.page_content[:400])
