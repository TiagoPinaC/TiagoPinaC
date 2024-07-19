import shutil
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import Chroma
import os
from dotenv import load_dotenv, dotenv_values
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.chains import RetrievalQA, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from langchain_huggingface import ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
import torch





class ChatBot():
  

    chunk_size = 1000
    chunk_overlap = 150

    def __init__(self):

        chunk_size = 1000
        chunk_overlap = 150

        self.device = 0 if torch.cuda.is_available() else -1 
        print(f"!!!!!!!!!!!DEVICE: {self.device}")

        load_dotenv()
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if not os.environ.get("LANGCHAIN_API_KEY"):
            os.environ["LANGCHAIN_API_KEY"] = getpass()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size= self.chunk_size,
            chunk_overlap= self.chunk_overlap
        )

        self.loader = PyPDFLoader("your_file.pdf")
        self.docs = []
        self.docs.extend(self.loader.load())

        self.splits = self.text_splitter.split_documents(self.docs)

        self.embeddings = HuggingFaceEmbeddings()

        self.persist_directory = 'docs/chroma'

        # Ensure vectordb is created or loaded
        self.vectordb = self.create_or_load_vectordb()

        print(f"Length DB : {len(self.vectordb.get()['documents'])}")

        # Set up the language model with Endpoint API
        #self.llm = HuggingFaceEndpoint(
        #    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        #    task="text-generation",
        #    temperature=0.1,
        #    top_k=50,
        #    huggingfacehub_api_token=os.getenv('HUGGING_FACE_ACCESS_TOKEN')
        #)


        self.llm = HuggingFacePipeline.from_model_id(  
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            task="text-generation",
            device = self.device,
            pipeline_kwargs={
                "max_new_tokens": 512,
                "top_k": 50,
                "temperature": 0.1,
            },
        )

        self.chat_model = ChatHuggingFace(llm=self.llm)

        # Define system prompt
        self.system_prompt = """
        This is a chat between a user and an artificial intelligence assistant.
        The assistant gives helpful, detailed, and polite answers to the user's questions based on the context.
        The assistant should also indicate when the answer cannot be found in the context.

        {context}
        """

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", "{input}"),
            ]
        )

        self.retriever = self.vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 6})

        self.question_answer_chain = create_stuff_documents_chain(self.chat_model, self.prompt)
        self.qa_chain = create_retrieval_chain(self.retriever, self.question_answer_chain)

    def create_or_load_vectordb(self):
        #return Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        DELETE = False
        if DELETE and os.path.exists(self.persist_directory):
            print("Deleting old db...")
            shutil.rmtree(self.persist_directory)
        if os.path.exists(self.persist_directory):  # If the vector store already exists, simply load it from memory
            print("LOADING DB FROM MEMORY!")
            return Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        else:  # If not, create it from the documents
            print("CREATING DB FROM SCRATCH!!")
            return Chroma.from_documents(
                documents=self.splits,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        




if __name__ == "__main__":

    bot = ChatBot()
    input_str = input("Ask me anything: ")
    while input_str != "END":
        result = bot.qa_chain.invoke({"input": input_str})
        print(result['answer'])
        input_str = input()