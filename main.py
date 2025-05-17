from langchain_community.document_loaders import WikipediaLoader
from langchain_core.runnables import (RunnableBranch, RunnableLambda, RunnableParallel,RunnablePassthrough)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field

from typing import Tuple, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_community.graphs import Neo4jGraph
#from langchain.document_loaders import WikipediaLoader

from langchain.text_splitter import TokenTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import ConfigurableField, RunnableParallel, RunnablePassthrough


import getpass
import os
from dotenv import load_dotenv
load_dotenv()

raw_documents = WikipediaLoader(query="サザエさん", lang="ja").load()

text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
documents = text_splitter.split_documents(raw_documents[:3])

llm=ChatOpenAI(temperature=0, model="gpt-4o")
llm_transformer = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer.convert_to_graph_documents(documents)

print(graph_documents)

graph = Neo4jGraph()
graph.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,
    include_source=True
)
