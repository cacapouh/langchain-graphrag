from langchain_community.document_loaders import WikipediaLoader
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from typing import List
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph

from langchain.text_splitter import TokenTextSplitter

from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars, SearchType
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


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

vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(model="text-embedding-ada-002"),
    search_type=SearchType.HYBRID,
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)

graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

class Entities(BaseModel):
    """エンティティに関する情報の識別"""
    names: List[str] = Field(
        ...,
        description="文章の中に登場する、人物、各人物の性格、各人物間の続柄、各人物が所属する組織、各人物の家族関係",
    )

# 指示文を含んだプロンプトテンプレート
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "テキストから家族と人物のエンティティを抽出します。",
        ),
        (
            "human",
            "指定された形式を使用して、以下から情報を抽出します。"
            "input: {question}",
        ),
    ]
)

# エンティティ抽出チェーン
entity_chain = prompt | llm.with_structured_output(Entities)

# 全文検索クエリ生成のヘルパー関数
def generate_full_text_query(input: str) -> str:
    """
    指定された入力文字列に対する全文検索クエリを生成します。
    この関数は、全文検索に適したクエリ文字列を構築します。
    入力文字列を単語に分割し、
    各単語に対する類似性のしきい値 (変更された最大 2 文字) を結合します。
    AND 演算子を使用してそれらを演算します。ユーザーの質問からエンティティをマッピングするのに役立ちます
    データベースの値と一致しており、多少のスペルミスは許容されます。
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

# グラフ検索用のretriever関数
def structured_retriever(question: str) -> str:
    """
    質問の中で言及されたエンティティの近傍を収集します。
    """
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
                WITH node
                MATCH (node)-[r:!MENTIONS]->(neighbor)
                RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                UNION ALL
                WITH node
                MATCH (node)<-[r:!MENTIONS]-(neighbor)
                RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result

# テスト: グラフ検索の実行
print(structured_retriever("カツオの両親は誰ですか？"))

# 最終的なretriever関数（グラフ検索とベクトル検索の組み合わせ）
def retriever(input_obj):
    if isinstance(input_obj, dict) and "question" in input_obj:
        question = input_obj["question"]
    elif isinstance(input_obj, str):
        question = input_obj
    else:
        raise TypeError("Input must be a string or a dict with a 'question' key")

    print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    final_data = f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ".join(unstructured_data)}
"""
    return final_data

# 最終応答を得るためのプロンプトテンプレート
template = """あなたは優秀なAIです。下記のコンテキストを利用してユーザーの質問に丁寧に答えてください。
必ず文脈からわかる情報のみを使用して回答を生成してください。
{context}
ユーザーの質問: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# ダミー関数（RunnablePassthroughのために必要）
_search_query = RunnablePassthrough()

# GraphRAGのパイプライン作成
chain = (
    RunnableParallel(
        {
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)

# 質問応答の実行例
print(chain.invoke({"question": "磯野カツオと一番仲の良い友達は誰ですか？"}))
print(chain.invoke({"question": "タラちゃんのお母さんは誰ですか？"}))
print(chain.invoke({"question": "カツオとタラちゃんの続柄は何ですか？"}))