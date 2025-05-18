"""
RAG検索を行うファイル - rag_search.py
このスクリプトはNeo4jに保存された埋め込みとグラフデータを使用して
ハイブリッドRAG（Graph RAGとベクトル検索）を実行します。
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars, SearchType
from langchain_community.graphs import Neo4jGraph
from pydantic import BaseModel, Field
from typing import List
import json
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()


class Entities(BaseModel):
    """エンティティに関する情報の識別"""
    names: List[str] = Field(
        ...,
        description="テキストに登場する重要な概念、人物、場所、組織、出来事、関係性、事実など、あらゆる種類の重要なエンティティや情報"
    )


class RAGSearch:
    def __init__(self):
        """初期化とインスタンスの設定"""
        # 設定ファイルを読み込む
        try:
            with open("embedding_config.json", "r") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                "embedding_config.json が見つかりません。先に embedding_batch.py を実行してください。")

        # LLMの初期化
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o")

        # Neo4jグラフへの接続
        self.graph = Neo4jGraph()

        # ベクトルインデックスの初期化
        self.vector_index = Neo4jVector.from_existing_graph(
            OpenAIEmbeddings(model=self.config["embedding_model"]),
            search_type=SearchType.HYBRID,
            node_label=self.config["node_label"],
            text_node_properties=self.config["text_properties"],
            embedding_node_property=self.config["embedding_property"]
        )

        # エンティティ抽出チェーンの初期化
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "テキストからあらゆる種類の重要な情報とエンティティを抽出します。これには人物、場所、組織、概念、関係性、事実などが含まれます。",
            ),
            (
                "human",
                "指定された形式を使用して、以下から重要な情報をすべて抽出します。"
                "input: {question}",
            )
        ])
        self.entity_chain = prompt | self.llm.with_structured_output(Entities)

        # RAGチェーンの構築
        self._build_chain()

    def _generate_full_text_query(self, input: str) -> str:
        """全文検索クエリ生成のヘルパー関数"""
        full_text_query = ""
        words = [el for el in remove_lucene_chars(input).split() if el]
        if not words:
            return ""
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()

    def _structured_retriever(self, question: str) -> str:
        """グラフ検索用のretriever関数"""
        result = ""
        entities = self.entity_chain.invoke({"question": question})
        for entity in entities.names:
            response = self.graph.query(
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
                {"query": self._generate_full_text_query(entity)},
            )
            result += "\n".join([el['output'] for el in response])
        return result

    def _retriever(self, input_obj):
        """ハイブリッド検索（グラフ検索とベクトル検索の組み合わせ）"""
        if isinstance(input_obj, dict) and "question" in input_obj:
            question = input_obj["question"]
        elif isinstance(input_obj, str):
            question = input_obj
        else:
            raise TypeError("Input must be a string or a dict with a 'question' key")

        print(f"検索クエリ: {question}")
        structured_data = self._structured_retriever(question)
        unstructured_data = [el.page_content for el in self.vector_index.similarity_search(question)]
        final_data = f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ".join(unstructured_data)}
"""
        return final_data

    def _build_chain(self):
        """RAGパイプラインの構築"""
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
        self.chain = (
                RunnableParallel(
                    {
                        "context": _search_query | self._retriever,
                        "question": RunnablePassthrough(),
                    }
                )
                | prompt
                | self.llm
                | StrOutputParser()
        )

    def ask(self, question: str) -> str:
        """質問に対する応答を取得"""
        return self.chain.invoke({"question": question})


if __name__ == "__main__":
    # RAG検索インスタンスの作成
    rag = RAGSearch()

    # 対話型インターフェース
    print(rag.ask("磯野カツオと一番仲の良い友達は誰ですか？"))
    print(rag.ask("タラちゃんのお母さんは誰ですか？"))
    print(rag.ask("カツオとタラちゃんの続柄は何ですか？"))
