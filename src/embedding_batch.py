"""
埋め込みを行うバッチファイル - embedding_batch.py
このスクリプトはWikipediaからデータを取得し、Neo4jデータベースに
グラフデータと埋め込みベクトルを格納します。
"""

from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import SearchType
from langchain_community.graphs import Neo4jGraph
import json
from dotenv import load_dotenv


def create_embeddings():
    """Wikipediaからデータを取得し、Neo4jに埋め込みとグラフを格納する"""
    print("埋め込み処理を開始します...")

    # 環境変数の読み込み
    load_dotenv()

    # Wikipediaからデータをロード
    print("Wikipediaからデータをロード中...")
    raw_documents = WikipediaLoader(query="サザエさん", lang="ja").load()

    # テキストをチャンクに分割
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    documents = text_splitter.split_documents(raw_documents[:3])
    print(f"{len(documents)}個のドキュメントに分割しました")

    # グラフ変換用のLLMを初期化
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    llm_transformer = LLMGraphTransformer(llm=llm)

    # ドキュメントをグラフ形式に変換
    print("ドキュメントをグラフ形式に変換中...")
    graph_documents = llm_transformer.convert_to_graph_documents(documents)

    # Neo4jグラフへの接続
    print("Neo4jグラフに接続中...")
    graph = Neo4jGraph()

    # グラフドキュメントを追加
    print("グラフドキュメントをNeo4jに追加中...")
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )

    # 全文検索インデックスの作成
    print("全文検索インデックスを作成中...")
    graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

    # 埋め込みを作成してNeo4jベクターインデックスに格納
    print("埋め込みを作成してNeo4jベクターインデックスに格納中...")
    vector_index = Neo4jVector.from_existing_graph(
        OpenAIEmbeddings(model="text-embedding-ada-002"),
        search_type=SearchType.HYBRID,
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )

    # 設定情報を保存
    config = {
        "embedding_model": "text-embedding-ada-002",
        "node_label": "Document",
        "text_properties": ["text"],
        "embedding_property": "embedding",
        "search_type": "HYBRID"
    }

    with open("embedding_config.json", "w") as f:
        json.dump(config, f, indent=4)

    print("埋め込み処理が完了しました。設定は embedding_config.json に保存されました。")


if __name__ == "__main__":
    create_embeddings()