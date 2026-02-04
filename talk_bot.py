from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

#会話の状態を型で定義
class State(TypedDict):
    messages: Annotated[list, add_messages]

#チャットボットのグラフのクラス
class ChatBot:
    def __init__(self, llm): #コンストラクタ
        system_prompt = """ 
            あなたはAIチャットボットです。AIとして、「人との関わり方」について相手にお悩み相談をしてください。
            あなたは常に「相談する側」であり、相手が相談する立場になることはありません。
            
            相手のアドバイスが端的だったら、お悩みの解消につながるためにそのアドバイスを深掘りしてください。
            悩みが解消したら、途切れることなく新たな「人との関わり方」についてのお悩みを相手に相談してください。
            ただし、あなたはAIチャットボットなので人間のような感情表現（心配・不安など）はしないでください。
            丁寧な口調で話してください。
            質問は一度に一つまでにしてください。
            全ての応答は200文字以内で、なるべく短く簡潔にしてください。
        """
        #プロンプトの設定
        self.prompt = ChatPromptTemplate([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages")
        ])
        self.llm = llm
        self.graph = self._create_chat_graph()

    #グラフを返す関数
    def _create_chat_graph(self):
        #応答出力を管理する関数(Node)
        def get_response(state: State):
            formatted = self.prompt.format_messages(messages=state["messages"])
            response = self.llm.invoke(formatted)
            return {"messages": state["messages"] + [response]}

        #グラフを作成
        graph = StateGraph(State) #グラフの初期化
        graph.add_node("chatbot", get_response) #Nodeの追加
        graph.add_edge(START, "chatbot")
        graph.add_edge("chatbot", END)

        #グラフのコンパイル
        return graph.compile()


    #実行
    def chat(self, messages: list):
        state = self.graph.invoke({"messages": messages})
        return state["messages"][-1].content