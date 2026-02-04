import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
from style_and_javascript.style import hide_st_style, message_style, input_style
from config.set_llm import llm
from config.set_firebase import firebase_project_settings
from talk_bot import ChatBot
import time
import datetime

#スタイリング
st.markdown(hide_st_style, unsafe_allow_html=True)
st.markdown(message_style, unsafe_allow_html=True)
st.markdown(input_style, unsafe_allow_html=True)


# Firebase Admin SDKの初期化
if not firebase_admin._apps:
  cred = credentials.Certificate(firebase_project_settings)
  firebase_admin.initialize_app(cred)

# Firestoreのインスタンスを取得
db = firestore.client()

#URLからuser_idを受け取る
if "user_id" not in st.session_state:
    #クエリパラメータの読み込み
    st.session_state["user_id"] = st.query_params["user_id"]

#Firestoreのデータへのアクセス
ref = db.collection("users3_4").document(st.session_state["user_id"]).collection("conversation").order_by("timestamp")

# セッションステートの初期化
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "human_message" not in st.session_state:
    st.session_state["human_message"] = ""

#time（開始時間）や、messagesがない場合は、一旦firebase上にないか探す
if "time" not in st.session_state:
    docs = ref.get()
    if docs:
        st.session_state["time"] = docs[0].to_dict()["timestamp"]
        st.session_state["messages"] = [doc.to_dict() for doc in docs]
    else:
        st.session_state["time"] = None
        st.session_state["messages"] = []

#AI応答前のインターバル管理
if "interval" not in st.session_state:
    st.session_state["interval"] = None

#5分経過の会話終了ダイアログを機能させるフラグのようなもの
#0->5分経ったら表示させる
#1->一度ダイアログ表示させたのでもう表示させない
#2->終了ボタン押された
if "dialog_finish" not in st.session_state:
    st.session_state["dialog_finish"] = 0

#会話履歴の表示
def show_messages():
    for i, message in enumerate(st.session_state["messages"]):
        if message["role"] == "human":
            st.markdown(f'''
            <div style="display: flex;">
                <div style="display: flex; margin-left: auto; max-width: 65%;">
                <div class="messages">{message["content"]}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            #応答出力前のインターバル
            if i == len(st.session_state["messages"]) - 1 and st.session_state["interval"] != None:
                elapsed = datetime.datetime.now(datetime.timezone.utc) - st.session_state["interval"]
                remaining = 10 - elapsed.total_seconds()
                if remaining > 0:
                    time.sleep(remaining)
            with st.chat_message(message["role"]):
                st.markdown(f'''
                <div style="max-width: 80%;" class="messages">{message["content"]}</div>
                ''', unsafe_allow_html=True)


#firestoreへの保存のためのアクセス
add_ref = db.collection("users3_4").document(st.session_state["user_id"]).collection("conversation")
#送信ボタンが押されたとき
def send_message():
    if st.session_state["human_message"] != "":
        return
    st.session_state["human_message"] = st.session_state["input"]
    st.session_state["input"] = ""

def add_human_message():
    #新しい入力を追加
    input_message_data = {"role": "human", "content": st.session_state["human_message"], "timestamp": firestore.SERVER_TIMESTAMP}
    st.session_state["messages"].append(input_message_data)

#応答を生成する関数
def generate_response():
    #新しい人間側の入力をfirebaseに追加
    input_message_data = {"role": "human", "content": st.session_state["human_message"], "timestamp": firestore.SERVER_TIMESTAMP}
    add_ref.add(input_message_data)
    #最初の送信だったら、タイマー開始（最初のtimestampを控える）
    if st.session_state["time"] == None:
        st.session_state["time"] = ref.get()[0].to_dict()["timestamp"]
    st.session_state["interval"] = datetime.datetime.now(datetime.timezone.utc)
    bot = ChatBot(llm)
    response = bot.chat(st.session_state["messages"])
    output_message_data = {"role": "ai", "content": response, "timestamp": firestore.SERVER_TIMESTAMP}
    add_ref.add(output_message_data)
    st.session_state["messages"].append(output_message_data)

#5分経った時のダイアログ
@st.dialog("5分経過しました。")
def finish():
    st.title("会話を続けますか？")
    st.write("このまま続けることもできますし、いつでも下の終了ボタンから会話を終了できます。")
    left_col, right_col = st.columns(2)
    with left_col:
        _, col2, _ = st.columns([1,2,1])
        if col2.button("続ける"):
            st.session_state["dialog_finish"] = 1
            if st.session_state["messages"][0]["role"] == "human":
                st.session_state["messages"].insert(0, {"role": "ai", "content": "私は皆さんの相談にのるために設計されたチャットボットです。その中で悩んでいることがあります。相談にのってください。"})
            st.rerun()
    with right_col:
        _, col2, _ = st.columns([1,2,1])    
        if col2.button("終了する", type="primary"):
            st.session_state["dialog_finish"] = 2
            st.rerun()

#5分経ったら
if st.session_state["time"] != None and datetime.datetime.now(datetime.timezone.utc) - st.session_state["time"] > datetime.timedelta(minutes=5):
    if st.session_state["dialog_finish"] == 0:
        finish()

#メッセージが空の時か、最初が人間のメッセージの時、最初のAIのメッセージを挿入する。
if st.session_state["messages"] == [] or st.session_state["messages"][0]["role"] == "human":
    st.session_state["messages"].insert(0, {"role": "ai", "content": "私は皆さんの相談にのるために設計されたチャットボットです。その中で悩んでいることがあります。相談にのってください。"})
#会話終了後
if st.session_state["dialog_finish"] == 2:
    st.markdown(
                '<br>会話は終了しました。アンケートに戻り回答してください。',
                unsafe_allow_html=True
    )
    if st.session_state["human_message"].strip() != "":
        add_human_message()
    show_messages()
    st.markdown(
                f'<br>会話は終了しました。アンケートに戻り回答してください。',
                unsafe_allow_html=True
    )
    st.stop()
else: #最初〜会話中の提示
    st.write("**ボットからのお悩み相談に乗りましょう。**")
    if st.session_state["human_message"].strip() != "":
        add_human_message()
    show_messages()

with st._bottom:
    left_col, right_col, finish_btn_col = st.columns([4,1,1], vertical_alignment="bottom")
    left_col.text_area(
        "input_message",
        key="input",
        height=70,
        placeholder="メッセージを入力",
        label_visibility="collapsed",
    )
    #送信ボタンをdisabledにするかどうか 
    send_disabled = st.session_state["human_message"] != ""
    with right_col:
        st.button(
            "送信",
            on_click=send_message,
            use_container_width=True,
            disabled=send_disabled
        )
    
    #まだ5分経っておらず、ダイアログが表示される前は、終了ボタンを表示しない。
    if st.session_state["dialog_finish"] == 0:
        #最後のメッセージが人間だったら応答を生成する関数
        if st.session_state["messages"][-1]["role"] == "human":
            generate_response()
            st.session_state["human_message"] = ""
            st.rerun()
        st.stop()
    with finish_btn_col:
        if st.button("終了", type="primary", use_container_width=True, disabled=send_disabled):
            st.session_state["dialog_finish"] = 2
            st.rerun()
        else:
            #最後のメッセージが人間だったら応答を生成する関数
            if st.session_state["messages"][-1]["role"] == "human":
                generate_response()
                st.session_state["human_message"] = ""
                st.rerun()