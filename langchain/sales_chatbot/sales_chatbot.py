import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS


def initialize_sales_bot(vector_store_dir: str="phone_sales"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    global SALES_BOT    
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT

def initialize_chat_bot():
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    global CHAT_BOT
    CHAT_BOT = llm

    return CHAT_BOT

def chat(message, history=None):
    template = ChatPromptTemplate.from_messages([
    ("system", "You are an excellent mobile phone salesman, proficient in sales words. Below is a conversation between you and a customer. Please continue the conversation.\n chat history: {history}"),
    ("human", "{user_input}"),
])

    # 生成提示
    messages = template.format_messages(
        user_input=message,
        history=history
    )
    print(f"[chat_messages]{messages}")
    result = CHAT_BOT.invoke(messages)
    print(f"[result]{result}")
    return result.content


def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"]:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    elif enable_chat:
        return chat(message, history)
    else:
        return "这个问题我要问问领导"
    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="房产销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    # 初始化房产销售机器人
    initialize_sales_bot()
    initialize_chat_bot()
    # 启动 Gradio 服务
    launch_gradio()
