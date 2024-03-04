from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.agents import load_tools, initialize_agent

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from huggingface_hub import snapshot_download

# LLMの定義
model_id = "TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ"
download_path = snapshot_download(repo_id=model_id)

tokenizer = AutoTokenizer.from_pretrained(download_path)
model = AutoModelForCausalLM.from_pretrained(download_path)

pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=pipe)

#==============================================================================
# 入力に質問を受け取り、出力に質問に対する回答を返す関数
def get_answer_with_search(question: str) -> str:

    # 質問のテンプレート
    template = """<s>[INST] <<SYS>>
    あなたはweb検索などを用いて事実が確認されたことだけを答えるとても正直なエージェントです。
    <</SYS>>

    質問：{question} [/INST]"""

    prompt_template = ChatPromptTemplate.from_template(template)
    
    tool_names = ["serpapi"]

    tools = load_tools(tool_names)

    agent = initialize_agent(tools, llm, agent="self-ask-with-search")

    output = agent.run(prompt_template.format(question=question))

    return output

#==============================================================================


#==============================================================================
# 入力にコンテキストと質問を受け取り，その質問に対する回答を返す関数
def get_answer_with_retrieval(context: str, question: str) -> str:

    # 埋め込みモデルの初期化
    embeddings = HuggingFaceEmbeddings(model_name="oshizo/sbert-jsnli-luke-japanese-base-lite")

    vectorstore = FAISS.from_texts(
        [context],  embedding=embeddings
    )

    retriever = vectorstore.as_retriever()

    # プロンプトテンプレートの作成
    prompt_template_qa = """あなたは親切で優しいアシスタントです。丁寧に、日本語でお答えください！
    もし以下の情報が探している情報に関連していない場合は、そのトピックに関する自身の知識を用いて質問
    に答えてください。

    {context}

    質問: {question}
    回答（日本語）:
    """
    prompt_qa = PromptTemplate(
        template=prompt_template_qa, 
        input_variables=["context", "question"]
    )

    prompt = ChatPromptTemplate.from_template(prompt_qa)

    # チェーンの生成
    chain = (
        {"context":retriever, "question":prompt}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.run(question)
#==============================================================================

#==============================================================================
# 数学が得意なエージェントによる回答生成
def get_answer_with_math(question: str) -> str:

    tool_names = ["serpapi", "llm_math"]
    # 数学が得意なエージェントの初期化
    agent = initialize_agent(
        agent="zero-shot-react-description",
        llm=llm,
        tools=tools,
        verbose=True
    )

    # 数学が得意なエージェントによる回答生成
    return agent.run(question)
#==============================================================================