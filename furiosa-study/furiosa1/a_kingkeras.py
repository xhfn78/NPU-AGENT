from langchain_openai import ChatOpenAI

def get_llm(
    temperature=1,
    max_tokens =1000,
    top_p=0.5,
        
):

    llm = ChatOpenAI(
        base_url="http://127.0.0.1:8000/v1", 
        api_key='EMPTY',
        temperature=temperature,
        max_tokens =max_tokens,
        top_p=top_p,
    )
    return llm

