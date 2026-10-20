from typing import Annotated,TypedDict
from langgraph.graph import add_messages

class MyDate(TypedDict):
    messages : Annotated[list,add_messages] #Reducer 자동으로 리스트에 메시지를 어펜드해줌,

from langchain_core.messages import AIMessage,HumanMessage

msg1 = [HumanMessage(content='안녕',id ='1')]
msg2 = [AIMessage(content='방가',id ='2')]

results = add_messages(msg1,msg2)
print(results)
# [HumanMessage(content='안녕',
#                additional_kwargs={}, response_metadata={}, id='1'), 
#  AIMessage(content='방가', additional_kwargs={}, 
#            response_metadata={}, id='2', tool_calls=[], invalid_tool_calls=[])]
#add메세지 기능으로 휴먼메세지랑,ai메세지가 한 리스트에 저장됨
#############################################################
msg3 = [HumanMessage(content='안녕',id ='3')]
msg4 = [AIMessage(content='방가',id ='3')]

results2 = add_messages(msg3,msg4)
print(results2)

# [AIMessage(content='방가', additional_kwargs={},
# response_metadata={}, id='3', tool_calls=[], invalid_tool_calls=[])]
#id가 똑같으면 마지막 메세지가 덮어써버림
  