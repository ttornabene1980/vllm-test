from langchain_openai import ChatOpenAI

def llm_create():
    llm = ChatOpenAI(
        model="deepseek-coder-6.7b-instruct",
        base_url="http://192.168.1.98:8000/v1",
        verbose=True,
       temperature=0
    )
    return llm

def llm_create_openai():
    return ChatOpenAI( model="gpt-4", temperature=0)

from langchain.callbacks.base import BaseCallbackHandler

# 1️⃣ Capture reasoning in a variable
class ReasoningLogger(BaseCallbackHandler):
    def __init__(self):
        self.logs = []

    def getLogs(self):
        return self.logs
    
    def on_agent_action(self, action, **kwargs):
        # Capture the Thought/Action/Action Input
        d = {"tool": action.tool, "tool_input": action.tool_input, "log": action.log}
        # print("Agent action:", d)
        self.logs.append(d)

    def on_agent_finish(self, finish, **kwargs):
        self.logs.append({"final_answer": finish.return_values})
