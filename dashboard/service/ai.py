


from langchain_openai import ChatOpenAI


def llm_create():
    llm = ChatOpenAI(
        model="deepseek-coder-1.3b-instruct",
        openai_api_base="http://192.168.1.98:8000/v1",
        openai_api_key="EMPTY",  # vllm ignores this
        verbose=True,
        # tools=tools,
    )
    return llm



# 1️⃣ Capture reasoning in a variable
class ReasoningLogger(BaseCallbackHandler):
    def __init__(self):
        self.logs = []

    def on_agent_action(self, action, **kwargs):
        # Capture the Thought/Action/Action Input
        d = {"tool": action.tool, "tool_input": action.tool_input, "log": action.log}
        # print("Agent action:", d)
        self.logs.append(d)

    def on_agent_finish(self, finish, **kwargs):
        self.logs.append({"final_answer": finish.return_values})
