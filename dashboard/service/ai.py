from langchain_openai import ChatOpenAI

LLM_CONFIGS = [
    {
        "name": "DeepSeek Coder 6.7B",
        "model": "deepseek-coder-6.7b-instruct",
        "base_url": "http://192.168.1.98:8000/v1",
        "temperature": 0,
        "verbose": True
    },
    {
        "name": "DeepSeek R1 Qwen3-8B",
        "model": "DeepSeek-R1-0528-Qwen3-8B",
        "base_url": "http://10.199.145.180:8080/v1",
        "temperature": 0,
        "verbose": True
    },
    {
        "name": "OpenAI GPT-4o-mini",
        "model": "gpt-4o-mini",
        "base_url": None,
        "temperature": 0,
        "verbose": False
    },
    {
        "name": "OpenAI GPT-4",
        "model": "gpt-4",
        "base_url": None,
        "temperature": 0,
        "verbose": False
    }
]

def create_llm(config):
    kwargs = {
        "model": config["model"],
        "temperature": config.get("temperature", 0),
        "verbose": config.get("verbose", False)
    }
    if config.get("base_url"):
        kwargs["base_url"] = config["base_url"]

    return ChatOpenAI(**kwargs)

def llm_create():
    llm = ChatOpenAI(
        model="deepseek-coder-6.7b-instruct",
        base_url="http://192.168.1.98:8000/v1",
        verbose=True,
       temperature=0
    )
    return llm

def llm_create_deepseekr1():
    llm = ChatOpenAI(
        model="DeepSeek-R1-0528-Qwen3-8B",
        base_url="http://10.199.145.180:8080/v1",
        verbose=True,
        temperature=0
    )
    return llm


def llm_create_openai_mini():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

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
