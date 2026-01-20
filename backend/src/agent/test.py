import os
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent  # 当前 .py 所在目录
env_path = project_root / "config" / ".env"

if env_path.exists() and env_path.is_file():
    load_dotenv(env_path, override=True)
    print("成功加载 .env 文件:", env_path)
else:
    raise FileNotFoundError(f"没找到 .env 文件: {env_path}")

from typing import TypedDict, List, Annotated
from datetime import datetime
from langgraph.checkpoint.memory import InMemorySaver
from backend.src.agent.contextbuilder.MyContextBuilder import MyContextBuilder
from backend.src.agent.models.LLM_MODEL import ModelInstances
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, add_messages
from hello_agents.context import ContextBuilder, ContextConfig
from hello_agents.core.message import Message
from hello_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry  # 假设这些是 helloagents 的模块
from hello_agents.tools import MemoryTool, RAGTool


# 定义 LangGraph 的 State（兼容 helloagents 的 Message）
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] # 用 dict 存 Message 数据：{"content": str, "role": str, "timestamp": datetime}
    user_query: str  # 当前用户查询
    thread_id: str  # 用于 Checkpointer 隔离会话


# 消息转换函数：LangGraph state["messages"] (dict) <-> helloagents Message
def langchain_to_hello_message(lc_msg: BaseMessage) -> Message:
    """
    将 LangChain 的消息转换为 helloagents 的 Message
    """
    # 角色映射：LangChain 的 role 是 type-based，helloagents 用字符串 "user"/"assistant"/"system"
    if isinstance(lc_msg, HumanMessage):
        role = "user"
    elif isinstance(lc_msg, AIMessage):
        role = "assistant"
    elif isinstance(lc_msg, SystemMessage):
        role = "system"
    else:
        # 其他类型（如 ToolMessage、FunctionMessage）可根据需要映射或忽略
        role = "assistant"  # 默认 fallback，或根据实际需求调整

    return Message(
        content=lc_msg.content,
        role=role,
        timestamp=lc_msg.additional_kwargs.get("timestamp") or datetime.now()
        # 如果 LangChain 消息里没有 timestamp，可以用当前时间，或从 metadata 取
    )


# def message_to_dict(m: Message) -> dict:
#     return {"content": m.content, "role": m.role, "timestamp": m.timestamp}


# 你的自定义类
class MyContextAgent:
    def __init__(self, knowledge_base_path="./knowledge_base",
                 user_id="default_user"):
        # 初始化 LLM
        self.llm = ModelInstances.leader_llm

        # 初始化 helloagents 工具和 ContextBuilder（同你的示例）
        self.memory_tool = MemoryTool(user_id=user_id)
        self.rag_tool = RAGTool(knowledge_base_path=knowledge_base_path)
        self.config = ContextConfig(
            max_tokens=3000,
            reserve_ratio=0.2,
            min_relevance=0.2,
            enable_compression=True
        )
        self.builder = MyContextBuilder(
            memory_tool=self.memory_tool,
            rag_tool=self.rag_tool,
            config=self.config
        )

        # Checkpointer
        self.checkpointer = InMemorySaver()

        # 构建 LangGraph
        self.graph = self._build_graph()

    def _build_graph(self) :
        builder = StateGraph(AgentState)

        # 唯一节点：构建上下文 + 生成响应
        def build_context_and_generate(state: AgentState) -> AgentState:
            # 从 state 取历史
            conversation_history=[]
            # 排除最后一条（当前用户查询）
            for msg in state["messages"][:-1]:
                conversation_history.append(langchain_to_hello_message(msg))

            # 用 helloagents Builder 构建上下文
            context = self.builder.build(
                user_query=state["user_query"],
                conversation_history=conversation_history,
                system_instructions="你是一位资深的AI助手。你的回答需要:1) 提供具体可行的建议 2) 解释技术原理 3) 给出代码示例"
                # 可自定义
            )

            print("=" * 80)
            print("构建的上下文:")
            print("=" * 80)
            print(context)
            print("=" * 80)

            # 用构建的 context 作为 prompt 调用 LLM
            response = self.llm.invoke([SystemMessage(content=context),HumanMessage(content=state["user_query"])])

            # 更新 messages：加AI 响应
            return {"messages": [response]}  # 返回 patch 更新 state

        builder.add_node("generate", build_context_and_generate)
        builder.add_edge(START, "generate")
        builder.add_edge("generate", END)

        return builder.compile(checkpointer=self.checkpointer)  # 启用 Checkpointer

    def run(self, user_query: str, thread_id: str = "user456") -> str:
        # 运行 graph，传入初始 state，并用 thread_id 加载/保存历史

        user_input=HumanMessage(content=user_query)
        result = self.graph.invoke({"messages":[user_input],"user_query":user_query,"thread_id":thread_id}, config={"configurable": {"thread_id": thread_id}})

        # 从最终 state 取最新响应
        last_message = result["messages"][-1] if result["messages"] else {"content": "No response"}
        return last_message.content


# 使用示例
if __name__ == "__main__":
    agent = MyContextAgent(user_id="user456")

    # 第一轮
    response1 = agent.run("如何优化Pandas的内存占用?", thread_id="user456")
    print("Response 1:", response1)

    # 第二轮（会加载上轮 messages）
    response2 = agent.run("那如何在代码中实现?", thread_id="user456")
    print("Response 2:", response2)