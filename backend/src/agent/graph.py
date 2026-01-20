from pathlib import Path
from dotenv import load_dotenv

from backend.src.agent.format.schema import MemoryExtractionOutput
from backend.src.agent.models.LLM_MODEL import ModelInstances
from backend.src.agent.nodes.access_relevance import assess_relevance
from backend.src.agent.nodes.assess_content_quality import assess_content_quality
from backend.src.agent.nodes.extract_and_add_memory import build_memory_extraction_input
from backend.src.agent.nodes.generate_verification_report import generate_verification_report, finalize_answer
from backend.src.agent.nodes.optimize_summary import optimize_summary
from backend.src.agent.nodes.reflection import reflection, evaluate_research
from backend.src.agent.nodes.verify_facts import verify_facts
from backend.src.agent.prompts.memory_prompt import memory_extraction_prompt

project_root = Path(__file__).resolve().parent  # 当前 .py 所在目录
env_path = project_root / "config" / ".env"

if env_path.exists() and env_path.is_file():
    load_dotenv(env_path, override=True)
    print("成功加载 .env 文件:", env_path)
else:
    raise FileNotFoundError(f"没找到 .env 文件: {env_path}")


from datetime import datetime

from hello_agents import Message
from hello_agents.context import ContextConfig
from hello_agents.tools import MemoryTool, RAGTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph,START,END
from langgraph.types import Command

from backend.src.agent.config.configuration import Configuration
from backend.src.agent.contextbuilder.MyContextBuilder import MyContextBuilder
from backend.src.agent.nodes.generate_query import generate_query
from backend.src.agent.nodes.should_regenerate_queried import should_regenerate_queried
from backend.src.agent.nodes.wait_for_confimation import wait_for_user_confirmation
from backend.src.agent.nodes.web_research import web_research
from backend.src.agent.states.overallstate import OverallState

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


class MyDeepResearchAgent:
    def __init__(self, knowledge_base_path="./knowledge_base",
                 user_id="default_user"):

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


    def _build_graph(self):
        workflow = StateGraph(OverallState,context_schema=Configuration)
        # 节点1：生成问题对
        def generate_query_node(state:OverallState,config:RunnableConfig):
            # 1.先判断用户是否已确认了问题生成成功
            # last_message = state["messages"][-1] if state["messages"] else None
            # if last_message and last_message.content and "[查询已确认]" in last_message.content:
            #     # 从确认消息中提取查询
            #     content = last_message.content
            #     queries_part = content.split("[查询已确认]")[1].strip()
            #     confirmed_queries = [q.strip() for q in queries_part.split("|")]
            #     return Command(update={
            #         "search_query": confirmed_queries,
            #         "generated_queries": confirmed_queries,
            #         "awaiting_user_confirmation": False,
            #         "user_confirmation_received": True
            #     })
            # 2.用户未确认，继续生成消息
            conversation_history = []
            # 排除最后一条（当前用户查询）
            for msg in state["messages"][:-1]:
                conversation_history.append(langchain_to_hello_message(msg))

            # 用 helloagents Builder 构建上下文
            context = self.builder.build(
                user_query=state["messages"][-1].content,
                conversation_history=conversation_history,
            )

            return generate_query(state,config,context)
        workflow.add_node("generate_query_node",generate_query_node)
        # 节点2：等待用户确认
        workflow.add_node("wait_for_user_confirmation",wait_for_user_confirmation)
        # 节点3：web查询
        def web_research_node(state:OverallState,config:RunnableConfig):
            conversation_history = []
            # 排除最后一条（当前用户查询）
            for msg in state["messages"][:-1]:
                conversation_history.append(langchain_to_hello_message(msg))

            # 用 helloagents Builder 构建上下文
            context = self.builder.build(
                user_query=state["messages"][-1].content,
                conversation_history=conversation_history,
            )
            return web_research(state, config, context)
        workflow.add_node("web_research",web_research_node)
        # 节点4：rag查询
        # 节点5：reflection评估
        workflow.add_node("reflection",reflection)
        # 节点6：
        workflow.add_node("assess_content_quality", assess_content_quality)
        workflow.add_node("verify_facts", verify_facts)
        workflow.add_node("assess_relevance", assess_relevance)
        workflow.add_node("optimize_summary", optimize_summary)
        workflow.add_node("generate_verification_report", generate_verification_report)
        workflow.add_node("finalize_answer", finalize_answer)

        def extract_and_add_memory(state: OverallState):
            print("开始提取记忆...")
            dialogue_history =build_memory_extraction_input(state)

            prompt = memory_extraction_prompt.format(full_state_text=dialogue_history)

            # 用 structured LLM（推荐 with_structured_output）
            llm = ModelInstances.answer_model
            structured_llm = llm.with_structured_output(MemoryExtractionOutput)

            try:
                result = structured_llm.invoke(prompt)
                memories = result.memories
                print(f"提取到 {len(memories)} 条记忆")
            except Exception as e:
                print("记忆提取失败:", e)
                memories = []

            added_count = 0
            for mem in memories:
                try:
                    self.memory_tool.execute(
                        "add",
                        content=mem.content,
                        memory_type=mem.memory_type,
                        importance=mem.importance,
                    )
                    added_count += 1
                    print(f"添加一条{mem.memory_type}记忆成功：{mem.content}")
                except Exception as e:
                    print(f"添加记忆失败: {mem}", e)

            # 可选：把添加结果记录到 state
            return
        workflow.add_node("extract_and_add_memory",extract_and_add_memory)
        # 边
        workflow.set_entry_point("generate_query_node")
        workflow.add_edge("generate_query_node","wait_for_user_confirmation")
        workflow.add_conditional_edges(
            "wait_for_user_confirmation",
            should_regenerate_queried,
            {
                "generate_query_node":"generate_query_node",
                "web_research":"web_research"
            }
        )
        workflow.add_edge("web_research","reflection")
        workflow.add_conditional_edges(
            "reflection",
            evaluate_research,
            {
                "assess_content_quality":"assess_content_quality",
                "web_research":"web_research"
            }
        )
        # Quality enhancement pipeline
        workflow.add_edge("assess_content_quality", "verify_facts")
        workflow.add_edge("verify_facts", "assess_relevance")
        workflow.add_edge("assess_relevance", "optimize_summary")
        workflow.add_edge("optimize_summary", "generate_verification_report")
        workflow.add_edge("generate_verification_report", "finalize_answer")
        # Finalize the answer
        workflow.add_edge("finalize_answer", "extract_and_add_memory")
        workflow.add_edge("extract_and_add_memory",END)
        return workflow.compile(checkpointer=self.checkpointer)  # 启用 Checkpointer
    def run(self, user_query: str, thread_id: str ) -> str:
        # 运行 graph，传入初始 state，并用 thread_id 加载/保存历史

        user_input=HumanMessage(content=user_query)
        result = self.graph.invoke({"messages":[user_input]}, config={"configurable": {"thread_id": thread_id}})

        # 从最终 state 取最新响应
        last_message = result["messages"][-1] if result["messages"] else {"content": "No response"}
        return last_message.content
# 使用示例
if __name__ == "__main__":
    agent = MyDeepResearchAgent(user_id="zhengbohao")

    # 第一轮
    response1 = agent.run("我想要学习吉他，是个新手，我应该怎么做？", thread_id="zhengbohao")
    print("Response 1:", response1)

    # # 第二轮（会加载上轮 messages）
    # response2 = agent.run("我现在按你说的做，已经成功减肥十斤了，接下来我想稳定当前体重，怎样才能做到不再继续廋，又能避免复胖?", thread_id="xuanran")
    # print("Response 2:", response2)