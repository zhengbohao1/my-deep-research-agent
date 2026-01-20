from collections import Counter
from typing import List, Optional

from hello_agents import Message
from hello_agents.context import ContextBuilder, ContextPacket, ContextConfig
from hello_agents.tools import MemoryTool, RAGTool
import jieba
from typing import  List, Optional, Tuple
from datetime import datetime
import tiktoken
import math
from rank_bm25 import BM25Okapi

class MyContextBuilder(ContextBuilder):
    def __init__(self, memory_tool: Optional[MemoryTool] = None, rag_tool: Optional[RAGTool] = None,
                 config: Optional[ContextConfig] = None):
        super().__init__(memory_tool, rag_tool, config)

    def tokenize(self,text: str) -> list[str]:
        # 可选：去停用词、过滤太短的词
        words = jieba.cut(text.lower())
        return [w for w in words if len(w) >= 2]  # 至少两个字符的词

    def _gather(
            self,
            user_query: str,
            conversation_history: List[Message],
            system_instructions: Optional[str],
            additional_packets: List[ContextPacket]
    ) -> List[ContextPacket]:
        """Gather: 收集候选信息"""
        packets = []

        # P0: 系统指令（强约束）
        if system_instructions:
            packets.append(ContextPacket(
                content=system_instructions,
                metadata={"type": "instructions"}
            ))

        # P1: 从记忆中获取任务状态与关键结论
        if self.memory_tool:
            try:
                # 搜索任务状态相关记忆
                state_results = self.memory_tool.execute(
                    "search",
                    query="(任务状态 OR 子目标 OR 结论 OR 阻塞)",
                    min_importance=0.7,
                    limit=5
                )
                if state_results and "未找到" not in state_results:
                    packets.append(ContextPacket(
                        content=state_results,
                        metadata={"type": "task_state", "importance": "high"}
                    ))

                # 搜索与当前查询相关的记忆
                related_results = self.memory_tool.execute(
                    "search",
                    query=user_query,
                    limit=5
                )
                if related_results and "未找到" not in related_results:
                    packets.append(ContextPacket(
                        content=related_results,
                        metadata={"type": "related_memory"}
                    ))
            except Exception as e:
                print(f"⚠️ 记忆检索失败: {e}")

        # P2: 从RAG中获取事实证据
        if self.rag_tool:
            try:
                rag_results = self.rag_tool.run({
                    "action": "search",
                    "query": user_query,
                    "top_k": 5
                })
                if rag_results and "未找到" not in rag_results and "错误" not in rag_results:
                    packets.append(ContextPacket(
                        content=rag_results,
                        metadata={"type": "knowledge_base"}
                    ))
            except Exception as e:
                print(f"⚠️ RAG检索失败: {e}")

        # P3: 对话历史（辅助材料）
        if conversation_history:
            # 只保留最近N条
            recent_history = conversation_history[-10:]
            history_text = "\n".join([
                f"[{msg.role}] {msg.content}"
                for msg in recent_history
            ])
            packets.append(ContextPacket(
                content=history_text,
                metadata={"type": "history", "count": len(recent_history)}
            ))

        # 添加额外包
        packets.extend(additional_packets)

        return packets



    #解析用户对话历史
    def parse_history(self,content: str) -> List[Tuple[str | None, str | None]]:
        """
    按 [user] / [assistant] 块解析 history
    返回 [(user_msg, assistant_msg), ...]
        """
        turns = []

        current_role = None  # "user" | "assistant"
        buffer = []

        current_user = None
        current_assistant = None

        def flush():
            nonlocal current_user, current_assistant, buffer, current_role
            text = "\n".join(buffer).strip() if buffer else None
            if current_role == "user":
                current_user = text
            elif current_role == "assistant":
                current_assistant = text
            buffer = []

        for line in content.splitlines():
            line_stripped = line.strip()

            if line_stripped.startswith("[user]"):
                # 先收尾前一个 block
                flush()
                # 如果已有一组完成，先入 turns
                if current_user or current_assistant:
                    turns.append((current_user, current_assistant))
                    current_user, current_assistant = None, None

                current_role = "user"
                buffer = [line_stripped[len("[user]"):].lstrip()]

            elif line_stripped.startswith("[assistant]"):
                flush()
                current_role = "assistant"
                buffer = [line_stripped[len("[assistant]"):].lstrip()]

            else:
                buffer.append(line)

        # 收尾
        flush()
        if current_user or current_assistant:
            turns.append((current_user, current_assistant))

        return turns

    #构建IDF
    def build_idf(self,turns, tokenize):
        doc_count = Counter()
        total_docs = 0

        for u, a in turns:
            text = (u or "") + " " + (a or "")
            tokens = set(tokenize(text))
            if tokens:
                total_docs += 1
                for t in tokens:
                    doc_count[t] += 1

        idf = {}
        for t, df in doc_count.items():
            idf[t] = math.log((1 + total_docs) / (1 + df)) + 1

        return idf

    #只计算单个对话组的相关性
    def turn_relevance(self,turn,query_tokens,tokenize,idf,assistant_weight=0.7) -> float:
        user_text, assistant_text = turn

        tokens = set()
        if user_text:
            tokens |= set(tokenize(user_text))
        if assistant_text:
            tokens |= set(tokenize(assistant_text))

        overlap = set(query_tokens) & tokens
        if not overlap:
            return 0.0

        num = sum(idf.get(t, 1.0) for t in overlap)
        den = sum(idf.get(t, 1.0) for t in query_tokens)

        return num / (den + 1e-6)

    def filter_history_packet(
            self,
            packet,
            user_query,
            tokenize,
            min_score=0.25
    ):
        turns = self.parse_history(packet.content)
        if not turns:
            return None

        query_tokens = tokenize(user_query)
        idf = self.build_idf(turns, tokenize)

        kept_turns = []
        for turn in turns:
            score = self.turn_relevance(turn, query_tokens, tokenize, idf)
            if score >= min_score:
                kept_turns.append(turn)

        if not kept_turns:
            return None

        # ===== 正确的 block 级重组 =====
        blocks = []
        idx=1
        for u, a in kept_turns:
            if u:
                blocks.append(f"对话历史片段{idx}:\n[user]\n" + u.strip())
            if a:
                blocks.append("[assistant]\n" + a.strip()+f"\n对话历史片段{idx}结束\n")
            idx+=1

        packet.content = "\n".join(blocks)

        # token_count 建议：只算 content token，不做猜测
        packet.token_count = len(tokenize(packet.content))

        return packet

    def _select(
            self,
            packets: List[ContextPacket],
            user_query: str
    ) -> List[ContextPacket]:
        #print(f"候选信息如下：{packets}")
        selected = []

        for p in packets:
            ptype = p.metadata.get("type")

            if ptype in ("instructions", "knowledge_base", "related_memory","task_state"):
                selected.append(p)

            elif ptype == "history":
                filtered = self.filter_history_packet(
                    p,
                    user_query,
                    self.tokenize,
                    min_score=0.25
                )
                if filtered:
                    selected.append(filtered)

        return selected

        # tokenized_corpus = [self.tokenize(p.content) for p in packets]
        #
        # # 构建 BM25 索引（只需做一次）
        # bm25 = BM25Okapi(tokenized_corpus)
        #
        # # ───────────────────────────────────────────────
        # # 1) 计算相关性（BM25 替换原来的关键词重叠）
        # # ───────────────────────────────────────────────
        # query_tokens = self.tokenize(user_query)
        #
        # if len(query_tokens) > 0 and len(packets) > 0:
        #     # bm25.get_scores() 返回一个 numpy array，对应每个 document 的分数
        #     bm25_scores = bm25.get_scores(query_tokens)
        #
        #     # # 归一化到 [0,1] 区间（可选，但推荐，便于和 recency 融合）
        #     # max_score = bm25_scores.max()
        #     # if max_score > 0:
        #     #     bm25_scores = bm25_scores / max_score
        #
        #     for i, packet in enumerate(packets):
        #         packet.relevance_score = float(bm25_scores[i])
        # else:
        #     for packet in packets:
        #         packet.relevance_score = 0.0
        #
        # # 2) 计算新近性（指数衰减）
        # def recency_score(ts: datetime) -> float:
        #     delta = max((datetime.now() - ts).total_seconds(), 0)
        #     tau = 3600  # 1小时时间尺度，可暴露到配置
        #     return math.exp(-delta / tau)
        #
        # # 3) 计算复合分：0.7*相关性 + 0.3*新近性
        # scored_packets: List[Tuple[float, ContextPacket]] = []
        # for p in packets:
        #     rec = recency_score(p.timestamp)
        #     score = 0.7 * p.relevance_score + 0.3 * rec
        #     scored_packets.append((score, p))
        #
        # # 4) 系统指令单独拿出，固定纳入
        # system_packets = [p for (_, p) in scored_packets if p.metadata.get("type") == "instructions"]
        # remaining = [p for (s, p) in sorted(scored_packets, key=lambda x: x[0], reverse=True)
        #              if p.metadata.get("type") != "instructions"]
        #
        # print(f"计算结果如下：{remaining}")
        #
        # # 5) 依据 min_relevance 过滤（对非系统包）
        # filtered = [p for p in remaining if p.relevance_score >= self.config.min_relevance]
        #
        # # 6) 按预算填充
        # available_tokens = self.config.get_available_tokens()
        # selected: List[ContextPacket] = []
        # used_tokens = 0
        #
        # # 先放入系统指令（不排序）
        # for p in system_packets:
        #     if used_tokens + p.token_count <= available_tokens:
        #         selected.append(p)
        #         used_tokens += p.token_count
        #
        # # 再按分数加入其余
        # for p in filtered:
        #     if used_tokens + p.token_count > available_tokens:
        #         continue
        #     selected.append(p)
        #     used_tokens += p.token_count
        #
        # print(f"筛选与排序结果如下：{selected}")

        #return selected