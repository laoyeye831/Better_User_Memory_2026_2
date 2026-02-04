from dotenv import load_dotenv
import os
from serpapi import SerpApiClient
from typing import Dict, Any
from datetime import datetime
import pytz

import sympy as sp
import numpy as np
from scipy import optimize

# 加载 .env 文件中的环境变量
load_dotenv()


def get_rag_history() -> List[str]:
    """
    引用RAG，返回以前需要的对话记录的片段

    详细说明：
    - 此函数用于从RAG系统中检索以前的对话记录片段
    - 这些片段是系统认为与当前任务相关的历史对话内容
    - 返回的是一个字符串列表，每个字符串代表一个对话记录片段
    - 后续实现将包含具体的RAG检索逻辑

    返回值：
    - List[str]: 包含以前需要的对话记录片段的列表
    """
    # 生成一些模拟的对话记录片段作为返回值
    return [
        "用户: 什么是RAG技术？",
        "系统: RAG (Retrieval-Augmented Generation) 是一种结合了信息检索和生成式AI的技术，通过从外部知识库检索相关信息来增强大语言模型的回答能力。",
        "用户: RAG与传统的LLM有什么区别？",
        "系统: 传统LLM依赖于训练数据中的知识，而RAG可以实时从外部数据源检索最新信息，克服了LLM知识截止日期的限制。",
        "用户: 如何实现一个简单的RAG系统？",
        "系统: 实现RAG系统通常需要以下步骤：1. 构建知识库并进行向量化；2. 实现检索模块；3. 设计提示词模板；4. 集成LLM生成回答。"
    ]


def update_rag_vector_store(action: str, concluded_content: str) -> None:

    """
    para:
    action: str, 具体的操作类型，有:
    {
    "Add" : 添加新的聊天记录
    "Correct" : 修改错误的聊天记录
    }
    concluded_content: str, 模型从提示词和自己生成的内容中总结出的聊天记录

    详细说明： 此函数用于Agent添加或修改RAG向量库的内容
    - 添加内容：
    如果 action 为“Add”，则添加内容。具体为在片段库和向量库的最后添加新的聊天记录

    - 修改内容：
    如果 action为“Correct”,则修改内容。 具体为让模型指定它收到的片段中哪几号片段需要删除并修改


    - 修改方式：
      1. 接收文档内容，将其分割成适当大小的片段
      2. 使用嵌入模型为每个片段生成向量表示
      3. 将向量和对应的文本片段存储到向量数据库中
      4. 对于更新操作，先删除旧版本，再添加新版本
      5. 对于删除操作，根据文档ID或内容特征定位并移除
    - 后续实现将包含具体的向量库操作逻辑，支持主流向量数据库如FAISS、Pinecone等
    - 此函数无返回值
    """
    # 函数体暂时为空，等待后续实现具体的向量库修改逻辑
    pass


def update_jcards_database() -> None:
    """
    修改Jcards库

    详细说明：
    - 此函数用于修改Jcards库的内容，包括添加新卡片、更新现有卡片或删除不需要的卡片
    - 修改内容：
      1. 添加新的卡片到Jcards库中，如产品信息、知识点、问答对等
      2. 更新Jcards库中已有的卡片，确保信息的准确性和完整性
      3. 删除Jcards库中过时或不再相关的卡片，保持库的质量
    - 修改方式：
      1. 接收卡片内容，包括卡片标题、正文、标签等元数据
      2. 对卡片内容进行预处理，如格式标准化、关键词提取等
      3. 将处理后的卡片存储到Jcards数据库中
      4. 对于更新操作，根据卡片ID或内容特征定位并替换旧版本
      5. 对于删除操作，根据卡片ID或内容特征定位并移除
    - 后续实现将包含具体的Jcards库操作逻辑，支持卡片的分类、索引和检索
    - 此函数无返回值
    """
    # 函数体暂时为空，等待后续实现具体的Jcards库修改逻辑
    pass


#
#
# def search(query: str) -> str:
#     """
#     一个基于SerpApi的实战网页搜索引擎工具。
#     它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
#     """
#     print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")
#     try:
#         api_key = os.getenv("SERPAPI_API_KEY")
#         if not api_key:
#             return "错误：SERPAPI_API_KEY 未在 .env 文件中配置。"
#
#         params = {
#             "engine": "google",
#             "q": query,
#             "api_key": api_key,
#             "gl": "cn",  # 国家代码
#             "hl": "zh-cn",  # 语言代码
#         }
#
#         client = SerpApiClient(params)
#         results = client.get_dict()
#
#         # 智能解析：优先寻找最直接的答案
#         if "answer_box_list" in results:
#             return "\n".join(results["answer_box_list"])
#         if "answer_box" in results and "answer" in results["answer_box"]:
#             return results["answer_box"]["answer"]
#         if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
#             return results["knowledge_graph"]["description"]
#         if "organic_results" in results and results["organic_results"]:
#             # 如果没有直接答案，则返回前三个有机结果的摘要
#             snippets = [
#                 f"[{i + 1}] {res.get('title', '')}\n{res.get('snippet', '')}"
#                 for i, res in enumerate(results["organic_results"][:3])
#             ]
#             return "\n\n".join(snippets)
#
#         return f"对不起，没有找到关于 '{query}' 的信息。"
#
#     except Exception as e:
#         return f"搜索时发生错误: {e}"
#
# # 时间工具
# # route 1-1-7
# def get_current_time(timezone: str = "Asia/Shanghai") -> str:
#     """
#     一个获取指定时区当前时间的工具。
#     默认返回中国标准时间（北京时间）。
#     参数:
#         timezone: 时区字符串，如 "Asia/Shanghai", "America/New_York", "UTC" 等
#     返回:
#         格式化的当前时间字符串，或错误信息
#     """
#     print(f"⏰ 正在获取 {timezone} 的当前时间...")
#     try:
#         # 获取时区对象
#         tz = pytz.timezone(timezone)
#         # 获取当前时间并转换为指定时区
#         current_time = datetime.now(tz)
#         # 格式化输出
#         formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S %Z")
#
#         return f"当前 {timezone} 时间: {formatted_time}"
#
#     except pytz.exceptions.UnknownTimeZoneError:
#         return f"错误：未知的时区 '{timezone}'。请使用如 'Asia/Shanghai' 这样的有效时区标识符。"
#     except Exception as e:
#         return f"获取时间时发生错误: {e}"
#
# # # 代码执行工具
# # def codeInterpreter(code: str):
# #     try:
# #         local_vars = {}
# #         exec(code, {"sp": sp, "np": np, "optimize": optimize}, local_vars)
# #         return local_vars.get("result", None)
# #     except Exception as e:
# #         return f"EXECUTION ERROR: {str(e)}"

class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具。
    """
    def __init__(self):
        # route: 1-1-1 引用参数：可引用的工具函数字典集，格式如下：

        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: callable):
        """
        向工具箱中注册一个新工具。
        """
        if name in self.tools:
            print(f"警告：工具 '{name}' 已存在，将被覆盖。")

        self.tools[name] = {"description": description, "func": func}
        print(f"工具 '{name}' 已注册。")

    # route: 1-1-6 根据名称获取一个工具的执行函数, name: 工具名称， 返回工具函数
    def getTool(self, name: str) -> callable:
        """
        根据名称获取一个工具的执行函数。
        """
        #Tools的数据类型： Dict[str, Dict[str, Any]]
        # 例： self.tools = {
        #             "search": {
        #                 "name": "search",
        #                 "description": "在网络上搜索信息",
        #                 "func": self.search_web  # ← 存储函数引用
        #             },
        #             "calculate": {
        #                 "name": "calculate",
        #                 "description": "执行数学计算",
        #                 "func": self.calculate_expression  # ← 存储函数引用
        #             }
        # }
        # name为工具名称，func存储函数引用
        return self.tools.get(name, {}).get("func")
    # route: 1-1-1 获取所有可用工具的格式化描述字符串。
    def getAvailableTools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串。
        """
        return "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])


# --- 工具初始化与使用示例 ---
if __name__ == '__main__':
    # 1. 初始化工具执行器
    toolExecutor = ToolExecutor()

    # 2. 注册工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.registerTool("Search", search_description, search)
    time_description = "一个获取当前时间的工具。当你需要回答关于时事的问题时，应使用此工具获取最新的时间。"
    toolExecutor.registerTool("Time", time_description, get_current_time)

    # 3. 打印可用的工具
    print("\n--- 可用的工具 ---")
    print(toolExecutor.getAvailableTools())

    # 4. 智能体的Action调用，这次我们问一个实时性的问题
    print("\n--- 执行 Action: Search['英伟达最新的GPU型号是什么'] ---")
    tool_name = "Search"
    tool_input = "英伟达最新的GPU型号是什么"

    tool_function = toolExecutor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- 观察 (Observation) ---")
        print(observation)
    else:
        print(f"错误：未找到名为 '{tool_name}' 的工具。")