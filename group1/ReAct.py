import re
from LLMCompatibleClient import LLMCompatibleClient
from ToolExecutor import ToolExecutor, search, get_current_time

# 系统提示词模板
AGENT_SYSTEM_PROMPT = """
请注意，你是一个有能力调用外部工具的智能助手，当你需要回答关于时事的问题时，应先使用工具获取最新的时间，然后再回答问题。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `tool_name[tool_input]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用`Finish["..."]`来输出最终答案。

现在，请开始吧！
"""

class ReActAgent:
    def __init__(self, llm_client: LLMCompatibleClient, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []
    # route: 1-1 ReAct架构的主循环
    def run(self, question: str):
        self.history = [f"用户请求: {question}"]
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            # route: 1-1-1
            # 返回系统prompt
            system_prompt = AGENT_SYSTEM_PROMPT.format(tools=self.tool_executor.getAvailableTools())
            prompt = "\n".join(self.history)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            # route: 1-1-2
            # 大模型的响应存入response_text
            response_text = self.llm_client.think(messages=messages)
            if not response_text:
                print("错误：LLM未能返回有效响应。");
                break


            self.history.append(response_text)
            # route: 1-1-3
            thought, action = self._parse_output(response_text)
            if thought: print(f"🤔 思考: {thought}")
            else: print("警告：未能解析出有效的Action，流程终止。"); break

            # 如果动作类型是Finish，即模型认为循环可以结束了
            if action.startswith("Finish"):
                # route: 1-1-4
                final_answer = self._parse_action_input(action)
                print(f"🎉 最终答案: {final_answer}")
                return final_answer

            # route: 1-1-5
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                self.history.append("Observation: 无效的Action格式，请检查。");
                continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")
            # route: 1-1-6
            tool_function = self.tool_executor.getTool(tool_name)
            # route: 1-1-7 执行tool_function函数，参数为tool_input,一般为str, 返回函数的结果给observation
            observation = tool_function(tool_input) if tool_function else f"错误：未找到名为 '{tool_name}' 的工具。"
            # 将工具的调用结果，即observation加入聊天历史
            self.history.append(observation)
            print(f"👀 观察: {observation}")

        print("已达到最大步数，流程终止。")
        return None
    # route: 1-1-3 将模型的thought和action从模型输出text中分离出来，返回thought, action
    def _parse_output(self, text: str):
        thought_match = re.search(r"Thought: (.*)", text)
        action_match = re.search(r"Action: (.*)", text)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    # route: 1-1-5
    #  输入示例：
    #  action_text = "Search[OpenAI最新消息]"
    #  _parse_action 处理后：
    #  返回("Search", "OpenAI最新消息")
    def _parse_action(self, action_text: str):
        match = re.match(r"(\w+)\[(.*)\]", action_text)
        return (match.group(1), match.group(2)) if match else (None, None)

    # route: 1-1-4
    #  用户问："中国的首都是哪里？"
    #  Agent思考：
    #  1.我需要找到中国的首都
    #  2.我知道是北京
    #  3.我应该输出Finish[北京]
    #  该函数功能为提取Finish后【】里的字符串
    def _parse_action_input(self, action_text: str):
        match = re.match(r"Finish\[(.*)\]", action_text, re.DOTALL)
        # match = re.match(r"\w+\[(.*)\]", action_text)
        return match.group(1) if match else ""


if __name__ == '__main__':
    llm = LLMCompatibleClient()
    tool_executor = ToolExecutor()
    time_description = "一个获取最新时间的工具，工具的输入是时区（如 'Asia/Shanghai', 'America/New_York', 'UTC' 等）。当你需要回答关于时事的问题时，应使用此工具获取最新的时间。"
    tool_executor.registerTool("Time", time_description, get_current_time)
    search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_desc, search)
    agent = ReActAgent(llm_client=llm, tool_executor=tool_executor)
    question = "华为最新的手机是哪一款？它的主要卖点是什么？"
    # route: 1
    agent.run(question)
