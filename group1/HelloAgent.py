import requests
import json
import os
from openai import OpenAI
import re

# 设计一个指令模板，告诉LLM它应该扮演什么角色、拥有哪些工具、以及如何格式化它的思考和行动
AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。
# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。
# 行动格式:
你的回答必须严格遵循以下格式。首先是你的思考过程，然后是你要执行的具体行动。
Thought: [这里是你的思考过程和下一步计划]
Action: [这里是你要调用的工具，格式为 function_name(arg_name="arg_value")]
# 任务完成:
当你收集到足够的信息，能够回答用户的最终问题时，你必须在`Action:`字段后使用 `finish(answer="...")`来输出最终答案。
请开始吧！
"""

# 天气查询工具：使用免费的天气查询服务wttr.in获取天气信息
def get_weather(city: str) -> str:
    # API端点，我们请求JSON格式的数据
    url = f"https://wttr.in/{city}?format=j1"
    try:
        # 发起网络请求
        response = requests.get(url)
        # 检查响应状态码是否为200 (成功)
        response.raise_for_status()
        # 解析返回的JSON数据
        data = response.json()
        # 提取当前天气状况
        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value']
        temp_c = current_condition['temp_C']
        # 格式化成自然语言返回
        return f"{city}当前天气:{weather_desc}，气温{temp_c}摄氏度"
    except requests.exceptions.RequestException as e:
        # 处理网络错误
        return f"错误:查询天气时遇到网络问题 - {e}"
    except (KeyError, IndexError) as e:
        # 处理数据解析错误
        return f"错误:解析天气数据失败，可能是城市名称无效 - {e}"

# 旅游景点查询工具：使用Tavily API根据城市和天气搜索推荐的旅游景点
def get_attraction(city: str, weather: str) -> str:
    # 读取API密钥
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "错误:未配置TAVILY_API_KEY环境变量。"
    # 初始化Tavily客户端
    tavily = TavilyClient(api_key=api_key)
    # 构造一个精确的查询
    query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"
    try:
        # 调用API，include_answer=True会返回一个综合性的回答
        response = tavily.search(query=query, search_depth="basic", include_answer=True)
        # Tavily返回的结果已经非常干净，可以直接使用
        # response['answer'] 是一个基于所有搜索结果的总结性回答
        if response.get("answer"):
            return response["answer"]
        # 如果没有综合性回答，则格式化原始结果
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")
        if not formatted_results:
            return "抱歉，没有找到相关的旅游景点推荐。"
        return "根据搜索，为您找到以下信息:\n" + "\n".join(formatted_results)
    except Exception as e:
        return f"错误:执行Tavily搜索时出现问题 - {e}"

# 将所有工具函数放入一个字典，方便后续调用
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}

# 可以用于调用任何兼容OpenAI接口的LLM服务的客户端
class OpenAICompatibleClient:
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """调用LLM API来生成回应。"""
        print("正在调用大语言模型...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt}, # 系统消息，设置模型行为
                {'role': 'user', 'content': prompt} # 用户消息，传入实际提问
            ]
            response = self.client.chat.completions.create(
                model=self.model,   # 指定模型名称
                messages=messages,  # 传入对话消息列表
                stream=False        # 关闭流式响应，一次返回完整结果
            )
            # 提取模型生成的回答内容
            answer = response.choices[0].message.content
            print("大语言模型响应成功。")
            return answer
        except Exception as e:
            # 捕获所有异常（如网络错误、API 密钥无效等），打印错误信息并返回错误提示
            print(f"调用LLM API时发生错误: {e}")
            return "错误:调用语言模型服务时出错。"

""" 请根据使用的服务，替换成对应的凭证和地址 """
# 配置LLM API密钥，这里可以使用任何支持OpenAI API的模型，如deepseek、gpt-4等
# 这里我们使用deepseek模型，需要在DeepSeek API 开放平台注册账号并获取API KEY
# DeepSeek API 开放平台 https://platform.deepseek.com/
OPENAI_API_KEY = "sk-********************"
# 配置LLM API的基础URL，这里使用deepseek模型的URL
BASE_URL = "https://api.deepseek.com"
# 使用的模型
MODEL_ID = "deepseek-chat"
# 配置Tavily API密钥，在https://www.tavily.com/在官网注册后免费获取API KEY
os.environ['TAVILY_API_KEY'] = "tvly-dev-************"

llm = OpenAICompatibleClient(
    model = MODEL_ID,
    api_key = OPENAI_API_KEY,
    base_url = BASE_URL
)
# 用户输入的初始请求
user_prompt = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
# 初始化对话历史记录，存储用户请求
prompt_history = [f"用户请求: {user_prompt}"]
print(f"用户输入: {user_prompt}\n" + "="*40)
for i in range(5): # 设置最大循环次数为5，防止无限循环
    print(f"--- 循环 {i+1} ---\n")
    # 构建完整的Prompt，将历史对话拼接成字符串
    full_prompt = "\n".join(prompt_history)
    # 调用LLM（大语言模型）生成响应
    # 参数说明：- full_prompt: 完整的对话历史
    # - system_prompt=AGENT_SYSTEM_PROMPT: 系统提示词，定义模型的行为
    llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
    print(f"模型输出:\n{llm_output}\n")
    # 将模型输出添加到历史记录中
    prompt_history.append(llm_output)
    # 使用正则表达式解析模型输出，提取Action部分
    # re.DOTALL 标志让 ".*" 匹配包括换行符在内的所有字符
    action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
    # 如果未找到Action，报错并终止循环
    if not action_match:
        print("解析错误:模型输出中未找到 Action。")
        break
    # 提取Action内容并去除首尾空格
    action_str = action_match.group(1).strip()
    # 检查是否为结束动作（finish）
    if action_str.startswith("finish"):
        final_answer = re.search(r'finish\(answer="(.*)"\)', action_str).group(1)
        print(f"任务完成，最终答案: {final_answer}")
        break
    # 解析工具名称和参数
    tool_name = re.search(r"(\w+)\(", action_str).group(1)
    args_str = re.search(r"\((.*)\)", action_str).group(1)
    # 将参数解析为字典，格式为：key="value"
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))
    # 检查工具是否可用，并调用对应工具
    if tool_name in available_tools:
        observation = available_tools[tool_name](**kwargs)
    else:
        observation = f"错误:未定义的工具 '{tool_name}'"
    # 记录工具执行结果
    observation_str = f"Observation: {observation}"
    print(f"{observation_str}\n" + "="*40)
    # 将观察结果添加到历史记录中，供下一轮循环使用
    prompt_history.append(observation_str)

#
# def test(query: str, a: int, b: str): {
#     print(query)
# }
#
#
# kwargs = {
#     "query":"test1",
#     "a":1,
#     "b":"test2"
# }
# test(**kwargs)
