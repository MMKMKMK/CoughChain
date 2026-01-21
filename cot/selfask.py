import os
import base64
import time
import re
from openai import OpenAI

def is_valid_base64(s):
    try:
        base64.b64decode(s, validate=True)
        return True
    except Exception:
        return False

# 初始化 OpenAI 客户端（兼容 DashScope）
client = OpenAI(
    api_key="xxx",  # 请替换为你的实际 key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 支持的音频格式
SUPPORTED_FORMATS = ('.mp3', '.wav', '.flac', '.aac', '.ogg')

# 将音频编码为 base64
def encode_audio(audio_file_path):
    with open(audio_file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")

# 构建自问自答式推理提示词，并规定输出格式
def build_selfask_prompt():
    return (
        "你是一名音频识别专家，请判断以下音频是否包含咳嗽声。\n\n"
        "请你自己提出判断所需的关键问题，并逐一回答：\n\n"
        "问题1：音频中是否存在突发的、响亮的声音事件？这些声音应具有突然爆发、短时、明显响亮的特点，区别于连续的说话声、背景噪声或鸟鸣等环境音。\n"
        "→ 回答：是 / 否 + 简要说明\n\n"
        "问题2：这些声音是否呈现典型的咳嗽特征？包括：伴随爆破性气流（如“kh”、“ugh”等发声）、由呼气主导、持续时间较短并具有间歇性，区别于如清嗓、打喷嚏、笑声等其他呼吸类或喉部声音。\n"
        "→ 回答：是 / 否 + 简要说明\n\n"
        "问题3：该声音是否整体在节奏、音色、结构上与咳嗽一致？即具有周期性、间歇性（可能为1次或连续几次），同时排除掉说话声、呼吸声、机械声等非咳嗽结构的声音。\n"
        "→ 回答：是 / 否 + 简要说明\n\n"
        "问题4：该声音是否不属于其他常见声音类别（如清嗓、笑声、打喷嚏、动物叫声、环境声等），这些通常缺乏咳嗽的气流爆破、呼气主导、节奏性特征？\n"
        "→ 回答：是 / 否 + 简要说明\n\n"
        "判断标准：若所有问题均回答“是”，才判断为“最终结论：有咳嗽”；否则判断为“最终结论：无咳嗽”。\n\n"
        "请最后输出：“最终结论：有咳嗽” 或 “最终结论：无咳嗽”，并说明简要理由。"
    )

def extract_final_judgment(text):
    """
    从模型的输出中提取最终的判断结论。
    假设模型输出包含“最终结论：有咳嗽”或“最终结论：无咳嗽”格式的结论。
    """
    # 匹配“最终结论：有咳嗽”或“最终结论：无咳嗽”
    match = re.search(r"(最终结论[:：]*)(有咳嗽|无咳嗽)", text)

    if match:
        return match.group(2)  # 返回“有咳嗽”或“无咳嗽”

    return "无法判断"  # 或者根据需要返回默认值如“无咳嗽”

def detect_cough_in_folder(root_dir, output_file="cough_latest_selfask2.txt"):
    with open(output_file, "w", encoding='utf-8') as f:
        f.write("=== 咳嗽检测结果（实时写入） ===\n\n")

    cough_count = 0
    non_cough_count = 0
    error_count = 0

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(SUPPORTED_FORMATS):
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")

                try:
                    time.sleep(2)
                    base64_audio = encode_audio(file_path)
                    file_ext = os.path.splitext(file)[1][1:].lower()

                    if not is_valid_base64(base64_audio):
                        result = "生成的 Base64 数据无效"
                        error_count += 1
                    else:
                        prompt_text = build_selfask_prompt()

                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_audio",
                                        "input_audio": {
                                            "data": f"data:audio/{file_ext};base64,{base64_audio}",
                                            "format": file_ext,
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": prompt_text
                                    }
                                ]
                            }
                        ]

                        response = client.chat.completions.create(
                            model="qwen-omni-turbo",
                            messages=messages,
                            modalities=["text"],
                            stream=True,
                            stream_options={"include_usage": True}
                        )

                        full_answer = ""
                        for chunk in response:
                            if chunk.choices:
                                delta = chunk.choices[0].delta
                                if delta.content:
                                    full_answer += delta.content

                        # 直接从模型的结论中提取结果
                        judgment = extract_final_judgment(full_answer.strip())

                        # 打印推理结果和最终判断
                        print(f"✅ {file_path} => {judgment}")

                        result = f"{judgment}\n模型回复：{full_answer.strip()}"
                        if judgment == "有咳嗽":
                            cough_count += 1
                        elif judgment == "无咳嗽":
                            non_cough_count += 1
                        else:
                            error_count += 1

                        # 处理完一个文件后才写入结果
                        with open(output_file, "a", encoding='utf-8') as f:
                            f.write(f"文件路径: {file_path}\n")
                            f.write(f"检测结果: {result}\n")
                            f.write("-" * 40 + "\n")

                except Exception as e:
                    print(f"❌ Failed to process {file_path}: {e}")
                    result = f"Error: {e}"
                    error_count += 1

                    # 处理失败时，也要写入失败信息
                    with open(output_file, "a", encoding='utf-8') as f:
                        f.write(f"文件路径: {file_path}\n")
                        f.write(f"检测结果: {result}\n")
                        f.write("-" * 40 + "\n")

    # 最后追加写入统计信息
    with open(output_file, "a", encoding='utf-8') as f:
        f.write("\n=== 统计信息 ===\n")
        f.write(f"有咳嗽音频数量: {cough_count}\n")
        f.write(f"无咳嗽音频数量: {non_cough_count}\n")
        f.write(f"处理失败数量: {error_count}\n")
        f.write("=== 检测完成 ===\n")

    print("✅ 检测完成，结果已实时保存。")
    return cough_count, non_cough_count, error_count

# 主程序入口
if __name__ == "__main__":
    root_audio_dir = " "  # 修改为你的音频目录
    detect_cough_in_folder(root_audio_dir)


