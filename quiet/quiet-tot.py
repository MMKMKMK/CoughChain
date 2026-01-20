import os
import base64
import time
from openai import OpenAI

def is_valid_base64(s):
    try:
        base64.b64decode(s, validate=True)
        return True
    except Exception:
        return False

# 初始化 OpenAI 客户端（兼容 DashScope）
client = OpenAI(
    api_key="xxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 支持的音频格式
SUPPORTED_FORMATS = ('.mp3', '.wav', '.flac', '.aac', '.ogg')

# 将音频编码为 base64
def encode_audio(audio_file_path):
    with open(audio_file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")

# 构建 CoT 提示词
def build_cot_prompt():
    return (
        "你是一名经验丰富的音频识别专家，请判断以下在**安静环境**中录制的音频是否包含咳嗽声。\n\n"
        "请从以下三个路径进行分析，并在每个路径中给出可信度评分（1~5分），同时写出评分依据：\n\n"
        "路径 A（咳嗽角度）：从“这是咳嗽”的角度出发，判断该声音是否具备以下咳嗽典型特征：\n"
        "- 爆破性气流（如“kh”、“ugh”样）\n"
        "- 呼气主导\n"
        "- 间歇性、短促节奏，可能1次或数次连发\n"
        "→ 咳嗽可信度评分（1~5）：\n\n"

        "路径 B（非咳嗽角度）：从“这不是咳嗽”的角度分析是否更像以下声音：清嗓（较轻、无爆破感）、说话声（语言结构）、笑声、打喷嚏（鼻腔）、动物声、背景杂音等等。\n"
        "→ 非咳嗽可信度评分（1~5）：\n\n"

        "路径 C（模糊角度）：若声音中缺乏明确特征、噪声干扰大、或介于多个类型之间，请说明不确定的原因。\n"
        "→ 模糊程度评分（1~5）：1表示极清晰，5表示非常模糊。\n\n"

        "判断标准：\n"
        "- 只有在咳嗽可信度高于非咳嗽和模糊评分时，才应判断为“有咳嗽”。\n\n"
        "- 其他情况一律判断为“无咳嗽”。\n\n"
        "由于环境安静，轻微但特征明确的声音也应视为有效证据。\n\n"
        "请最终只输出：“有咳嗽” 或 “无咳嗽”。可简要说明判断依据。"
    )

# 遍历文件夹，批量识别咳嗽（使用 CoT 提示），并实时写入结果
def detect_cough_in_folder(root_dir, output_file="tot_quiet.txt"):
    # 清空旧文件内容，准备写入新结果
    with open(output_file, "w", encoding='utf-8') as f:
        f.write("=== 咳嗽检测结果（结构化思维链 - 实时写入） ===\n\n")

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
                        prompt_text = build_cot_prompt()

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

                        answer = full_answer.strip()
                        print(f"✅ {file_path} => {answer}")

                        result = answer
                        if "有咳嗽" in answer:
                            cough_count += 1
                        elif "无咳嗽" in answer:
                            non_cough_count += 1
                        else:
                            error_count += 1

                except Exception as e:
                    print(f"❌ Failed to process {file_path}: {e}")
                    result = f"Error: {e}"
                    error_count += 1

                # 每处理完一个文件就写入一次结果
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
    root_audio_dir = " "
    detect_cough_in_folder(root_audio_dir)

