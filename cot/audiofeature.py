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
        "你是一个专业的音频分析助手，请直接分析输入的音频内容，并判断是否包含咳嗽声。\n\n"
        "咳嗽通常具有以下典型特征：\n"
        "- 突发性强，常表现为突然出现的短促声音，持续时间通常在0.3到1秒之间；\n"
        "- 音调变化明显，能量集中在中高频段，常伴有‘爆破’或‘摩擦’的音色；\n"
        "- 由呼气主导，能量迅速上升后快速衰减，有较强的气流感；\n"
        "- 通常孤立存在，不具有语音节奏和语义内容；\n"
        "- 请注意排除与咳嗽相似的非咳嗽声，如清嗓、打喷嚏、大笑、语言中的爆破音等。\n\n"
        "只有当音频中**同时具备多个典型咳嗽特征，并能清晰排除其他声音类型**时，才能判断为“有咳嗽”。\n"
        "若不确定，请保守判断为“无咳嗽”。\n\n"
        "最终，请仅输出以下两种之一：\n"
        "“有咳嗽” 或 “无咳嗽”。"
    )


# 遍历文件夹，批量识别咳嗽（使用 CoT 提示），并实时写入结果
def detect_cough_in_folder(root_dir, output_file="cough_omni-latest_audiofeature.txt"):
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


