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

# 遍历文件夹，批量识别咳嗽
def detect_cough_in_folder(root_dir):
    results = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(SUPPORTED_FORMATS):
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")

                try:
                    time.sleep(2)
                    base64_audio = encode_audio(file_path)
                    file_ext = os.path.splitext(file)[1][1:].lower()  # 获取扩展名：wav/mp3等

                    if not is_valid_base64(base64_audio):
                        results.append((file_path, "生成的 Base64 数据无效"))
                        continue

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
                                    "text": "这个音频中是否有人在咳嗽？只需回答“有咳嗽”或“无咳嗽”。"
                                }
                            ]
                        }
                    ]

                    response = client.chat.completions.create(
                        model="qwen-omni-turbo-latest",
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
                    print(f" {file_path} => {answer}")
                    results.append((file_path, answer))

                except Exception as e:
                    print(f" Failed to process {file_path}: {e}")
                    results.append((file_path, f"Error: {e}"))

    return results

# 主程序入口
if __name__ == "__main__":
    root_audio_dir = ""
    results = detect_cough_in_folder(root_audio_dir)

    cough_count = 0
    non_cough_count = 0
    error_count = 0

    with open("cough_detection_results_omni-latest.txt", "w", encoding='utf-8') as f:
        f.write("=== 咳嗽检测结果 ===\n\n")
        for file_path, result in results:
            f.write(f"文件路径: {file_path}\n")
            f.write(f"检测结果: {result}\n")
            f.write("-" * 40 + "\n")

            if result == "有咳嗽":
                cough_count += 1
            elif result == "无咳嗽":
                non_cough_count += 1
            else:
                error_count += 1

        f.write("\n=== 统计信息 ===\n")
        f.write(f"有咳嗽音频数量: {cough_count}\n")
        f.write(f"无咳嗽音频数量: {non_cough_count}\n")
        f.write(f"处理失败数量: {error_count}\n")

    print(" 检测完成，结果已保存为 cough_detection_results_omni-latest.txt")

