import json
import os

"""
构建词表，这里针对于两种不同的数据集进行处理；主要是读取方式和字不同
"""
def build_vocab(file_path):
    # # 读取 JSON 文件
    # with open(file_path, 'r', encoding='utf-8') as r:
    #     data = json.load(r)
    #
    # # 提取所有文本
    # texts = []
    # for item in data["data"]:  # 从 "data" 字段中提取数据
    #     question = item.get("question", "")
    #     answer = item.get("answer", "")
    #     texts.append(question)
    #     texts.append(answer)

    # 读取所有文本
    texts = []
    with open(file_path, 'r', encoding='utf-8') as r:
        for line in r:
            if not line:
                continue
            line = json.loads(line)
            question = line["question"]
            answer = line["answer"]
            texts.append(question)
            texts.append(answer)

    # 拆分 Token
    words = set()
    for t in texts:
        if not t:
            continue
        for word in t.strip():
            words.add(word)
    words = list(words)
    words.sort()
    # 特殊Token
    # pad 占位、unk 未知、sep 结束
    word2id = {"<pad>": 0, "<unk>": 1, "<sep>": 2}
    # 构建词表
    word2id.update({word: i + len(word2id) for i, word in enumerate(words)})
    id2word = list(word2id.keys())
    vocab = {"word2id": word2id, "id2word": id2word}

    # 确保目录存在，没有则新建
    output_dir = 'data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存词表为 JSON 文件
    vocab_json = json.dumps(vocab, ensure_ascii=False)
    output_file = os.path.join(output_dir, 'vocab.json')
    with open(output_file, 'w', encoding='utf-8') as w:
        w.write(vocab_json)
    print(f"finish. words: {len(id2word)}")

if __name__ == '__main__':
    build_vocab("data/train.jsonl")

