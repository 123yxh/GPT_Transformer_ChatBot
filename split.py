import os.path

"""
划分数据集：获取数据集长度并且按照比例划分train以及val
"""
def split_dataset(file_path, output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    datas = []
    with open(file_path, "r", encoding='utf-8') as f:
        for line in f:
            if not line or line == "":
                continue
            datas.append(line)
    # 总的数据量
    print(len(datas))
    # 划分数据集：获取数据集长度并且按照比例划分train以及val
    train = datas[0:270000]
    val = datas[270000:274147]

    # 划分数据集：获取数据集长度并且按照比例划分train以及val
    # total_samples = len(datas)
    # train = datas[:total_samples - 3000]
    # val = datas[-3000:]

    with open(os.path.join(output_path, "train.json"), "w", encoding="utf-8") as w:
        for line in train:
            w.write(line)
            w.flush()

    with open(os.path.join(output_path, "val.json"), "w", encoding="utf-8") as w:
        for line in val:
            w.write(line)
            w.flush()
    print("train count: ", len(train))
    print("val count: ", len(val))


if __name__ == '__main__':
    file_path = "data/train.jsonl"
    split_dataset(file_path=file_path, output_path="data")

