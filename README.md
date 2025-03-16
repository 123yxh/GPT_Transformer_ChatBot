<div align='center' ><font size='170'>GPT_Transformer_ChatBot</font></div>

## First Step
Download the dataset and process it into a jsonl file and place it in the data folder; the format of the dataset should be like this：
```
{"question": "你有品尝过或者了解过北京烤鸭这一美食吗？", "answer": "当然有，北京烤鸭是中华美食的瑰宝之一，外皮酥脆、肉质鲜美，配上薄饼和特制酱料食用，非常美味。"}
{"question": "请问徐州这座城市的所在省份是中国的哪一个？", "answer": "江苏省"}
{"question": "您是否有过参与或者亲身体验摄影活动的经历？", "answer": "是的，我有过多次摄影体验，享受捕捉瞬间的美好。"}
```

## Second Step
Divide the training set and test set according to the length of the data set======**python split.py**

## Thrid Step
Building a vocabulary======**python vocab.py** 

## Four Step
Training the model======**python train.py** 

## Five Step
Training the model======**python inference.py** ; This process will load the trained **best.pt**, which is saved in the output directory.

The dataset and model I used for training are placed in: [data_model_link](https://drive.google.com/drive/u/0/folders/1fo03cko_eLEt9DjZXVibHygjDKK5T9CK); Besides, token_max.py can be used to view the maximum token length of the data set, providing a reference for the subsequent _max_len_ setting.

Reference：https://blog.csdn.net/qq_43692950/article/details/143642844
