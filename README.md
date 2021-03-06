# ð Stage 3 - MRC (Machine Reading Comprehension) âï¸
> More explanations of our codes are in ð[TEAM-IKYO's WIKI](https://github.com/TEAM-IKYO/Open-Domain-Question-Answering/wiki)

<!-- Hosted by [2021 Boostcamp AI Tech](https://boostcamp.connect.or.kr/)-->
![TEAM-IKYOâs ODQA Final Score_v3](https://user-images.githubusercontent.com/45359366/121469986-6ea03e80-c9f8-11eb-9fd1-355e50537450.png)

## ð» Task Description
> More Description can be found at [AI Stages](http://boostcamp.stages.ai/competitions/31/overview/description)  
  
When you have a question like, "What is the oldest tree in Korea?", you may have asked it at a search engine. And these days it gives you an especially surprisingly accurate answer. How is it possible?

Question Answering is a field of research that creates artificial intelligence model that answers various kinds of questions. Among them, Open-Domain Question Answering is a more challenging issue because it has to find documents that can answer questions only using pre-built knowledge resources.

![image](https://user-images.githubusercontent.com/59340911/119260267-118d4600-bc0d-11eb-95bc-6ea68f7b0df4.png)

The model we'll create in this competition is made up of 2 stages.The first stage is called "retriever", which is the step to find question-related documents, and the next stage is called "reader", which is the step to read the document we've found at 1st stage and find the answer in the document. If we concatenate these two stages properly, we can make a question answering system that can answer no matter how tough questions are. The team which create a model that makes a more accurate answer will win this stage.
![image](https://user-images.githubusercontent.com/59340911/119260915-f1ab5180-bc0f-11eb-9ddc-cad4585bc8ce.png)

---

## ð Directory
```
p3-mrc-team-ikyo
âââ code
â   âââ arguments.py
â   âââ data_processing.py
â   âââ elasticsearch_retrieval.py
â   âââ inference.py
â   âââ mask.py
â   âââ mk_retrieval_dataset.py
â   âââ model
â   â   âââ ConvModel.py
â   â   âââ QAConvModelV1.py
â   â   âââ QAConvModelV2.py
â   â   âââ QueryAttentionModel.py
â   âââ prepare_dataset.py
â   âââ question_labeling
â   â   âââ data_set.py
â   â   âââ question_labeling.py
â   â   âââ train.py
â   âââ requirements.txt
â   âââ retrieval_dataset.py
â   âââ retrieval_inference.py
â   âââ retrieval_model.py
â   âââ retrieval_train.py
â   âââ run_elastic_search.py
â   âââ script
â   â   âââ inference.sh
â   â   âââ pretrain.sh
â   â   âââ retrieval_inference.sh
â   â   âââ retrieval_prepare_dataset.sh
â   â   âââ retrieval_train.sh
â   â   âââ run_elastic_search.sh
â   â   âââ train.sh
â   âââ train_mrc.py
â   âââ trainer_qa.py
â   âââ utils_qa.py
âââ etc
    âââ my_stop_dic.txt
```
