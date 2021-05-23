## TEAM-IKYO

### Stage 3 - MRC (Machine Reading Comprehension) :question: :exclamation:

### What I tried

- **`Retrieval`**
  - 기존 baseline TF-IDF **max_features** 변경
    - max_features가 커질수록 precision@5, 10, 30, 50 값이 올라감
  - 형태소가 아닌 **단어(2글자 이상)** 만으로 TF-IDF
    - 기존 baseline보다 precision@1 기준 10% 성능 향상
    - 기존 baseline보다는 대체로 성능 향상을 보임
  - 형태소 기준으로 단어만 check(**연속된 단어는 하나의 단어**라고 가정 ex) "조선" "중기" => "조선 중기")
    - 기존 baseline보다 더 떨어짐..
    - Why? vocab수가 너무 많이 늘어남
  - **`BM25`** + mecab 형태소 기준으로 tokenizing
    - 기존 baseline보다 precision@1 기준 30% 성능 향상
- **`Reader`**
  - Data preprocessing
    - 필요 없는 **특수 문자 제거** ex) \n, \s+ ... => EM 기준 약1.5% 상승 & F1 score 기준 2% 상승
  - model 변경
    - bert-base-multilingual-cased -> monologg/koelectra-base-v3-discriminator -> xlm-roberta-large
  - Data augmentation
    - **`Question`**에서 **단어만 확률적으로 Masking**(20% 확률로 적용) => EM 기준 약1.5% 상승 & F1 score 기준 약 10% 상승
    - 재희님 Question Augmentation 적용 (**역번역 활용하여 Question 데이터 증강**) => 
- **`Inference`**
  - **`Top k`** 인 context concat 후 Reader
  - Reader로 부터 나온 **answer postprocessing** ex) 은, 는, 의.. 제거

### How to Use

- **train.py 실행**
  - python train.py --output_dir ./models/train_dataset --do_train --do_eval --overwrite_cache

- **inference.py 실행**
  - python inference.py --output_dir ../outputs/test_dataset/ --dataset_name ../input/data/data/test_dataset/ --model_name_or_path ./models/train_dataset/ --do_predict --overwrite_output_dir
