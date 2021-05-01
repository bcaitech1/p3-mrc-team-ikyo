## TEAM-IKYO

### Stage 3 - MRC (Machine Reading Comprehension) :question: :exclamation:

#### What I tried

- **`Retrieval`**
  - 기존 baseline TF-IDF **max_features** 변경
    - max_features가 커질수록 precision@5, 10, 30, 50 값이 올라감
  - 형태소가 아닌 **단어(2글자 이상)**만으로 TF-IDF
    - 기존 baseline보다 precision@1 기준 10% 성능 향상
    - 기존 baseline보다는 대체로 성능 향상을 보임
  - 형태소 기준으로 단어만 check(**연속된 단어는 하나의 단어**라고 가정 ex) "조선" "중기" => "조선 중기")
    - 기존 baseline보다 더 떨어짐..
    - Why? vocab수가 너무 많이 늘어남
  - BM25 + mecab 형태소 기준으로 tokenizing
    - 기존 baseline보다 precision@1 기준 30% 성능 향상
  - BM25 + mecab 단어(2글자 이상) 기준으로 tokenizing
    - BM25 + mecab 형태소 성능보다 약 5% 낮음.
- **`Reader`**
  - model 변경
    - bert-base-multilingual-cased -> monologg/koelectra-base-v3-discriminator

#### 적용한 IDEA

> - BM25
> - monologg/koelectra-base-v3-discriminator

#### How to Use

- **train.py 실행**
  - python train.py --output_dir ./models/train_dataset --do_train --overwrite_cache

- **inference.py 실행**
  - python inference.py --output_dir ../outputs/test_dataset/ --dataset_name ../input/data/data/test_dataset/ --model_name_or_path ./models/train_dataset/ --do_predict --overwrite_output_dir
