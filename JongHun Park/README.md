## TEAM-IKYO

### Stage 3 - MRC (Machine Reading Comprehension) :question: :exclamation:

#### Retrieval - Dense Embedding v4

> - mecab을 이용한 형태소 분석으로 pre tokeinzing
> - answer token이 context 내에 동일한 token으로 존대하도록 처리
> - context를 sentence 단위로 position ids 생성
> - training - model의 max length(512)에 맞춰서 random하게 일부 context를 input으로 사용
> - inference - 512 size로 50% overlap 하는 sliding window 방식으로 전체 context에 대한 feature를 여러개 뽑은 뒤 average polling으로 최종 output featrue로 similarity 계산
> - Top 1 context로 accuracy를 측정하여 best model 결정

#### Retrieval - Dense Embedding v5

> - 기존의 preprocessing을 학습과는 별개로 해준 후 new dataset으로 저장 (retrieval_preprocessing.ipynb 먼저 실행해야함)
> - Elastic search를 이용하여 1개의 question 마다 ground truth를 포함하는 top 20의 context를 추출하여 학습에 사용함
> - wiki data에서 top 20으로 추출된 context마다 position_ids를 만들어 주는 과정이 생각보다 복잡하여 구현 안됨 (추후 예정)
> - training - top 20의 context 별로 max length(512)에 맞춰서 random하게 일부 context를 input으로 사용 (추후 mrc task의 answer를 포함하는 context 일부만 사용하는 방식과 비교 예정)
> - inference - 기존 sliding window 방식으로 context를 자르는 것은 동일하지만 마지막에 average polling을 하지 않고, 각각의 context 전부와similarity를 계산하여 가장 높은 similarity를 보이는 것이 ground truth에서 잘라진 context인지 확인하는 방식으로 accuracy 계산
> - validation part에서 loss 계산 제거 (불필요)
> - 학습 시간이 길어짐에 따라 best model 갱신시 모델 저장, 저장된 모델이 존재하면 덮어써서 저장되는 방식
