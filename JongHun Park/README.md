## TEAM-IKYO

### Stage 3 - MRC (Machine Reading Comprehension) :question: :exclamation:

#### Retrieval - Dense Embedding

> - mecab을 이용한 형태소 분석으로 pre tokeinzing
> - answer token이 context 내에 동일한 token으로 존대하도록 처리
> - context를 sentence 단위로 position ids 생성
> - training - model의 max length(512)에 맞춰서 random하게 일부 context를 input으로 사용
> - inference - 512 size로 50% overlap 하는 sliding window 방식으로 전체 context에 대한 feature를 여러개 뽑은 뒤 average polling으로 최종 output featrue로 similirity 계산
> - Top 1 context로 accuracy를 측정하여 best model 결정
