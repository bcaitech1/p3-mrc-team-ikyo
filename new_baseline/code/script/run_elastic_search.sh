# Home Directory에 elasticsearch-7.6.2 설치
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.6.2-linux-x86_64.tar.gz -q -P /opt/ml
tar -xzf /opt/ml/elasticsearch-7.6.2-linux-x86_64.tar.gz -C /opt/ml/
chown -R daemon:daemon /opt/ml/elasticsearch-7.6.2

# Python Library 설치
pip install elasticsearch
pip install tqdm

# nori Tokenizer 설치
/opt/ml/elasticsearch-7.6.2/bin/elasticsearch-plugin install analysis-nori

# elastic search stop word 설정
mkdir /opt/ml/elasticsearch-7.6.2/config/user_dic
cp ../../new_baseline/etc/my_stop_dic.txt /opt/ml/elasticsearch-7.6.2/config/user_dic/.

# python script file 실행
python3 run_elastic_search.py

# elastic search 실행 여부 확인
ps -ef | grep elastic