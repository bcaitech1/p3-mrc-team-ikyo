wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.6.2-linux-x86_64.tar.gz -q -P /opt/ml
tar -xzf /opt/ml/elasticsearch-7.6.2-linux-x86_64.tar.gz -C /opt/ml/
chown -R daemon:daemon /opt/ml/elasticsearch-7.6.2
pip install elasticsearch
pip install tqdm
/opt/ml/elasticsearch-7.6.2/bin/elasticsearch-plugin install analysis-nori
mkdir /opt/ml/elasticsearch-7.6.2/config/user_dic
cp ../../new_baseline/etc/my_stop_dic.txt /opt/ml/elasticsearch-7.6.2/config/user_dic/.

python3 run_elastic_search.py

ps -ef | grep elastic