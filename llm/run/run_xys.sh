eval_path='./data/mini_dev_mysql.json' # _sqlite.json, _mysql.json, _postgresql.json
dev_path='./output/'
db_root_path='./data/dev_databases/'
use_knowledge='True'
mode='mini_dev' # dev, train, mini_dev
cot='True'

# Choose the engine to run XiYanSQL-QwenCoder
engine='XiYanSQL-QwenCoder'
model_path='/data/XiYanSQL-QwenCoder-3B-2502'

# Choose the SQL dialect to run
sql_dialect='MySQL'

# MySQL connection parameters
mysql_host='localhost'
mysql_user='root'
mysql_password='cs774'  # Replace with actual password
mysql_database='BIRD'

# Choose the output path for the generated SQL queries
data_output_path='./exp_result/xys_output/'
data_kg_output_path='./exp_result/xys_output_kg/'

echo "generate $engine batch, with knowledge: $use_knowledge, with chain of thought: $cot"
python3 -u ./src/xys_request.py --db_root_path ${db_root_path} --mode ${mode} --model_path ${model_path} \
--engine ${engine} --eval_path ${eval_path} --data_output_path ${data_kg_output_path} --use_knowledge ${use_knowledge} \
--chain_of_thought ${cot} --sql_dialect ${sql_dialect} \
--mysql_host ${mysql_host} --mysql_user ${mysql_user} --mysql_password ${mysql_password} --mysql_database ${mysql_database}