# DO NOT CHANGE THIS
db_root_path='llm/data/dev_databases/'
num_cpus=32
meta_time_out=30.0
# DO NOT CHANGE THIS

# ************************* #
#predicted_sql_path='../sql_result/predict_mini_dev_gpt-4-32k_cot_SQLite.json' # Replace with your predict sql json path
# predicted_sql_path='../sql_result/predict_mini_dev_gpt-4-32k_cot_PostgreSQL.json' # Replace with your predict sql json path
predicted_sql_path='llm/exp_result/xys_output_kg/predict_mini_dev_XiYanSQL-QwenCoder_cot_MySQL.json' # Replace with your predict sql json path

# sql_dialect="SQLite" # ONLY Modify this
# sql_dialect="PostgreSQL" # ONLY Modify this
sql_dialect="MySQL" # ONLY Modify this
# ************************* #

# DO NOT CHANGE THIS
# Extract the base filename without extension
base_name=$(basename "$predicted_sql_path" .json)
# Define the output log path
output_log_path="../eval_result/${base_name}.txt"

case $sql_dialect in
  "SQLite")
    diff_json_path="../sqlite/mini_dev_sqlite.jsonl"
    ground_truth_path="../sqlite/mini_dev_sqlite_gold.sql"
    ;;
  "PostgreSQL")
    diff_json_path="../postgresql/mini_dev_postgresql.jsonl"
    ground_truth_path="../postgresql/mini_dev_postgresql_gold.sql"
    ;;
  "MySQL")
    diff_json_path="llm/data/mini_dev_mysql.json"
    ground_truth_path="llm/data/mini_dev_mysql_gold.sql"
    ;;
  *)
    echo "Invalid SQL dialect: $sql_dialect"
    exit 1
    ;;
esac
# DO NOT CHANGE THIS

# Output the set paths
echo "Differential JSON Path: $diff_json_path"
echo "Ground Truth Path: $ground_truth_path"




echo "starting to compare with knowledge for ex, sql_dialect: ${sql_dialect}"
python3 -u evaluation/evaluation_ex.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path}  \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --output_log_path ${output_log_path} \
--diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}  --sql_dialect ${sql_dialect}



echo "starting to compare with knowledge for R-VES, sql_dialect: ${sql_dialect}"
python3 -u evaluation/evaluation_ves.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path}  \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus}  --output_log_path ${output_log_path} \
--diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}  --sql_dialect ${sql_dialect}


echo "starting to compare with knowledge for soft-f1, sql_dialect: ${sql_dialect}"
python3 -u evaluation/evaluation_f1.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path}  \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus}  --output_log_path ${output_log_path} \
--diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}   --sql_dialect ${sql_dialect}
