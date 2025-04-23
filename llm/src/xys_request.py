#!/usr/bin/env python3
import argparse
import json
import os
import pymysql
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm


# Prompt template for XiYanSQL-QwenCoder
nl2sqlite_template_cn = """你是一名{dialect}专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，并运用{dialect}知识生成sql语句回答【用户问题】。
【用户问题】
{question}

【数据库schema】
{db_schema}

【参考信息】
{evidence}

【用户问题】
{question}

```sql"""

# CoT prompt for chain-of-thought
cot_prompt_template = """Generate the {dialect} for the above question after thinking step by step: """


def new_directory(path):
    """
    Create a new directory if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def connect_mysql(host, user, password, database):
    """
    Connect to MySQL database.
    """
    try:
        db = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=3306
            #unix_socket="/tmp/mysql.sock"
        )
        return db
    except Exception as e:
        raise Exception(f"Failed to connect to MySQL: {e}")


def format_mysql_create_table(table_name, columns_info):
    """
    Format MySQL CREATE TABLE statement.
    """
    lines = []
    lines.append(f"CREATE TABLE {table_name}\n(")

    primary_key_defined = False

    for col in columns_info:
        column_name, data_type, nullable, key, _, _ = col
        sql_type = str.upper(data_type)
        null_type = "not null" if nullable == "NO" else "null"
        primary_key_part = (
            "primary key" if "PRI" in key and not primary_key_defined else ""
        )
        primary_key_defined = True if "PRI" in key else primary_key_defined
        column_line = (
            f"    `{column_name}` {sql_type} {null_type} {primary_key_part},".strip()
        )
        lines.append(column_line)
    lines[-1] = lines[-1].rstrip(",")
    lines.append(");")
    return "\n".join(lines)


# Predefined mapping of db_id to tables
db_table_map = {
    "debit_card_specializing": [
        "customers",
        "gasstations",
        "products",
        "transactions_1k",
        "yearmonth",
    ],
    "student_club": [
        "major",
        "member",
        "attendance",
        "budget",
        "event",
        "expense",
        "income",
        "zip_code",
    ],
    "thrombosis_prediction": ["Patient", "Examination", "Laboratory"],
    "european_football_2": [
        "League",
        "Match",
        "Player",
        "Player_Attributes",
        "Team",
        "Team_Attributes",
    ],
    "formula_1": [
        "circuits",
        "seasons",
        "races",
        "constructors",
        "constructorResults",
        "constructorStandings",
        "drivers",
        "driverStandings",
        "lapTimes",
        "pitStops",
        "qualifying",
        "status",
        "results",
    ],
    "superhero": [
        "alignment",
        "attribute",
        "colour",
        "gender",
        "publisher",
        "race",
        "superpower",
        "superhero",
        "hero_attribute",
        "hero_power",
    ],
    "codebase_community": [
        "posts",
        "users",
        "badges",
        "comments",
        "postHistory",
        "postLinks",
        "tags",
        "votes",
    ],
    "card_games": [
        "cards",
        "foreign_data",
        "legalities",
        "rulings",
        "set_translations",
        "sets",
    ],
    "toxicology": ["molecule", "atom", "bond", "connected"],
    "california_schools": ["satscores", "frpm", "schools"],
    "financial": [
        "district",
        "account",
        "client",
        "disp",
        "card",
        "loan",
        "order",
        "trans",
    ],
}


def generate_schema_prompt_mysql(db_id, mysql_host, mysql_user, mysql_password, mysql_database):
    """
    Generate schema prompt for MySQL database.
    """
    db = connect_mysql(mysql_host, mysql_user, mysql_password, mysql_database)
    cursor = db.cursor()
    tables = db_table_map.get(db_id, [])
    schemas = {}
    for table in tables:
        cursor.execute(f"DESCRIBE {mysql_database}.{table}")
        raw_schema = cursor.fetchall()
        pretty_schema = format_mysql_create_table(table, raw_schema)
        schemas[table] = pretty_schema
    schema_prompt = "\n\n".join(schemas.values())
    db.close()
    return schema_prompt


def connect_local_model(model, tokenizer, prompt, max_tokens, temperature, stop):
    """
    Function to generate response using the local model.
    """
    try:
        # Format prompt as a chat message
        message = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        # Tokenize input
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        # Generate output
        generated_ids = model.generate(
            **model_inputs,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=1024,
            temperature=0.1,
            top_p=0.8,
            do_sample=True,
        )
        # Decode generated tokens, excluding input tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    except Exception as e:
        return f"error:{e}"


def decouple_question_schema(datasets):
    """
    Extract questions, database IDs, and knowledge from datasets.
    """
    question_list = []
    db_id_list = []
    knowledge_list = []
    for i, data in enumerate(datasets):
        question_list.append(data["question"])
        db_id_list.append(data["db_id"])
        knowledge_list.append(data["evidence"])
    return question_list, db_id_list, knowledge_list


def generate_sql_file(sql_lst, output_path=None):
    """
    Save the SQL results to a file.
    """
    sql_lst.sort(key=lambda x: x[1])
    result = {}
    for i, (sql, _) in enumerate(sql_lst):
        result[i] = sql
    if output_path:
        directory_path = os.path.dirname(output_path)
        new_directory(directory_path)
        json.dump(result, open(output_path, "w"), indent=4)
    return result


def post_process_response(response, db_id):
    """
    Extract SQL from model response and format output.
    """
    # Try to extract SQL between ```sql and ``` (if present)
    sql_match = re.search(r"```sql\s*(.*?)\s*```", response, re.DOTALL)
    if sql_match:
        sql = sql_match.group(1).strip()
    else:
        # Fallback: assume response is SQL or extract first SELECT statement
        sql_lines = [line.strip() for line in response.split("\n") if line.strip().upper().startswith("SELECT")]
        sql = sql_lines[0] if sql_lines else response.strip()

    # Ensure SQL is non-empty
    if not sql:
        sql = "/* No valid SQL generated */"

    return f"{sql}\t----- bird -----\t{db_id}"


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--eval_path", type=str, default="")
    args_parser.add_argument("--mode", type=str, default="dev")
    args_parser.add_argument("--use_knowledge", type=str, default="False")
    args_parser.add_argument("--db_root_path", type=str, default="")  # Used for db_id extraction
    args_parser.add_argument("--model_path", type=str, required=True)
    args_parser.add_argument("--engine", type=str, required=True)
    args_parser.add_argument("--data_output_path", type=str)
    args_parser.add_argument("--chain_of_thought", type=str)
    args_parser.add_argument("--sql_dialect", type=str, default="MySQL")
    args_parser.add_argument("--mysql_host", type=str, default="localhost")
    args_parser.add_argument("--mysql_user", type=str, default="root")
    args_parser.add_argument("--mysql_password", type=str, required=True)
    args_parser.add_argument("--mysql_database", type=str, default="BIRD")
    args = args_parser.parse_args()

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    eval_data = json.load(open(args.eval_path, "r"))

    question_list, db_id_list, knowledge_list = decouple_question_schema(
        datasets=eval_data
    )
    assert len(question_list) == len(db_id_list) == len(knowledge_list)

    # Process questions sequentially
    responses = []
    for i in tqdm(range(len(question_list)), desc="Processing questions"):
        # Get database schema
        db_schema = generate_schema_prompt_mysql(
            db_id_list[i],
            args.mysql_host,
            args.mysql_user,
            args.mysql_password,
            args.mysql_database
        )
        # Generate prompt using nl2sqlite_template_cn
        evidence = knowledge_list[i] if args.use_knowledge == "True" else ""
        prompt = nl2sqlite_template_cn.format(
            dialect=args.sql_dialect,
            question=question_list[i],
            db_schema=db_schema,
            evidence=evidence
        )
        # Add CoT prompt if enabled
        if args.chain_of_thought == "True":
            prompt += "\n" + cot_prompt_template.format(dialect=args.sql_dialect)
        
        response = connect_local_model(
            model, tokenizer, prompt, max_tokens=512, temperature=0.1, stop=["--", "\n\n", ";", "#"]
        )
        sql = post_process_response(response, db_id_list[i])
        print(f"Processed {i}th question: {question_list[i]}")
        responses.append((sql, i))

    if args.chain_of_thought == "True":
        output_name = (
            args.data_output_path
            + "predict_"
            + args.mode
            + "_"
            + args.engine
            + "_cot"
            + "_"
            + args.sql_dialect
            + ".json"
        )
    else:
        output_name = (
            args.data_output_path
            + "predict_"
            + args.mode
            + "_"
            + args.engine
            + "_"
            + args.sql_dialect
            + ".json"
        )
    generate_sql_file(sql_lst=responses, output_path=output_name)

    print(
        "Successfully collected results from {} for {} evaluation; SQL dialect {} Use knowledge: {}; Use COT: {}".format(
            args.engine,
            args.mode,
            args.sql_dialect,
            args.use_knowledge,
            args.chain_of_thought,
        )
    )