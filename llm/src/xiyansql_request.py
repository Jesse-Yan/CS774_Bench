#!/usr/bin/env python3
import argparse
import json
import os
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

from prompt import generate_combined_prompts_one

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def new_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def connect_gpt(engine, prompt, max_tokens, temperature, stop, client):
    """
    Function to connect to the GPT API and get the response.
    """
    MAX_API_RETRY = 10
    for i in range(MAX_API_RETRY):
        time.sleep(2)
        try:

            if engine == "gpt-35-turbo-instruct":
                result = client.completions.create(
                    model="gpt-3.5-turbo-instruct",
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                )
                result = result.choices[0].text
            else:  # gpt-4-turbo, gpt-4, gpt-4-32k, gpt-35-turbo
                messages = [
                    {"role": "user", "content": prompt},
                ]
                result = client.chat.completions.create(
                    model=engine,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                )
            break
        except Exception as e:
            result = "error:{}".format(e)
            print(result)
            time.sleep(4)
    return result


def decouple_question_schema(datasets, db_root_path):
    question_list = []
    db_path_list = []
    knowledge_list = []
    for i, data in enumerate(datasets):
        question_list.append(data["question"])
        cur_db_path = db_root_path + data["db_id"] + "/" + data["db_id"] + ".sqlite"
        db_path_list.append(cur_db_path)
        knowledge_list.append(data["evidence"])

    return question_list, db_path_list, knowledge_list


def generate_sql_file(sql_lst, output_path=None):
    """
    Function to save the SQL results to a file.
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


def post_process_response(response, db_path):
    sql = response if isinstance(response, str) else response.choices[0].message.content
    db_id = db_path.split("/")[-1].split(".sqlite")[0]
    sql = f"{sql}\t----- bird -----\t{db_id}"
    return sql


def worker_function(question_data):
    """
    Function to process each question, set up the client,
    generate the prompt, and collect the GPT response.
    """
    prompt, engine, client, db_path, question, i = question_data
    response = connect_gpt(engine, prompt, 512, 0, ["--", "\n\n", ";", "#"], client)
    sql = post_process_response(response, db_path)
    print(f"Processed {i}th question: {question}")
    return sql, i


def collect_response_from_xys(
    db_path_list,
    question_list,
    engine,
    sql_dialect,
    knowledge_list=None,
):
    """
    Collect responses from XiYanSQL-QwenCoder using multiple threads.
    """
    
    tasks = [
        (
            generate_combined_prompts_one(
                db_path=db_path_list[i],
                question=question_list[i],
                sql_dialect=sql_dialect,
                knowledge=knowledge_list[i],
            ),
            engine,
            client,
            db_path_list[i],
            question_list[i],
            i,
        )
        for i in range(len(question_list))
    ]
    responses = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_task = {
            executor.submit(worker_function, task): task for task in tasks
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_task), total=len(tasks)
        ):
            responses.append(future.result())
    return responses


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--eval_path", type=str, default="")
    args_parser.add_argument("--model_path", type=str, default="")
    args_parser.add_argument("--mode", type=str, default="dev")
    args_parser.add_argument("--test_path", type=str, default="")
    args_parser.add_argument("--use_knowledge", type=str, default="False")
    args_parser.add_argument("--db_root_path", type=str, default="")
    args_parser.add_argument(
        "--engine", type=str, required=True, default="code-davinci-002"
    )
    args_parser.add_argument("--data_output_path", type=str)
    args_parser.add_argument("--chain_of_thought", type=str)
    args_parser.add_argument("--sql_dialect", type=str, default="SQLite")
    args = args_parser.parse_args()

    eval_data = json.load(open(args.eval_path, "r"))

    question_list, db_path_list, knowledge_list = decouple_question_schema(
        datasets=eval_data, db_root_path=args.db_root_path
    )
    assert len(question_list) == len(db_path_list) == len(knowledge_list)

    model_path = args.model_path
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if args.use_knowledge == "True":
        responses = collect_response_from_xys(
            db_path_list,
            question_list,
            args.engine,
            args.sql_dialect,
            knowledge_list,
        )
    else:
        responses = collect_response_from_xys(
            db_path_list,
            question_list,
            args.engine,
            args.sql_dialect,
        )

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
        "successfully collect results from {} for {} evaluation; SQL dialect {} Use knowledge: {}; Use COT: {}".format(
            args.engine,
            args.mode,
            args.sql_dialect,
            args.use_knowledge,
            args.chain_of_thought,
        )
    )