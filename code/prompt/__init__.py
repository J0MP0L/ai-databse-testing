import yaml
from pathlib import Path

__dir__ = Path(__file__).parent

prompt_files = {
    'prompt_code_agent': __dir__ / 'prompt_code.yaml',
    'prompt_eval_graph': __dir__ / 'prompt_eval.yaml',
    'prompt_mongodb_agent': __dir__ / 'prompt_mongodb.yaml',
    'prompt_supervised_agent': __dir__ / 'prompt_supervised.yaml',
    'prompt_database': __dir__ / 'prompt_database.yaml',
    'prompt_mockdata': __dir__ / 'prompt_mockdata.yaml',
}

prompts = {}
for name, path in prompt_files.items():
    with open(path, 'r', encoding='utf-8') as f:
        prompts[name] = yaml.safe_load(f)

prompt_code_agent = prompts['prompt_code_agent']['prompt_code_agent']
prompt_eval_graph = prompts['prompt_eval_graph']['prompt_eval_graph']
prompt_mongodb_agent = prompts['prompt_mongodb_agent']['prompt_mongodb_agent']
prompt_supervised_agent = prompts['prompt_supervised_agent']['prompt_supervised_agent']
prompt_database = prompts['prompt_database']['prompt_database']
prompt_mockdata = prompts['prompt_mockdata']['prompt_mockdata']