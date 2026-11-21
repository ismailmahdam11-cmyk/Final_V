from llm_sdk import Small_LLM_Model
from pathlib import Path
import numpy as np 
import json


def load_json(path, key):
    with open(path, "r") as f:
        return [d.get(key) for d in json.load(f)]

def token_tree(model, names):
    root = {}
    for n in names:
        node = root
        for tid in model.encode(n).tolist()[0]:
            node = node.setdefault(tid , {})
        node["end"] = n
    return root

def context_prompt(names, prompt):
    return(
        "you are a function calling assitant\n" \
        f"here is the prompt: {prompt}\n" \
        "your job is pick the correct function name\n" \
        f"here is the list you should pick from: {names}\n" \
        "Answer: "
    )

def build_dict(model, trie, prompt, names):
    input_ids = model.encode(context_prompt(names, prompt)).tolist()[0]
    node = trie

    while True:
        logits = np.array(model.get_logits_from_input_ids(input_ids))
        valid_ids = [tid for tid in node if node != "end"]
        mask = np.full_like(logits, -np.inf)
        mask[valid_ids] = logits[valid_ids]
        pred_tid = np.argmax(mask)
        node = node[pred_tid]
        if "end" in node:
            break
        input_ids.append(pred_tid)

    return {
        "prompt" : prompt,
        "name" : node["end"]
    }

def get_value(model, prompt, key, typ):
    start = '"' if typ == "string" else ""
    ids = model.encode(f'Request: {prompt}\n{{"{key}": {start}').tolist()[0]
    text = ""
    while len(text) < 50:
        tid = int(np.argmax(model.get_logits_from_input_ids(ids)))
        tok = model.decode([tid])
        if typ == "string" and '"' in tok:
            text += tok.split('"')[0]
            break
        if typ == "number" and not (tok.isdigit() or tok == "."):
            break
        text += tok
        ids.append(tid)
    return float(text) if typ == "number" else text

def example_request():
    return (
  'Request: What is 4 plus 7?\n{"a": 4, "b": 7}\n'
  'Request: Greet bob\n{"name": "bob"}\n'
  'Request: Replace all digits in "a1b2" with X\n'
  '{"source_string": "a1b2", "regex": "[0-9]", "replacement": "X"}\n'
  'Request: '
)

def main():
    print("test1")
    prompts = load_json("data/input/function_calling_tests.json", "prompt")
    names = load_json("data/input/functions_definition.json", "name")
    params = load_json("data/input/functions_definition.json", "parameters")
    model = Small_LLM_Model()

    defs = dict(zip(names, params))
    trie = token_tree(model, names)
    result = [build_dict(model, trie, p, names) for p in prompts]

    for r in result:
      examples = example_request()
      p = examples + r["prompt"] + "\nFunction: " + r["name"]
      r["parameters"] = {k: get_value(model, p, k, v["type"]) for k, v in defs[r["name"]].items()}

    output_path = Path("data/output2/output13.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)
    print("test2")
if __name__ == "__main__":
  main()
