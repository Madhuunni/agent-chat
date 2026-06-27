import copy
import json
import random
import re
from pathlib import Path


URL_PATTERN = re.compile(r"https?://[^\s]+", re.IGNORECASE)
SIMPLE_NAVIGATION_PATTERN = re.compile(
    r"^\s*(?:navigate\s+to|open|go\s+to|launch(?:\s+the\s+application\s+at)?|visit)\s+"
    r"(?P<url>https?://[^\s]+)\s*$",
    re.IGNORECASE,
)


def extract_url(sentence):
    """Return the first URL in a user instruction, stripped of sentence punctuation."""
    match = URL_PATTERN.search(sentence)
    if not match:
        return None
    return match.group(0).rstrip(".!?,;:)\"'")


def is_navigation_instruction(sentence):
    """Detect simple browser-navigation requests before model classification."""
    match = SIMPLE_NAVIGATION_PATTERN.match(sentence)
    if not match:
        return False
    return extract_url(match.group("url")) is not None


def navigation_response(url):
    """Build a Selenium navigation JSON response for any URL found at runtime."""
    return {
        "test_name": "Dynamic URL Navigation Test",
        "base_url": url,
        "steps": [
            {
                "step_no": 1,
                "action": "navigate",
                "element": None,
                "by": None,
                "selector": None,
                "value": url,
                "value_from_env": None,
                "description": "Navigate to the application URL",
            }
        ],
    }


def apply_dynamic_url(response, sentence):
    """Copy an intent response and replace any navigation URL with the user's URL."""
    url = extract_url(sentence)
    if not url:
        return response

    response = copy.deepcopy(response)
    response["base_url"] = url
    for step in response.get("steps", []):
        if step.get("action") == "navigate":
            step["value"] = url
    return response


def format_response(response):
    """Print dictionaries as valid JSON while leaving text responses unchanged."""
    if isinstance(response, (dict, list)):
        return json.dumps(response, indent=2)
    return str(response)


def load_intents(path="intents.json"):
    with open(path, "r") as json_data:
        return json.load(json_data)


def load_model(file_path="data.pth"):
    import torch

    from model import NeuralNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(file_path, map_location=device)

    model = NeuralNet(data["input_size"], data["hidden_size"], data["output_size"]).to(device)
    model.load_state_dict(data["model_state"])
    model.eval()
    return model, data["all_words"], data["tags"]


def predict_intent(sentence, model, all_words, tags):
    import torch

    from nltk_utils import bag_of_words, tokenize

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentence_tokens = tokenize(sentence)
    X = bag_of_words(sentence_tokens, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    return tag, prob.item()


if __name__ == "__main__":
    intents = load_intents()

    bot_name = "Sam"
    model = None
    all_words = None
    tags = None
    model_file = Path("data.pth")
    if model_file.exists():
        model, all_words, tags = load_model(model_file)

    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        if is_navigation_instruction(sentence):
            print(f"{bot_name}: {format_response(navigation_response(extract_url(sentence)))}")
            continue

        if model is None:
            print(f"{bot_name}: I do not understand...")
            continue

        tag, prob = predict_intent(sentence, model, all_words, tags)
        if prob > 0.75:
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    response = random.choice(intent["responses"])
                    response = apply_dynamic_url(response, sentence)
                    print(f"{bot_name}: {format_response(response)}")
                    break
        else:
            print(f"{bot_name}: I do not understand...")
