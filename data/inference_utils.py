import time
import anthropic
from dotenv import load_dotenv
import os


def _claudeqa(prompt: str, system_message: str):
    load_dotenv()
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=8192,
        system=system_message,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text, message.usage


def apiqa(prompt: str, model_name: str, system_message: str, json_format: bool = True):
    completion = None
    while completion is None:
        try:
            if model_name == "claude-3-5-sonnet-20241022":
                assert not json_format, "Claude does not support json format"
                completion, usage = _claudeqa(prompt, system_message)
        except Exception as e:
            print(f"Exception: {str(e)}")
            time.sleep(60)

    return completion, usage
