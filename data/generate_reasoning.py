import utils
import time

def deepseek_qa(prompt: str, model: str):
    max_attempts = 1000
    answer = None 
    attempts = 0
    while answer is None and attempts < max_attempts:
        try:
            answer = utils.ask_model(prompt, model)
            attempts += 1
            thinking, answer = answer["choices"][0]["message"]["content"].split("</think>")                
        except Exception as e:
            print(f"Exception: {str(e)}")
            time.sleep(60)

    return thinking, answer

if __name__ == "__main__":
    model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    question = "An elephant and a lion are currently 1 mile apart. The elephant runs directly away from the lion at 19 miles per hour, while the lion runs directly towards the elephant at 24 miles per hour. How many minutes will it take for the lion to catch the elephant?"
    thinking, answer = deepseek_qa(question, model)
    
    print(f"thinking: {thinking}")           
    print(f"answer: {answer}")           