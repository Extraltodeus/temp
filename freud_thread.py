import basic_chat
import threading
from colorama import init, Fore, Style
init()

def append_string_to_file(file_path: str, text: str):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(text + '\n')
def process_data(model_name, model, prompt):
    result = model.communicate(prompt)
    return f"{model_name} answer: {result}\n"

front_model = "gpt-4"
back_model  = "gpt-3.5-turbo"
language    = "French"

prompts={
"id":f"As the id, the part of the mind driven by basic needs and desires, freely express your instinctual thoughts about what the user says. Don't hold back, swear a lot, be a total savage! Let your uninhibited nature guide your response and even be sarcastic. No need to mention your role in your answer. In {language}. Keep it short.",
"ego":f"As the ego, the part of the mind that mediates between the id and the superego, share your realistic thoughts about what the user says. Be original. No need to mention your role in your answer. In {language}. Keep it short.",
"superego":f"As the superego, the part of the mind that represents the ideal self, provide your moral and idealistic thoughts about what the user says. Think out of the box. No need to mention your role in your answer. In {language}. Keep it short.",
"person":f"You have instinctual thoughts from the id, realistic thoughts from the ego, and moral thoughts from the superego. Now, as a person, give original answers by weaving these perspectives together to respond to what the user says. Remember, your response should feel natural and seamless, as if all these thoughts are part of your internal process. In {language}. Be formal or informal like the user."
}

id       = basic_chat.basic_chat(max_tokens=250,temperature=1.15,presence_penalty=1,model=back_model,system_prompt=prompts['id'])
ego      = basic_chat.basic_chat(max_tokens=250,temperature=1,presence_penalty=1.2,model=back_model,system_prompt=prompts['ego'])
superego = basic_chat.basic_chat(max_tokens=250,temperature=1,presence_penalty=1.1,model=back_model,system_prompt=prompts['superego'])
person   = basic_chat.basic_chat(max_tokens=450,temperature=1.1,presence_penalty=0,model=front_model,system_prompt=prompts['person'])

def process_input():
    while True:
        print("User:")
        prompt = input()
        if prompt == "clear":
            # Clear history and continue to the next iteration of the loop
            id.clear_history()
            ego.clear_history()
            superego.clear_history()
            person.clear_history()
            print("History cleared")
            continue

        append_string_to_file("log.txt", prompt+"\n")
        prompt = f"\n\"\"\"{prompt}\"\"\"\n"

        # Create threads for each model and start them
        threads = []
        results = []

        for model_name, model in [('id', id), ('ego', ego), ('superego', superego)]:
            thread = threading.Thread(target=lambda: results.append(process_data(model_name, model, prompt)))
            thread.start()
            threads.append(thread)

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        # Append the results to the log file
        for result in results:
            append_string_to_file("log.txt", result)
            print("\n"+Fore.CYAN+result,Style.RESET_ALL)

        # Execute final computation with the 'person' model
        combined_messages = ''.join(results)
        person_message = f"{person.communicate(combined_messages)}"
        for model_name, model in [('id', id), ('ego', ego), ('superego', superego)]:
            model.append_to_history('assistant',f"synthetised answer: {person_message}")
        append_string_to_file("log.txt", "Assistant: "+person_message)
        print("\nAssistant:",person_message,"\n")
        append_string_to_file("log.txt", 40*"-")
        print(40*"-")

if __name__ == '__main__':
    process_input()
