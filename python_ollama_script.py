import ollama

response = ollama.chat(model="llama3", messages=[
    {"role": "user", "content": "Explain Docker/kubernetes in simple terms."}
])

print(response['message']['content'])

