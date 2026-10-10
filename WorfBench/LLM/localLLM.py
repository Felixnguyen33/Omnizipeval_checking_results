import requests
import json

def localLLM(messages,model="gpt-3.5-turbo",temperature=0.5,top_p=1,max_tokens=1024,api_key="api_key",api_port=8000):
    url = f'http://0.0.0.0:{api_port}/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }
    try:

        json_data = json.dumps(data)
        response = requests.post(url, headers=headers, data=json_data)
        
        if response.status_code == 200:
            response = response.json()
            return response['choices'][0]['message']['content']
        else:
            return {"error": f"HTTP error {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}
if __name__ == '__main__':

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Why is the sky blue?"},
    ]
    api_key = "api_key"
    response = localLLM(messages, api_key=api_key)
    print(response) 
