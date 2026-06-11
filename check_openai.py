from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads .env and loads its values into the environment

client = OpenAI()  # SDK automatically finds OPENAI_API_KEY in the environment

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Reply with exactly: QueryPilot is connected."}
    ],
)

print(response.choices[0].message.content)