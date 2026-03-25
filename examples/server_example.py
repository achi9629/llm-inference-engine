from llm_engine import Client

client = Client("http://localhost:8000")

print("Health:", client.health())

result = client.generate("The meaning of life is", 50)

print("Request ID:", result["request_id"])
print("Generated Text:", result["generated_text"])
print("Token Count:", result["token_count"])
print("Stop Reason:", result["stop_reason"])