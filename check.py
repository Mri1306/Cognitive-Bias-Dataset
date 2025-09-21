from groq import Groq

client = Groq(api_key="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")

try:
    models = client.models.list()
    print("Groq Models:", [m.id for m in models.data])
except Exception as e:
    print("Error:", e)