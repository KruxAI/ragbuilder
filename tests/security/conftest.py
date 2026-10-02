import os

os.environ["ENABLE_ANALYTICS"] = "false"
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
