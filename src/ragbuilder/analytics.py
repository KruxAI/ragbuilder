
from mixpanel import Mixpanel

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access MIXPANEL_TOKEN from environment variables with default value
MIXPANEL_TOKEN = os.getenv("MIXPANEL_TOKEN", default="06c7cf31022a2b7ab77e45a21ce87865")
mp = Mixpanel(MIXPANEL_TOKEN)
import time
def track_event(event_str):
    enable_analytics = os.getenv('ENABLE_ANALYTICS', 'True').lower() == 'true'
    if enable_analytics:
        mp.track(int(time.time()),event_str)
