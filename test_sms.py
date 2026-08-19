from dotenv import load_dotenv
load_dotenv()
from utils.twilio_api import send_energy_tip_sms

result = send_energy_tip_sms('+919257136015', 'Test message from EcoGrid')
print(result)
