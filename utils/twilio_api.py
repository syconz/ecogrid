import os
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

# Twilio trial accounts can only send one of a few fixed, pre-approved
# template bodies — NOT arbitrary custom text. This is a Twilio anti-abuse
# restriction on trial accounts, not something we can work around in code.
# "sms_account_alerts" is the closest semantic fit for an energy-tip
# notification. Once the account is upgraded off trial, this restriction
# goes away and any custom message string can be sent instead.
TRIAL_TEMPLATE_BODY = "sms_account_alerts"


def send_energy_tip_sms(to_number, message):
    """
    Sends an SMS via Twilio.

    NOTE: On a Twilio trial account, `message` is NOT actually sent —
    trial accounts can only send Twilio's fixed template bodies. We send
    TRIAL_TEMPLATE_BODY instead, and `message` is only used for logging so
    you can see what a real (post-upgrade) message would have said.

    Returns a dict: {"success": True, "sid": "..."} on success,
    or {"success": False, "error": "..."} on failure.

    `to_number` must be a phone number verified in the Twilio console
    (Console -> Phone Numbers -> Verified Caller IDs) while on a trial account.
    """
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_number = os.environ.get("TWILIO_PHONE_NUMBER")

    if not all([account_sid, auth_token, from_number]):
        return {"success": False, "error": "Twilio credentials not configured in .env"}

    if not to_number:
        return {"success": False, "error": "No destination phone number provided"}

    print(f"[twilio_api] Intended message (not actually sent on trial): {message}")

    try:
        client = Client(account_sid, auth_token)
        sms = client.messages.create(
            body=TRIAL_TEMPLATE_BODY,
            from_=from_number,
            to=to_number,
        )
        return {"success": True, "sid": sms.sid}

    except TwilioRestException as e:
        return {"success": False, "error": f"Twilio error {e.code}: {e.msg}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {e}"}