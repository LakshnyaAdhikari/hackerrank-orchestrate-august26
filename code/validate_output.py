import os
import sys
import pandas as pd
from config import Config

ALLOWED_ACTIONS = {'notify', 'digest', 'mute'}
ALLOWED_MESSAGE_TYPES = {
    'personal', 'urgent', 'event', 'payment', 'business_update', 
    'promotion', 'greeting', 'forward', 'spam', 'scam', 'unknown'
}
REQUIRED_COLUMNS = ['message_id', 'action', 'message_type', 'reason', 'confidence', 'evidence_message_ids']

def validate(output_csv_path: str = None) -> bool:
    path = output_csv_path or Config.get_dataset_path('output.csv')
    if not os.path.exists(path):
        print(f"Validation Error: Output file not found at {path}")
        return False

    df = pd.read_csv(path)
    
    # 1. Column Order & Names
    if list(df.columns) != REQUIRED_COLUMNS:
        print(f"Validation Error: Column mismatch.\nExpected: {REQUIRED_COLUMNS}\nGot: {list(df.columns)}")
        return False

    # 2. Row Count
    msg_path = Config.get_dataset_path('messages.csv')
    if os.path.exists(msg_path):
        df_msg = pd.read_csv(msg_path)
        expected_rows = len(df_msg)
        if len(df) != expected_rows:
            print(f"Validation Error: Row count mismatch. Expected {expected_rows}, got {len(df)}")
            return False

    # 3. Enum & Value Checking
    errors = []
    for idx, row in df.iterrows():
        msg_id = row['message_id']
        action = row['action']
        msg_type = row['message_type']
        conf = row['confidence']
        ev_ids = row['evidence_message_ids']
        reason = row['reason']

        if action not in ALLOWED_ACTIONS:
            errors.append(f"Row {idx} ({msg_id}): Invalid action '{action}'")

        if msg_type not in ALLOWED_MESSAGE_TYPES:
            errors.append(f"Row {idx} ({msg_id}): Invalid message_type '{msg_type}'")

        try:
            conf_val = float(conf)
            if not (0.0 <= conf_val <= 1.0):
                errors.append(f"Row {idx} ({msg_id}): Confidence {conf_val} out of range [0.0, 1.0]")
        except (ValueError, TypeError):
            errors.append(f"Row {idx} ({msg_id}): Invalid confidence value '{conf}'")

        if pd.isna(reason) or not str(reason).strip():
            errors.append(f"Row {idx} ({msg_id}): Empty reason string")

        if pd.isna(ev_ids) or not str(ev_ids).strip():
            errors.append(f"Row {idx} ({msg_id}): Empty evidence_message_ids string (should be 'none' if empty)")

    if errors:
        print(f"Validation Failed with {len(errors)} errors:")
        for err in errors[:10]:
            print(" -", err)
        return False

    print(f"✅ Validation Passed Successfully! {len(df)} predictions verified against schema.")
    return True

if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
