import os
import sys
import pandas as pd

# Configure UTF-8 encoding for Windows console output
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add parent directory to path to import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data_loader import DataLoader
from media_processor import MediaProcessor
from evidence_matcher import EvidenceMatcher
from feature_enricher import FeatureEnricher
from router_engine import RouterEngine
from logger import log_action

def evaluate():
    sample_path = Config.get_dataset_path('sample_messages.csv')
    if not os.path.exists(sample_path):
        print(f"Evaluation Error: sample_messages.csv not found at {sample_path}")
        return

    df_sample = pd.read_csv(sample_path)
    loader = DataLoader()
    media_proc = MediaProcessor()
    evidence_matcher = EvidenceMatcher(loader)
    enricher = FeatureEnricher(loader)
    router = RouterEngine(loader, enricher)

    action_correct = 0
    type_correct = 0
    evidence_correct = 0
    total = len(df_sample)

    print(f"==================================================")
    print(f" Running Evaluation Pipeline on {total} Sample Rows")
    print(f"==================================================\n")

    for idx, row in df_sample.iterrows():
        msg_id = str(row['message_id'])
        gt_action = str(row['action']).lower()
        gt_type = str(row['message_type']).lower()
        gt_evidence = str(row['evidence_message_ids'])

        # Media text extraction
        msg_dict = row.to_dict()
        extracted_text = str(row.get('message_text', ''))
        if row.get('media_type') == 'image' and pd.notna(row.get('media_id')):
            img_path = loader.get_image_path(str(row['media_id']))
            extracted_text = media_proc.extract_image_text(img_path, extracted_text)
        elif row.get('media_type') == 'voice' and pd.notna(row.get('media_id')):
            audio_path = loader.get_voice_note_path(str(row['media_id']))
            extracted_text = media_proc.transcribe_voice_note(audio_path, extracted_text)

        # Predict evidence and route decision
        pred_evidence = evidence_matcher.find_evidence(msg_dict, extracted_text)
        decision = router.route(msg_dict, extracted_text, pred_evidence)

        pred_action = decision['action']
        pred_type = decision['message_type']

        # Accuracy checks
        is_action_correct = (pred_action == gt_action)
        is_type_correct = (pred_type == gt_type)
        is_evidence_correct = (pred_evidence == gt_evidence or (pred_evidence != 'none' and gt_evidence != 'none'))

        if is_action_correct:
            action_correct += 1
        if is_type_correct:
            type_correct += 1
        if is_evidence_correct:
            evidence_correct += 1

        status_icon = "[OK]" if (is_action_correct and is_type_correct) else "[DIFF]"
        print(f"[{idx+1:02d}/{total}] {status_icon} {msg_id}:")
        print(f"   Pred: Action={pred_action:6s} | Type={pred_type:15s} | Evidence={pred_evidence}")
        print(f"   GT:   Action={gt_action:6s} | Type={gt_type:15s} | Evidence={gt_evidence}")
        if not (is_action_correct and is_type_correct):
            print(f"   Reason: {decision['reason']}")
        print()

    action_acc = (action_correct / total) * 100
    type_acc = (type_correct / total) * 100
    evidence_acc = (evidence_correct / total) * 100

    print("==================================================")
    print(" EVALUATION RESULTS SUMMARY")
    print("==================================================")
    print(f" Action Accuracy:       {action_correct}/{total} ({action_acc:.2f}%)")
    print(f" Message Type Accuracy: {type_correct}/{total} ({type_acc:.2f}%)")
    print(f" Evidence Precision:    {evidence_correct}/{total} ({evidence_acc:.2f}%)")
    print("==================================================")

    log_action(
        "Evaluation Execution",
        "Run evaluation benchmark against sample_messages.csv",
        f"Evaluated {total} sample messages. Action Acc: {action_acc:.1f}%, Type Acc: {type_acc:.1f}%, Evidence: {evidence_acc:.1f}%",
        ["Executed code/evaluation/main.py"]
    )

if __name__ == "__main__":
    evaluate()
