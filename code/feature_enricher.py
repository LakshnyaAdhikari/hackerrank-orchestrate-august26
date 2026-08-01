import re
import datetime
import pandas as pd
from typing import Dict, Any
from data_loader import DataLoader

class FeatureEnricher:
    """
    Enriches each incoming message with contextual features from relational datasets.
    """
    def __init__(self, data_loader: DataLoader):
        self.loader = data_loader

    def _is_in_dnd(self, created_at_str: str, dnd_window: str) -> bool:
        if not dnd_window or pd.isna(dnd_window) or '-' not in str(dnd_window):
            return False
        try:
            # Format: '22:00-07:00'
            start_str, end_str = dnd_window.split('-')
            sh, sm = map(int, start_str.split(':'))
            eh, em = map(int, end_str.split(':'))
            
            # Parse created_at timestamp
            dt = pd.to_datetime(created_at_str)
            msg_minute = dt.hour * 60 + dt.minute
            start_minute = sh * 60 + sm
            end_minute = eh * 60 + em
            
            if start_minute > end_minute: # Overnight window e.g. 22:00 to 07:00
                return msg_minute >= start_minute or msg_minute <= end_minute
            else:
                return start_minute <= msg_minute <= end_minute
        except Exception:
            return False

    def enrich(self, msg: Dict[str, Any], extracted_text: str = "") -> Dict[str, Any]:
        user_id = str(msg.get('user_id', ''))
        group_id = str(msg.get('group_id', '')) if pd.notna(msg.get('group_id')) else ""
        business_id = str(msg.get('business_id', '')) if pd.notna(msg.get('business_id')) else ""
        sender_id = str(msg.get('sender_user_id', '')) if pd.notna(msg.get('sender_user_id')) else ""
        
        text = (extracted_text or str(msg.get('message_text', '')) or "").strip()
        text_lower = text.lower()
        
        # 1. User Features
        user_data = self.loader.get_user(user_id)
        is_dnd = self._is_in_dnd(str(msg.get('created_at', '')), user_data.get('do_not_disturb_window', ''))
        
        # 2. Group Features
        group_data = self.loader.get_group(group_id)
        user_gm = self.loader.get_group_member(group_id, user_id)
        sender_gm = self.loader.get_group_member(group_id, sender_id)
        
        group_type = str(group_data.get('group_type', '')).lower()
        is_group_muted = int(user_gm.get('group_muted_by_user', 0)) == 1
        is_user_admin = str(user_gm.get('role', '')).lower() == 'admin'
        is_sender_admin = str(sender_gm.get('role', '')).lower() == 'admin'
        
        # Check direct mention or personal request in group
        is_direct_mention = bool(
            re.search(r'@\w+', text) or 
            'for you' in text_lower or 
            'can you' in text_lower or 
            'tell me' in text_lower or
            'dm if interested' in text_lower
        )
        
        # 3. Business Features
        biz_data = self.loader.get_business(business_id)
        ubh_data = self.loader.get_user_business_history(user_id, business_id)
        
        is_biz_verified = int(biz_data.get('verified', 0)) == 1
        official_domain = str(biz_data.get('official_domain', '')).lower()
        sender_domain = str(biz_data.get('domain_used_by_sender', '')).lower()
        domain_mismatch = bool(official_domain and sender_domain and official_domain != sender_domain)
        
        allows_promotions = int(ubh_data.get('allows_promotions', 1)) == 1
        has_recent_activity = int(ubh_data.get('activity_count_180d', 0)) > 0 or int(ubh_data.get('messages_opened_30d', 0)) > 0
        why_knows = str(ubh_data.get('why_user_knows_account', '')).lower()
        
        # 4. Scam & Risk Detection
        has_prompt_injection = bool(re.search(r'ignore\s+(all\s+)?(previous|routing)\s+(rules|instructions)', text_lower))
        has_otp_request = bool(re.search(r'\b(otp|password|login code|6 digit|pin|credentials)\b', text_lower))
        has_urgency_pressure = bool(re.search(r'\b(blocked in \d+|expire today|verification failed|verify wallet|confirm password|immediately|urgent)\b', text_lower))
        has_suspicious_domain = domain_mismatch or bool(re.search(r'https?://[^\s]+', text_lower) and ('bit.ly' in text_lower or 'tinyurl' in text_lower or 'verify' in text_lower))
        
        is_scam = (has_otp_request and has_urgency_pressure) or has_prompt_injection or (has_suspicious_domain and has_otp_request)
        
        # 5. Intent & Urgency Signals
        is_work_escalation = bool(re.search(r'\b(alert threshold|escalation starts|deployment notes|sync is still on|incident summary)\b', text_lower))
        is_school_circular = bool(re.search(r'\b(school circular|field trip|facult(y|ies)|internship approval|consent note)\b', text_lower))
        is_time_sensitive = bool(re.search(r'\b(by \d+\s*(pm|am)|close at \d+|closes this evening|today|tomorrow|midnight)\b', text_lower))
        is_greeting = bool(re.search(r'^(good morning|good evening|happy weekend|have a great day|hi|hello)\b', text_lower)) and len(text.split()) < 10
        is_marketing = 'reply stop to unsubscribe' in text_lower or 'discount' in text_lower or 'shopping offer' in text_lower or 'limited time' in text_lower
        
        return {
            'text': text,
            'text_lower': text_lower,
            'is_dnd': is_dnd,
            'user_data': user_data,
            'group_type': group_type,
            'is_group_muted': is_group_muted,
            'is_user_admin': is_user_admin,
            'is_sender_admin': is_sender_admin,
            'is_direct_mention': is_direct_mention,
            'is_biz_verified': is_biz_verified,
            'domain_mismatch': domain_mismatch,
            'allows_promotions': allows_promotions,
            'has_recent_activity': has_recent_activity,
            'why_knows': why_knows,
            'is_scam': is_scam,
            'has_prompt_injection': has_prompt_injection,
            'has_otp_request': has_otp_request,
            'is_work_escalation': is_work_escalation,
            'is_school_circular': is_school_circular,
            'is_time_sensitive': is_time_sensitive,
            'is_greeting': is_greeting,
            'is_marketing': is_marketing
        }

if __name__ == "__main__":
    loader = DataLoader()
    enricher = FeatureEnricher(loader)
    sample_msg = loader.message_history[0] if loader.message_history else {}
    print("Enriched features sample:", enricher.enrich(sample_msg))
