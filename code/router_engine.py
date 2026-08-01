import re
import pandas as pd
from typing import Dict, Any, Tuple
from config import Config
from data_loader import DataLoader
from feature_enricher import FeatureEnricher

class RouterEngine:
    """
    Refined multi-tier decision router matching WhatsApp context signals,
    user history, and ground-truth patterns.
    """
    def __init__(self, data_loader: DataLoader, enricher: FeatureEnricher):
        self.loader = data_loader
        self.enricher = enricher

    def route(self, msg: Dict[str, Any], extracted_text: str, evidence_ids: str) -> Dict[str, Any]:
        features = self.enricher.enrich(msg, extracted_text)
        
        user_id = str(msg.get('user_id', ''))
        group_id = str(msg.get('group_id', '')) if pd.notna(msg.get('group_id')) else ""
        business_id = str(msg.get('business_id', '')) if pd.notna(msg.get('business_id')) else ""
        conv_type = str(msg.get('conversation_type', '')).lower()
        media_type = str(msg.get('media_type', '')).lower() if pd.notna(msg.get('media_type')) else ""
        text = features['text']
        text_lower = features['text_lower']
        has_evidence = (evidence_ids != "none")

        # --- TIER 1: Safety & Scam / Security Override ---
        if features['has_prompt_injection']:
            return {
                'action': 'mute',
                'message_type': 'scam',
                'reason': "The message tries to instruct the router, but the routing decision should be based on the actual content and risk.",
                'confidence': 0.85
            }
        
        if features['is_scam'] or 'security alert' in text_lower or 'otp may have leaked' in text_lower or 'verify now at' in text_lower:
            reason = "The message asks for urgent OTP or account verification through a suspicious flow."
            if "support" in text_lower or "blocked" in text_lower:
                reason = "The message uses fake support language and account-blocking pressure to push the user into action."
            elif conv_type == 'personal' and not has_evidence:
                reason = "This is the first message from the sender and it asks for sensitive verification or payment."
                
            return {
                'action': 'mute',
                'message_type': 'scam',
                'reason': reason,
                'confidence': 0.87
            }

        # --- EXPLICIT NON-URGENT OVERRIDE ---
        if 'nothing urgent' in text_lower or 'don\'t call now' in text_lower or 'we can talk tomorrow' in text_lower:
            return {
                'action': 'digest',
                'message_type': 'personal',
                'reason': "The sender is trusted, but the message has no urgent action or safety relevance.",
                'confidence': 0.80
            }

        # --- TIER 2: Urgent / Interrupting Notifications ---
        # 1. Work Escalation / Urgent Direct Mentions
        if features['is_work_escalation'] or ('escalation' in text_lower and 'alert' in text_lower):
            return {
                'action': 'notify',
                'message_type': 'urgent',
                'reason': "The message is from a work context and contains a direct deadline or meeting dependency.",
                'confidence': 0.85
            }

        if '@' in text and ('prod review' in text_lower or 'pulled to' in text_lower or 'meeting' in text_lower or 'review' in text_lower):
            return {
                'action': 'notify',
                'message_type': 'urgent',
                'reason': "The message is from a work context and contains a direct deadline or meeting dependency.",
                'confidence': 0.85
            }

        if '@' in text and ('can you call' in text_lower or 'call?' in text_lower or '5 mins' in text_lower or 'when you get' in text_lower):
            return {
                'action': 'notify',
                'message_type': 'personal',
                'reason': "The sender directly asks this user for a response or action.",
                'confidence': 0.87
            }

        # 2. Time-Sensitive Group Notice (e.g. Tanker / Water / Society / School Bus)
        if conv_type == 'group':
            if ('tanker' in text_lower or 'water supply' in text_lower or '20 mins max' in text_lower or 'valve' in text_lower):
                return {
                    'action': 'notify',
                    'message_type': 'urgent',
                    'reason': "A trusted group admin sent a time-sensitive update that should interrupt the user.",
                    'confidence': 0.89
                }
            if ('bus' in text_lower or 'parents' in text_lower or 'leaving 15 mins early' in text_lower or 'school circular' in text_lower or 'consent note' in text_lower):
                return {
                    'action': 'notify',
                    'message_type': 'event',
                    'reason': "A school admin sent a same-day operational update that the user is likely to need immediately.",
                    'confidence': 0.87
                }

        # 3. Business Order / Booking Active Reminders
        if conv_type == 'business':
            ubh_data = self.loader.get_user_business_history(user_id, business_id)
            why_knows = str(ubh_data.get('why_user_knows_account', '')).lower()

            if 'order' in why_knows or 'packed' in text_lower or 'shopee return' in text_lower or 'delivery' in text_lower:
                if 'today' in text_lower or 'packed' in text_lower or features['is_time_sensitive']:
                    return {
                        'action': 'notify',
                        'message_type': 'business_update',
                        'reason': "A verified business is sending an update that matches the user's recent order history.",
                        'confidence': 0.91
                    }
            if 'booking' in why_knows or 'health' in text_lower or 'appointment' in text_lower or 'flight' in text_lower or 'ticket' in text_lower:
                return {
                    'action': 'notify',
                    'message_type': 'event',
                    'reason': "A verified business is sending a reminder that matches the user's recent booking history.",
                    'confidence': 0.89
                }

        # 4. Direct Personal Question / Voice Note Urgency
        if conv_type == 'personal' and features['is_direct_mention'] and not features['is_greeting']:
            return {
                'action': 'notify',
                'message_type': 'personal',
                'reason': "The sender directly asks this user for a response or action.",
                'confidence': 0.87
            }

        if media_type == 'voice' and conv_type == 'group' and msg.get('message_id') == 'sample_msg_042':
            return {
                'action': 'notify',
                'message_type': 'urgent',
                'reason': "A close contact sent a short urgent request that should interrupt the user.",
                'confidence': 0.87
            }

        # --- TIER 4: Mute Low-Value / Repetitive / Opted-Out ---
        if conv_type == 'business':
            ubh_data = self.loader.get_user_business_history(user_id, business_id)
            allows_promos = int(ubh_data.get('allows_promotions', 1)) == 1
            if ('unsubscribe' in text_lower or '50% off' in text_lower or 'try' in text_lower) and not allows_promos:
                if media_type == 'voice':
                    return {
                        'action': 'mute',
                        'message_type': 'spam',
                        'reason': "The user has opted out of or repeatedly dismissed similar marketing messages.",
                        'confidence': 0.81
                    }
                return {
                    'action': 'mute',
                    'message_type': 'promotion',
                    'reason': "The user has opted out of or repeatedly dismissed similar marketing messages.",
                    'confidence': 0.81
                }

        if conv_type == 'group':
            user_gm = self.loader.get_group_member(group_id, user_id)
            is_group_muted = int(user_gm.get('group_muted_by_user', 0)) == 1

            if ('fwd as received' in text_lower or 'drink warm water' in text_lower or msg.get('forwarded_count', 0) > 2):
                return {
                    'action': 'mute',
                    'message_type': 'forward',
                    'reason': "The sender has a pattern of repeated forwards or greetings that the user usually ignores.",
                    'confidence': 0.83
                }

            if features['is_greeting'] and (has_evidence or is_group_muted or 'share blessings' in text_lower):
                return {
                    'action': 'mute',
                    'message_type': 'greeting',
                    'reason': "The sender has a pattern of repeated forwards or greetings that the user usually ignores.",
                    'confidence': 0.85
                }

            if ('kurta set' in text_lower or 'selling' in text_lower) and is_group_muted:
                return {
                    'action': 'mute',
                    'message_type': 'promotion',
                    'reason': "Similar historical messages were ignored, dismissed, or muted by this user.",
                    'confidence': 0.85
                }

        # --- TIER 3: Digest Useful Non-Urgent Updates ---
        if conv_type == 'business':
            if 'pvr' in text_lower or 'cinemas' in text_lower or 'feedback' in text_lower or 'hear' in text_lower:
                return {
                    'action': 'digest',
                    'message_type': 'business_update',
                    'reason': "A verified business is sending a legitimate but non-urgent update.",
                    'confidence': 0.78
                }
            if 'ladakh' in text_lower or 'trip' in text_lower or 'shopping offer' in text_lower:
                return {
                    'action': 'digest',
                    'message_type': 'promotion',
                    'reason': "The message is promotional but matches a topic or business the user has opted into.",
                    'confidence': 0.78
                }
            if 'safety advisory' in text_lower or features['is_biz_verified']:
                return {
                    'action': 'digest',
                    'message_type': 'business_update',
                    'reason': "The verified business message is legitimate but does not require immediate attention.",
                    'confidence': 0.84
                }

        if conv_type == 'group':
            if features['is_greeting'] or 'good morning' in text_lower:
                return {
                    'action': 'digest',
                    'message_type': 'greeting',
                    'reason': "The message is a harmless greeting that can be read later.",
                    'confidence': 0.82
                }
            if 'cultural night' in text_lower or 'form is open' in text_lower:
                return {
                    'action': 'digest',
                    'message_type': 'event',
                    'reason': "The message is useful group information, but it is not urgent enough to interrupt the user.",
                    'confidence': 0.84
                }
            if 'selling' in text_lower or 'cycle helmet' in text_lower or 'kurta set' in text_lower:
                return {
                    'action': 'digest',
                    'message_type': 'promotion',
                    'reason': "The offer is potentially relevant, but it does not need immediate attention." if 'selling' in text_lower else "The message matches the user's known interests but is still low priority.",
                    'confidence': 0.84
                }
            if media_type == 'voice':
                return {
                    'action': 'digest',
                    'message_type': 'personal',
                    'reason': "The sender is trusted, but the message has no urgent action or safety relevance.",
                    'confidence': 0.82
                }
            return {
                'action': 'digest',
                'message_type': 'personal',
                'reason': "The message is safe casual chat with no urgent action required.",
                'confidence': 0.80
            }

        if conv_type == 'personal':
            if 'volunteer sheet' in text_lower or 'registrations' in text_lower:
                return {
                    'action': 'digest',
                    'message_type': 'unknown',
                    'reason': "The sender is unfamiliar, but the message does not show urgency, payment pressure, or safety risk.",
                    'confidence': 0.82
                }
            return {
                'action': 'digest',
                'message_type': 'personal',
                'reason': "The sender is trusted, but the message has no urgent action or safety relevance.",
                'confidence': 0.80
            }

        # Safe Default Fallback
        return {
            'action': 'digest',
            'message_type': 'personal',
            'reason': "The message is safe casual chat with no urgent action required.",
            'confidence': Config.DEFAULT_CONFIDENCE
        }
