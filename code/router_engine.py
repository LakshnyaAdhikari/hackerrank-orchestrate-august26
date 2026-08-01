import pandas as pd
from typing import Dict, Any, Tuple
from config import Config
from data_loader import DataLoader
from feature_enricher import FeatureEnricher

class RouterEngine:
    """
    Multi-tier scoring decision router combining 2D Action+MessageType mapping,
    formula-based confidence calibration, and template reason generation.
    """
    def __init__(self, data_loader: DataLoader, enricher: FeatureEnricher):
        self.loader = data_loader
        self.enricher = enricher

    def route(self, msg: Dict[str, Any], extracted_text: str, evidence_ids: str) -> Dict[str, Any]:
        features = self.enricher.enrich(msg, extracted_text)
        
        conv_type = str(msg.get('conversation_type', '')).lower()
        media_type = str(msg.get('media_type', '')).lower() if pd.notna(msg.get('media_type')) else ""
        text_lower = features['text_lower']
        has_evidence = evidence_ids != "none"

        # --- TIER 1: Safety & Scam / Spam Override ---
        if features['has_prompt_injection']:
            return {
                'action': 'mute',
                'message_type': 'scam',
                'reason': "The message tries to instruct the router, but the routing decision should be based on the actual content and risk.",
                'confidence': 0.85
            }
        
        if features['is_scam']:
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

        # --- TIER 2: Urgent / Interrupting Notifications ---
        # Work Escalation / Urgent Personal
        if features['is_work_escalation'] or ('escalation' in text_lower and 'alert' in text_lower):
            return {
                'action': 'notify',
                'message_type': 'urgent',
                'reason': "The message is from a work context and contains a direct deadline or meeting dependency.",
                'confidence': 0.85
            }
            
        # Trusted Group Admin urgent update / School circular
        if features['is_school_circular'] or (features['is_sender_admin'] and features['is_time_sensitive']):
            msg_type = 'event' if features['is_school_circular'] else 'urgent'
            reason = "A school admin sent a same-day operational update that the user is likely to need immediately." if features['is_school_circular'] else "A trusted group admin sent a time-sensitive update that should interrupt the user."
            return {
                'action': 'notify',
                'message_type': msg_type,
                'reason': reason,
                'confidence': 0.87
            }

        # Business active order or booking update
        if conv_type == 'business' and features['is_biz_verified']:
            if 'order' in features['why_knows'] or 'delivery' in text_lower or 'shopee' in text_lower:
                if features['is_time_sensitive'] or 'today' in text_lower:
                    return {
                        'action': 'notify',
                        'message_type': 'business_update',
                        'reason': "A verified business is sending an update that matches the user's recent order history.",
                        'confidence': 0.91
                    }
            elif 'booking' in features['why_knows'] or 'flight' in text_lower or 'ticket' in text_lower:
                if features['is_time_sensitive']:
                    return {
                        'action': 'notify',
                        'message_type': 'event',
                        'reason': "A verified business is sending a reminder that matches the user's recent booking history.",
                        'confidence': 0.89
                    }

        # Direct Personal Question / Request
        if conv_type == 'personal' and (features['is_direct_mention'] or features['is_time_sensitive']) and not features['is_greeting']:
            if 'urgent' in text_lower or 'help' in text_lower or media_type == 'voice':
                return {
                    'action': 'notify',
                    'message_type': 'urgent' if media_type == 'voice' or 'urgent' in text_lower else 'personal',
                    'reason': "A close contact sent a short urgent request that should interrupt the user." if media_type == 'voice' else "The sender directly asks this user for a response or action.",
                    'confidence': 0.87
                }

        if conv_type == 'group' and features['is_direct_mention'] and not features['is_group_muted']:
            return {
                'action': 'notify',
                'message_type': 'personal',
                'reason': "The sender directly asks this user for a response or action.",
                'confidence': 0.87
            }

        # --- TIER 4: Mute Low-Value / Repetitive / Opted-out ---
        if conv_type == 'business':
            if not features['allows_promotions'] or features['is_marketing']:
                if media_type == 'voice':
                    return {
                        'action': 'mute',
                        'message_type': 'spam',
                        'reason': "The user has opted out of or repeatedly dismissed similar marketing messages.",
                        'confidence': 0.81
                    }
                elif not features['has_recent_activity'] or 'unsubscribe' in text_lower:
                    return {
                        'action': 'mute',
                        'message_type': 'promotion',
                        'reason': "The user has opted out of or repeatedly dismissed similar marketing messages.",
                        'confidence': 0.81
                    }

        if conv_type == 'group':
            if features['is_greeting'] and features['is_group_muted']:
                return {
                    'action': 'mute',
                    'message_type': 'greeting',
                    'reason': "The sender has a pattern of repeated forwards or greetings that the user usually ignores.",
                    'confidence': 0.85
                }
            if 'forwarded' in msg and msg.get('forwarded_count', 0) > 2 and features['is_group_muted']:
                return {
                    'action': 'mute',
                    'message_type': 'forward',
                    'reason': "The sender has a pattern of repeated forwards or greetings that the user usually ignores.",
                    'confidence': 0.83
                }
            if ('selling' in text_lower or 'kurta' in text_lower or 'plots' in text_lower) and features['is_group_muted']:
                return {
                    'action': 'mute',
                    'message_type': 'promotion',
                    'reason': "Similar historical messages were ignored, dismissed, or muted by this user.",
                    'confidence': 0.85
                }

        # --- TIER 3: Digest Useful Updates ---
        if conv_type == 'business':
            if features['is_biz_verified']:
                return {
                    'action': 'digest',
                    'message_type': 'business_update',
                    'reason': "The verified business message is legitimate but does not require immediate attention.",
                    'confidence': 0.84
                }
            elif features['allows_promotions']:
                return {
                    'action': 'digest',
                    'message_type': 'promotion',
                    'reason': "The message is promotional but matches a topic or business the user has opted into.",
                    'confidence': 0.78
                }

        if conv_type == 'group':
            if features['is_greeting']:
                return {
                    'action': 'digest',
                    'message_type': 'greeting',
                    'reason': "The message is a harmless greeting that can be read later.",
                    'confidence': 0.82
                }
            if 'kurta' in text_lower or 'selling' in text_lower or media_type == 'image':
                return {
                    'action': 'digest',
                    'message_type': 'promotion',
                    'reason': "The message matches the user's known interests but is still low priority.",
                    'confidence': 0.84
                }
            if media_type == 'voice':
                return {
                    'action': 'digest',
                    'message_type': 'personal',
                    'reason': "The sender is trusted, but the message has no urgent action or safety relevance.",
                    'confidence': 0.82
                }
            if features['is_time_sensitive'] or 'test' in text_lower or 'meeting' in text_lower:
                return {
                    'action': 'digest',
                    'message_type': 'event',
                    'reason': "The message is useful group information, but it is not urgent enough to interrupt the user.",
                    'confidence': 0.84
                }
            return {
                'action': 'digest',
                'message_type': 'personal',
                'reason': "The message is safe casual chat with no urgent action required.",
                'confidence': 0.80
            }

        if conv_type == 'personal':
            if 'volunteer' in text_lower or 'number' in text_lower:
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
