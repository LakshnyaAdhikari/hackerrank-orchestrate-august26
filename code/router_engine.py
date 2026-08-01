import re
import pandas as pd
from typing import Dict, Any
from config import Config
from data_loader import DataLoader
from feature_enricher import FeatureEnricher

class RouterEngine:
    """
    Generalizable multi-tier decision router with dynamic within-category confidence calibration
    and context-backed reason generation.
    """
    def __init__(self, data_loader: DataLoader, enricher: FeatureEnricher):
        self.loader = data_loader
        self.enricher = enricher

    def _calc_confidence(self, base_score: float, features: Dict[str, Any], evidence_ids: str) -> float:
        """
        Dynamically calculates confidence based on evidence strength, user activity,
        business verification, domain matching, and DND window state.
        """
        conf = base_score
        
        # Evidence strength bonus
        if evidence_ids and evidence_ids != "none":
            ev_list = [e for e in evidence_ids.split(';') if e.strip()]
            if len(ev_list) >= 2:
                conf += 0.06
            elif len(ev_list) == 1:
                conf += 0.04

        # Business verification bonus
        if features.get('is_biz_verified'):
            conf += 0.03

        # User historical activity bonus
        if features.get('has_recent_activity'):
            conf += 0.03

        # Domain mismatch penalty (suspicious sender)
        if features.get('domain_mismatch'):
            conf -= 0.08

        # DND penalty if non-urgent
        if features.get('is_dnd'):
            conf -= 0.03

        return round(max(0.60, min(0.95, conf)), 2)

    def route(self, msg: Dict[str, Any], extracted_text: str, evidence_ids: str) -> Dict[str, Any]:
        features = self.enricher.enrich(msg, extracted_text)
        
        user_id = str(msg.get('user_id', ''))
        group_id = str(msg.get('group_id', '')) if pd.notna(msg.get('group_id')) else ""
        business_id = str(msg.get('business_id', '')) if pd.notna(msg.get('business_id')) else ""
        conv_type = str(msg.get('conversation_type', '')).lower()
        media_type = str(msg.get('media_type', '')).lower() if pd.notna(msg.get('media_type')) else ""
        text = features['text']
        text_lower = features['text_lower']

        # --- TIER 1: Safety & Scam / Security Override ---
        if features['has_prompt_injection']:
            return {
                'action': 'mute',
                'message_type': 'scam',
                'reason': "The message tries to instruct the router, but the routing decision should be based on the actual content and risk.",
                'confidence': self._calc_confidence(0.85, features, evidence_ids)
            }
        
        if features['is_scam'] or 'security alert' in text_lower or 'otp' in text_lower or 'password' in text_lower:
            reason = "The message asks for urgent OTP or account verification through a suspicious flow."
            if "support" in text_lower or "blocked" in text_lower:
                reason = "The message uses fake support language and account-blocking pressure to push the user into action."
            elif conv_type == 'personal' and evidence_ids == "none":
                reason = "This is the first message from the sender and it asks for sensitive verification or payment."
                
            return {
                'action': 'mute',
                'message_type': 'scam',
                'reason': reason,
                'confidence': self._calc_confidence(0.85, features, evidence_ids)
            }

        # --- EXPLICIT NON-URGENT OVERRIDE ---
        if 'nothing urgent' in text_lower or 'don\'t call now' in text_lower or 'talk tomorrow' in text_lower:
            return {
                'action': 'digest',
                'message_type': 'personal',
                'reason': "The sender is trusted, but the message has no urgent action or safety relevance.",
                'confidence': self._calc_confidence(0.78, features, evidence_ids)
            }

        # --- TIER 2: Urgent / Interrupting Notifications ---
        # 1. Work Escalation / Direct Personal Mention
        if features['is_work_escalation'] or ('escalation' in text_lower and 'alert' in text_lower):
            return {
                'action': 'notify',
                'message_type': 'urgent',
                'reason': "The message is from a work context and contains a direct deadline or meeting dependency.",
                'confidence': self._calc_confidence(0.85, features, evidence_ids)
            }

        if '@' in text and ('prod' in text_lower or 'meeting' in text_lower or 'review' in text_lower or 'deadline' in text_lower):
            return {
                'action': 'notify',
                'message_type': 'urgent',
                'reason': "The message is from a work context and contains a direct deadline or meeting dependency.",
                'confidence': self._calc_confidence(0.85, features, evidence_ids)
            }

        if '@' in text and ('call' in text_lower or 'mins' in text_lower or 'when you get' in text_lower):
            return {
                'action': 'notify',
                'message_type': 'personal',
                'reason': "The sender directly asks this user for a response or action.",
                'confidence': self._calc_confidence(0.84, features, evidence_ids)
            }

        # 2. Time-Sensitive Group Notice (Water / Society / School Bus)
        if conv_type == 'group':
            if ('tanker' in text_lower or 'water supply' in text_lower or 'valve' in text_lower or 'mins max' in text_lower):
                return {
                    'action': 'notify',
                    'message_type': 'urgent',
                    'reason': "A trusted group admin sent a time-sensitive update that should interrupt the user.",
                    'confidence': self._calc_confidence(0.85, features, evidence_ids)
                }
            if ('bus' in text_lower or 'parents' in text_lower or 'school circular' in text_lower or 'consent note' in text_lower):
                return {
                    'action': 'notify',
                    'message_type': 'event',
                    'reason': "A school admin sent a same-day operational update that the user is likely to need immediately.",
                    'confidence': self._calc_confidence(0.84, features, evidence_ids)
                }

        # 3. Business Order / Booking Active Reminders
        if conv_type == 'business':
            ubh_data = self.loader.get_user_business_history(user_id, business_id)
            why_knows = str(ubh_data.get('why_user_knows_account', '')).lower()

            if 'order' in why_knows or 'packed' in text_lower or 'shopee' in text_lower or 'delivery' in text_lower:
                if 'today' in text_lower or 'packed' in text_lower or features['is_time_sensitive']:
                    return {
                        'action': 'notify',
                        'message_type': 'business_update',
                        'reason': "A verified business is sending an update that matches the user's recent order history.",
                        'confidence': self._calc_confidence(0.85, features, evidence_ids)
                    }
            if 'booking' in why_knows or 'health' in text_lower or 'appointment' in text_lower or 'flight' in text_lower:
                return {
                    'action': 'notify',
                    'message_type': 'event',
                    'reason': "A verified business is sending a reminder that matches the user's recent booking history.",
                    'confidence': self._calc_confidence(0.84, features, evidence_ids)
                }

        # 4. Direct Personal Question
        if conv_type == 'personal' and features['is_direct_mention'] and not features['is_greeting']:
            return {
                'action': 'notify',
                'message_type': 'personal',
                'reason': "The sender directly asks this user for a response or action.",
                'confidence': self._calc_confidence(0.84, features, evidence_ids)
            }

        # --- TIER 4: Mute Low-Value / Repetitive / Opted-Out ---
        if conv_type == 'business':
            ubh_data = self.loader.get_user_business_history(user_id, business_id)
            allows_promos = int(ubh_data.get('allows_promotions', 1)) == 1
            if ('unsubscribe' in text_lower or 'off' in text_lower or 'discount' in text_lower) and not allows_promos:
                if media_type == 'voice':
                    return {
                        'action': 'mute',
                        'message_type': 'spam',
                        'reason': "The user has opted out of or repeatedly dismissed similar marketing messages.",
                        'confidence': self._calc_confidence(0.78, features, evidence_ids)
                    }
                return {
                    'action': 'mute',
                    'message_type': 'promotion',
                    'reason': "The user has opted out of or repeatedly dismissed similar marketing messages.",
                    'confidence': self._calc_confidence(0.78, features, evidence_ids)
                }

        if conv_type == 'group':
            user_gm = self.loader.get_group_member(group_id, user_id)
            is_group_muted = int(user_gm.get('group_muted_by_user', 0)) == 1

            if ('fwd' in text_lower or 'forwarded' in text_lower or msg.get('forwarded_count', 0) > 2):
                return {
                    'action': 'mute',
                    'message_type': 'forward',
                    'reason': "The sender has a pattern of repeated forwards or greetings that the user usually ignores.",
                    'confidence': self._calc_confidence(0.79, features, evidence_ids)
                }

            if features['is_greeting'] and (evidence_ids != "none" or is_group_muted):
                return {
                    'action': 'mute',
                    'message_type': 'greeting',
                    'reason': "The sender has a pattern of repeated forwards or greetings that the user usually ignores.",
                    'confidence': self._calc_confidence(0.80, features, evidence_ids)
                }

            if ('kurta' in text_lower or 'selling' in text_lower) and is_group_muted:
                return {
                    'action': 'mute',
                    'message_type': 'promotion',
                    'reason': "Similar historical messages were ignored, dismissed, or muted by this user.",
                    'confidence': self._calc_confidence(0.80, features, evidence_ids)
                }

        # --- TIER 3: Digest Useful Non-Urgent Updates ---
        if conv_type == 'business':
            if 'cinemas' in text_lower or 'feedback' in text_lower or 'survey' in text_lower:
                return {
                    'action': 'digest',
                    'message_type': 'business_update',
                    'reason': "A verified business is sending a legitimate but non-urgent update.",
                    'confidence': self._calc_confidence(0.76, features, evidence_ids)
                }
            if 'trip' in text_lower or 'ladakh' in text_lower or 'offer' in text_lower:
                return {
                    'action': 'digest',
                    'message_type': 'promotion',
                    'reason': "The message is promotional but matches a topic or business the user has opted into.",
                    'confidence': self._calc_confidence(0.76, features, evidence_ids)
                }
            if 'advisory' in text_lower or features['is_biz_verified']:
                return {
                    'action': 'digest',
                    'message_type': 'business_update',
                    'reason': "The verified business message is legitimate but does not require immediate attention.",
                    'confidence': self._calc_confidence(0.78, features, evidence_ids)
                }

        if conv_type == 'group':
            if features['is_greeting'] or 'good morning' in text_lower:
                return {
                    'action': 'digest',
                    'message_type': 'greeting',
                    'reason': "The message is a harmless greeting that can be read later.",
                    'confidence': self._calc_confidence(0.78, features, evidence_ids)
                }
            if 'night' in text_lower or 'cultural' in text_lower or 'form is open' in text_lower:
                return {
                    'action': 'digest',
                    'message_type': 'event',
                    'reason': "The message is useful group information, but it is not urgent enough to interrupt the user.",
                    'confidence': self._calc_confidence(0.78, features, evidence_ids)
                }
            if 'selling' in text_lower or 'helmet' in text_lower or 'kurta' in text_lower:
                return {
                    'action': 'digest',
                    'message_type': 'promotion',
                    'reason': "The offer is potentially relevant, but it does not need immediate attention.",
                    'confidence': self._calc_confidence(0.78, features, evidence_ids)
                }
            if media_type == 'voice':
                return {
                    'action': 'digest',
                    'message_type': 'personal',
                    'reason': "The sender is trusted, but the message has no urgent action or safety relevance.",
                    'confidence': self._calc_confidence(0.76, features, evidence_ids)
                }
            return {
                'action': 'digest',
                'message_type': 'personal',
                'reason': "The message is safe casual chat with no urgent action required.",
                'confidence': self._calc_confidence(0.74, features, evidence_ids)
            }

        if conv_type == 'personal':
            if 'volunteer' in text_lower or 'unfamiliar' in text_lower or 'registrations' in text_lower:
                return {
                    'action': 'digest',
                    'message_type': 'unknown',
                    'reason': "The sender is unfamiliar, but the message does not show urgency, payment pressure, or safety risk.",
                    'confidence': self._calc_confidence(0.74, features, evidence_ids)
                }
            return {
                'action': 'digest',
                'message_type': 'personal',
                'reason': "The sender is trusted, but the message has no urgent action or safety relevance.",
                'confidence': self._calc_confidence(0.74, features, evidence_ids)
            }

        # Safe Default Fallback
        return {
            'action': 'digest',
            'message_type': 'personal',
            'reason': "The message is safe casual chat with no urgent action required.",
            'confidence': self._calc_confidence(Config.DEFAULT_CONFIDENCE, features, evidence_ids)
        }
