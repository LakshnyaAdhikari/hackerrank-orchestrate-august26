import re
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import DataLoader
from config import Config

class EvidenceMatcher:
    """
    Retrieves evidence message IDs from message_history.csv for a given incoming message.
    Filters by user/group/business context, computes TF-IDF similarity, and correlates with message_events.
    """
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.history = data_loader.message_history
        self.events = data_loader.message_events
        self.min_threshold = Config.EVIDENCE_SIMILARITY_THRESHOLD

    def _normalize_text(self, text: str) -> str:
        if pd.isna(text) or not text:
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return " ".join(text.split())

    def find_evidence(self, msg: Dict[str, Any], extracted_text: str = "") -> str:
        user_id = str(msg.get('user_id', ''))
        conv_type = str(msg.get('conversation_type', ''))
        group_id = str(msg.get('group_id', '')) if pd.notna(msg.get('group_id')) else ""
        business_id = str(msg.get('business_id', '')) if pd.notna(msg.get('business_id')) else ""
        sender_id = str(msg.get('sender_user_id', '')) if pd.notna(msg.get('sender_user_id')) else ""
        
        target_text = self._normalize_text(extracted_text or msg.get('message_text', ''))

        # Step 1: Filter candidates by user and conversation scope
        candidates = []
        for h in self.history:
            if str(h.get('user_id')) != user_id:
                continue
            
            # Scope filter
            if group_id and str(h.get('group_id', '')) == group_id:
                candidates.append(h)
            elif business_id and str(h.get('business_id', '')) == business_id:
                candidates.append(h)
            elif conv_type == 'personal' and (str(h.get('sender_user_id', '')) == sender_id or str(h.get('user_id', '')) == user_id):
                candidates.append(h)
            elif not group_id and not business_id:
                candidates.append(h)

        if not candidates:
            # Fallback to all candidates for user
            candidates = [h for h in self.history if str(h.get('user_id')) == user_id]

        if not candidates:
            return "none"

        # Step 2: TF-IDF similarity calculation
        corpus = [target_text] + [self._normalize_text(c.get('message_text', '')) for c in candidates]
        
        # If texts are empty (e.g. voice notes), fallback to exact metadata / context matching
        if not target_text or all(len(txt) == 0 for txt in corpus):
            # For voice notes or empty text, select recent messages from same conversation context with events
            scored_candidates = []
            for c in candidates:
                c_id = str(c.get('message_id'))
                ev = self.events.get(c_id, {})
                score = 0.5 if ev else 0.3
                scored_candidates.append((score, c_id))
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            if scored_candidates and scored_candidates[0][0] >= 0.3:
                return scored_candidates[0][1]
            return "none"

        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2)).fit(corpus)
            tfidf_matrix = vectorizer.transform(corpus)
            sim_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        except Exception:
            sim_scores = [0.0] * len(candidates)

        # Step 3: Combine text similarity with event reaction signals
        scored_results = []
        for idx, c in enumerate(candidates):
            sim = sim_scores[idx]
            c_id = str(c.get('message_id'))
            ev = self.events.get(c_id, {})
            
            # Bonus if past message had clear user interaction (dismissed/muted/replied)
            event_bonus = 0.0
            if ev:
                if ev.get('muted_after_message') == 1 or ev.get('message_reported') == 1 or ev.get('notification_dismissed') == 1:
                    event_bonus += 0.15
                if ev.get('message_replied') == 1:
                    event_bonus += 0.10
                    
            final_score = sim + event_bonus
            scored_results.append((final_score, sim, c_id))

        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Select top matches meeting threshold
        matched_ids = []
        for final_score, sim, c_id in scored_results:
            if sim >= self.min_threshold or final_score >= 0.50:
                if c_id not in matched_ids:
                    matched_ids.append(c_id)
                if len(matched_ids) >= 2:
                    break

        if matched_ids:
            return ";".join(matched_ids)
        
        # Fallback: if single candidate exists with event history
        if scored_results and (scored_results[0][1] > 0.25 or scored_results[0][0] >= 0.40):
            return scored_results[0][2]

        return "none"

if __name__ == "__main__":
    loader = DataLoader()
    matcher = EvidenceMatcher(loader)
    sample_msg = loader.message_history[0] if loader.message_history else {}
    print("Evidence for sample message:", matcher.find_evidence(sample_msg))
