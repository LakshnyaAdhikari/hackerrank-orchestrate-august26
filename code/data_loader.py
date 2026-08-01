import os
import pandas as pd
from typing import Dict, Any, List, Optional
from config import Config

class DataLoader:
    """
    Loads and indexes all 11 relational datasets with safe O(1) lookups and fallback defaults.
    """
    def __init__(self, dataset_dir: Optional[str] = None):
        self.dataset_dir = dataset_dir or Config.DATASET_DIR
        
        self.users: Dict[str, Dict[str, Any]] = {}
        self.groups: Dict[str, Dict[str, Any]] = {}
        self.group_members: Dict[tuple, Dict[str, Any]] = {}
        self.business_accounts: Dict[str, Dict[str, Any]] = {}
        self.user_business_history: Dict[tuple, Dict[str, Any]] = {}
        self.message_history: List[Dict[str, Any]] = []
        self.message_history_by_id: Dict[str, Dict[str, Any]] = {}
        self.message_events: Dict[str, Dict[str, Any]] = {}
        self.images: Dict[str, str] = {}
        self.voice_notes: Dict[str, str] = {}
        self.daily_summary: Dict[tuple, Dict[str, Any]] = {}
        
        self.load_all()

    def _safe_read_csv(self, filename: str) -> pd.DataFrame:
        filepath = os.path.join(self.dataset_dir, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        print(f"Warning: {filename} not found at {filepath}")
        return pd.DataFrame()

    def load_all(self):
        # 1. Users
        df_users = self._safe_read_csv('users.csv')
        for _, row in df_users.iterrows():
            self.users[str(row['user_id'])] = row.to_dict()

        # 2. Groups
        df_groups = self._safe_read_csv('groups.csv')
        for _, row in df_groups.iterrows():
            self.groups[str(row['group_id'])] = row.to_dict()

        # 3. Group Members
        df_gm = self._safe_read_csv('group_members.csv')
        for _, row in df_gm.iterrows():
            key = (str(row['group_id']), str(row['user_id']))
            self.group_members[key] = row.to_dict()

        # 4. Business Accounts
        df_biz = self._safe_read_csv('business_accounts.csv')
        for _, row in df_biz.iterrows():
            self.business_accounts[str(row['business_id'])] = row.to_dict()

        # 5. User Business History
        df_ubh = self._safe_read_csv('user_business_history.csv')
        for _, row in df_ubh.iterrows():
            key = (str(row['user_id']), str(row['business_id']))
            self.user_business_history[key] = row.to_dict()

        # 6. Message History
        df_mh = self._safe_read_csv('message_history.csv')
        for _, row in df_mh.iterrows():
            rec = row.to_dict()
            self.message_history.append(rec)
            self.message_history_by_id[str(rec['message_id'])] = rec

        # 7. Message Events
        df_me = self._safe_read_csv('message_events.csv')
        for _, row in df_me.iterrows():
            self.message_events[str(row['message_id'])] = row.to_dict()

        # 8. Images
        df_img = self._safe_read_csv('images.csv')
        for _, row in df_img.iterrows():
            self.images[str(row['image_id'])] = str(row['file_path'])

        # 9. Voice Notes
        df_vn = self._safe_read_csv('voice_notes.csv')
        for _, row in df_vn.iterrows():
            self.voice_notes[str(row['voice_note_id'])] = str(row['file_path'])

        # 10. Daily Notification Summary
        df_dns = self._safe_read_csv('daily_notification_summary.csv')
        for _, row in df_dns.iterrows():
            key = (str(row['user_id']), str(row['date']))
            self.daily_summary[key] = row.to_dict()

    # --- Safe Accessor Methods ---
    def get_user(self, user_id: str) -> Dict[str, Any]:
        return self.users.get(str(user_id), {
            'user_id': user_id,
            'do_not_disturb_window': '',
            'messages_opened_30d': 10,
            'messages_replied_30d': 5,
            'notifications_dismissed_30d': 2,
            'messages_reported_30d': 0
        })

    def get_group(self, group_id: str) -> Dict[str, Any]:
        if pd.isna(group_id) or not group_id:
            return {}
        return self.groups.get(str(group_id), {
            'group_id': group_id,
            'group_name': 'Unknown Group',
            'group_type': 'general',
            'member_count': 10,
            'admin_count': 1
        })

    def get_group_member(self, group_id: str, user_id: str) -> Dict[str, Any]:
        if pd.isna(group_id) or not group_id or pd.isna(user_id) or not user_id:
            return {}
        key = (str(group_id), str(user_id))
        return self.group_members.get(key, {
            'role': 'member',
            'group_muted_by_user': 0,
            'replies_sent_30d': 0,
            'notifications_dismissed_30d': 0
        })

    def get_business(self, business_id: str) -> Dict[str, Any]:
        if pd.isna(business_id) or not business_id:
            return {}
        return self.business_accounts.get(str(business_id), {
            'business_id': business_id,
            'display_name': 'Unknown Business',
            'verified': 0,
            'official_domain': '',
            'domain_used_by_sender': ''
        })

    def get_user_business_history(self, user_id: str, business_id: str) -> Dict[str, Any]:
        if pd.isna(business_id) or not business_id or pd.isna(user_id) or not user_id:
            return {}
        key = (str(user_id), str(business_id))
        return self.user_business_history.get(key, {
            'allows_promotions': 1,
            'activity_count_180d': 0,
            'messages_opened_30d': 0,
            'messages_replied_30d': 0,
            'messages_dismissed_30d': 0
        })

    def get_image_path(self, image_id: str) -> str:
        if pd.isna(image_id) or not image_id:
            return ""
        rel_path = self.images.get(str(image_id), f"media/images/{image_id}.jpg")
        return os.path.join(self.dataset_dir, rel_path)

    def get_voice_note_path(self, voice_note_id: str) -> str:
        if pd.isna(voice_note_id) or not voice_note_id:
            return ""
        rel_path = self.voice_notes.get(str(voice_note_id), f"media/audio/{voice_note_id}.mp3")
        return os.path.join(self.dataset_dir, rel_path)

if __name__ == "__main__":
    loader = DataLoader()
    print("Data loader initialized successfully.")
    print("Users count:", len(loader.users))
    print("Groups count:", len(loader.groups))
    print("Message history count:", len(loader.message_history))
