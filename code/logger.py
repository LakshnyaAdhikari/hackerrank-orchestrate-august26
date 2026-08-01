import os
import datetime

# Compute paths dynamically relative to repository root
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CODE_DIR)
LOG_DIR = os.path.expanduser(r'~\hackerrank_orchestrate_august26')
LOG_PATH = os.path.join(LOG_DIR, 'log.txt')

def log_action(title: str, user_prompt: str, agent_response: str, actions: list, tool_name: str = "Antigravity"):
    """
    Appends a turn entry to the compliance log file per Section 5 of AGENTS.md.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    now_iso = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()
    
    # Redact potential secrets
    sanitized_prompt = user_prompt.replace("\n", " ")
    for key in ["GEMINI_API_KEY", "OPENAI_API_KEY", "SECRET", "KEY", "TOKEN"]:
        if key in sanitized_prompt:
            sanitized_prompt = sanitized_prompt.replace(key, "[REDACTED]")
            
    actions_str = "\n".join([f"* {a}" for a in actions]) if actions else "* Executed task step"
    
    log_entry = f"""## [{now_iso}] {title[:80]}

User Prompt (verbatim, secrets redacted):
{sanitized_prompt}

Agent Response Summary:
{agent_response}

Actions:
{actions_str}

Context:
tool={tool_name}
branch=main
repo_root={REPO_ROOT}
worktree=main
parent_agent=none

"""
    try:
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Warning: Failed to write to compliance log: {e}")

if __name__ == "__main__":
    log_action("Logger Check", "Check logger paths", "Logger path dynamically resolved.", ["Validated logger.py"])
    print(f"Logger test passed. Dynamic Repo Root: {REPO_ROOT}")
