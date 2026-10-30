"""Pass local Hub credentials through SSH stdin without logging the token."""
import subprocess
from huggingface_hub import get_token

token = get_token()
assert token
subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', 'trinity-2-23',
                'umask 077; mkdir -p /tmp/shihangz/virap_auth; cat > /tmp/shihangz/virap_auth/token'], input=token, text=True, check=True)
subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', 'trinity-2-23',
                "tmux new-session -d -s virap-publish 'HF_TOKEN_PATH=/tmp/shihangz/virap_auth/token /tmp/shihangz/photo-tone-transfer/Runtime/venv/bin/python -u /tmp/shihangz/photo-tone-transfer/Research/publish_virap.py > /tmp/shihangz/virap_publish.log 2>&1'"], check=True)
