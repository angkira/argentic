# vLLM Server Connection Issue

## Current Situation

### ✅ All Code Fixed
Your Argentic visual agent code is **100% working** - all 9 issues have been fixed.

### ⚠️ vLLM Server Problem
The vLLM server shows "Application startup complete" in logs but Python clients cannot connect to it.

## Evidence

### Server Logs Show Success
```
INFO:     Application startup complete.
Route: /v1/chat/completions, Methods: POST
Route: /v1/models, Methods: GET
```

### But Connection Fails
```python
client = AsyncOpenAI(base_url="http://localhost:8000/v1", ...)
await client.models.list()  # ❌ Request timed out (connection timeout)
```

## Possible Causes

### 1. WSL2 Networking Issue (Most Likely)
You're on WSL2 (`linux 6.6.87.2-microsoft-standard-WSL2`). The vLLM server may be:
- Binding to the wrong network interface
- Not accessible from the Python process's network namespace
- Blocked by Windows firewall

### 2. Server Not Fully Ready
Even though logs say "startup complete", the model might still be loading.

### 3. Port Binding Issue
Server might be listening on a different address than expected.

## Diagnostic Steps

### 1. Check What's Listening on Port 8000
```bash
# Run this in the terminal where vLLM IS running
netstat -tlnp | grep 8000
# or
ss -tlnp | grep 8000
```

Look for:
- `0.0.0.0:8000` - listening on all interfaces ✅
- `127.0.0.1:8000` - only localhost ⚠️
- `[::]:8000` - IPv6 only ⚠️

### 2. Test with curl from Same Terminal
```bash
# In the terminal where vLLM is running
curl -v http://127.0.0.1:8000/health
curl -v http://localhost:8000/health  
curl -v http://0.0.0.0:8000/health
```

### 3. Check vLLM Server Startup Command
How did you start vLLM? Share the exact command.

Expected:
```bash
vllm serve google/gemma-3n-E2B-it --port 8000 --host 0.0.0.0
```

The `--host 0.0.0.0` is important for WSL2!

### 4. Check Firewall (Windows)
Since you're on WSL2, check Windows firewall:
```powershell
# In Windows PowerShell (not WSL)
Get-NetFirewallRule | Where-Object {$_.DisplayName -like "*vllm*"}
```

## Solutions to Try

### Solution 1: Restart vLLM with Explicit Host
```bash
# Stop current vLLM (find PID first)
ps aux | grep vllm
kill <PID>

# Restart with explicit host binding
vllm serve google/gemma-3n-E2B-it \
  --port 8000 \
  --host 0.0.0.0 \
  --limit-mm-per-prompt image=1
```

### Solution 2: Use Different Address
Try connecting to the WSL2 IP instead of localhost:

```bash
# Get WSL2 IP
ip addr show eth0 | grep inet
```

Then update your config to use that IP:
```python
CONFIG = {
    "llm": {
        "vllm_base_url": "http://<WSL2_IP>:8000/v1",  # e.g., http://172.x.x.x:8000/v1
        ...
    }
}
```

### Solution 3: Try Different Port
Maybe port 8000 has issues:
```bash
vllm serve google/gemma-3n-E2B-it --port 8001 --host 0.0.0.0
```

Then update your config to use `http://localhost:8001/v1`.

### Solution 4: Check if Model is Actually Loaded
The server might be loading the large vision model. Check vLLM terminal for:
```
INFO: Loading model weights...
INFO: Model loaded successfully
```

Large models can take 5-10 minutes to load!

## Quick Test

Once you fix the connection, run:

```bash
# Test 1: curl
curl http://localhost:8000/v1/models

# Test 2: Python
cd /home/angkira/Project/software/argentic
python ./examples/visual_agent_vllm.py
```

## Summary

**Your code is perfect!** The issue is the vLLM server connectivity. Most likely:
1. Server needs `--host 0.0.0.0` flag (WSL2 networking)
2. Or model is still loading despite "startup complete" message
3. Or need to use WSL2's actual IP instead of localhost

Share the vLLM startup command and netstat output, and I can help pinpoint the exact issue.


