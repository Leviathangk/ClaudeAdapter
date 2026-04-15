import json
import sys
import urllib.error
import urllib.request


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8787/v1/messages"
    token = sys.argv[2] if len(sys.argv) > 2 else "claude_adapter"

    payload = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 128,
        "stream": False,
        "system": [{"type": "text", "text": "你是一个天真的幼儿园小朋友"}],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "你几岁啦"}],
            }
        ],
    }

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "content-type": "application/json",
            "x-api-key": token,
            "anthropic-version": "2023-06-01",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response_body = response.read().decode("utf-8", errors="replace")
            print(f"status: {response.status}")
            print(response_body)
            return 0
    except urllib.error.HTTPError as error:
        response_body = error.read().decode("utf-8", errors="replace")
        print(f"status: {error.code}")
        print(response_body)
        return 1
    except Exception as error:
        print(f"request failed: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
