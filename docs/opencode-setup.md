# OpenCode V2 connection

1. Keep Strata and the ngrok tunnel running in Kaggle.
2. Copy the current HTTPS URL from the notebook output.
3. Run `strata launch --show-key` in the Kaggle terminal or a private notebook cell and copy the key. Clear it from any shared output and never commit it.
4. Save the key in a local ignored file: `~/.config/opencode/strata-key.txt` (`%USERPROFILE%\.config\opencode\strata-key.txt` on Windows). Do not commit it.
5. Merge the notebook's `provider.strata` entry into the existing top-level `provider` object in `opencode.jsonc`. Do not replace other providers or MCP settings.
6. Set the top-level `model` to `strata/<model-id>`.
7. Restart OpenCode and start a fresh session after changing provider or context settings.

The provider must use `npm` and `options`; a top-level `strata` object or `settings` is ignored by OpenCode. Example:

```jsonc
{
  "provider": {
    "strata": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Strata Kaggle",
      "options": {
        "baseURL": "https://YOUR-TUNNEL.ngrok-free.dev/v1",
        "apiKey": "{file:~/.config/opencode/strata-key.txt}",
        "headers": {
          "ngrok-skip-browser-warning": "1"
        }
      },
      "models": {
        "qwen3.8-flash-next-gyro-gyro-s": {
          "name": "qwen3.8-flash-next-gyro-gyro-s",
          "tool_call": true,
          "limit": {
            "context": 204800,
            "output": 8192
          }
        }
      }
    }
  },
  "model": "strata/qwen3.8-flash-next-gyro-gyro-s"
}
```

`{file:...}` reads the key without embedding it in the config. The `ngrok-skip-browser-warning` header bypasses ngrok's browser interstitial for API requests. Update `options.baseURL` when the tunnel changes.

Strata generates a new API key when it restarts. Update the key file after each fresh launch and restart OpenCode; the key stays stable while the same server is running. Keep the key out of GitHub and shared notebook output. The ngrok URL is public and changes after a tunnel restart unless you configure a reserved/static domain.
