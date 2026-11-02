# OpenCode V2 connection

1. Keep Strata and the ngrok tunnel running in Kaggle.
2. Copy the current HTTPS URL from the notebook output.
3. In the Kaggle shell, run `strata launch --show-key` and copy the key privately.
4. Save the key in a local ignored file: `~/.config/opencode/strata-key.txt`.
5. Merge the provider object printed by the notebook into the existing `providers` map in `opencode.jsonc`. Do not replace the rest of the config.
6. Restart OpenCode and select the `strata/<model-id>` model.
7. Start a fresh OpenCode session after changing provider or context settings.

Keep the key out of GitHub and chat. The ngrok URL is public and changes after a tunnel restart unless you configure a reserved/static domain.

8. The generated model configuration includes `reasoningEffort: "none"` for the configured model entry.
