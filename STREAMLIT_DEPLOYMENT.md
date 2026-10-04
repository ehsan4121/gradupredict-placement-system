# Streamlit Community Cloud Deployment

1. Create a GitHub repository and upload the contents of this project folder.
2. Keep `app.py`, `prediction.py`, `requirements.txt`, `model/placement_model.json`, and `.streamlit/config.toml` in their existing locations.
3. Sign in to Streamlit Community Cloud with GitHub.
4. Select **Create app** and choose the GitHub repository.
5. Set the main file path to `app.py`.
6. Deploy and wait until the application health check succeeds.
7. Open the public link and test prediction, result display, tabs, and CSV export.

The SQLite file works for demonstrations, but Streamlit Community Cloud storage is ephemeral. Saved history may reset after an application restart or redeployment. Use an external hosted database only if permanent multi-user history becomes a formal requirement.
