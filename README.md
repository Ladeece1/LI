# LinkedIn Network Explorer — Simple (Streamlit)

Upload your LinkedIn `connections.csv` and explore:
- Interactive, draggable network
- Super connectors (degree + betweenness)
- Find connectors to reach a target (company/domain/role/keywords)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (instructions below).
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. Click **New app** → Select this repo, choose branch and `app.py` as the app file.
4. Click **Deploy**.

### App configuration
- `app.py` is at the repo root.
- Python deps are pinned in `requirements.txt`.
- (Optional) If you need secrets, set them in **Streamlit Cloud → App → Settings → Secrets**.

## Files
- `app.py` — Streamlit app
- `requirements.txt` — Python dependencies
- `.gitignore` — ignores Python/build artifacts and `.streamlit/secrets.toml`
- `README.md` — this file

## GitHub quick start

Create a new repo on GitHub (no README/license — we'll push from local). Then in your local project folder:

```bash
git init
git add app.py requirements.txt README.md .gitignore
git commit -m "Initial commit: LinkedIn Network Explorer (Streamlit)"
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git
git push -u origin main
```

> Using SSH?
> Replace the remote line with:
> ```bash
> git remote add origin git@github.com:<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git
> ```

## Updating the app
Commit and push changes. In Streamlit Cloud, enable **Deploy on push** to auto-redeploy.

## Privacy
All processing happens within the Streamlit session. No external APIs are called.
