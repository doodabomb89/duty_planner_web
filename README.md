# Ops Planner Web App Deployment

This repository contains your existing Flask application along with the
additional files needed to deploy it to a platform‐as‐a‐service provider such
as **Render** or **Railway**. Your application code (in `app.py`, `templates/`
and `static/`) has **not been modified**. The added files simply instruct the
hosting service how to run your app.

## Files Added

| File      | Purpose                                                                                       |
|-----------|------------------------------------------------------------------------------------------------|
| `wsgi.py` | A minimal entry point that imports your Flask application as `app` so that WSGI servers can    |
|           | locate it. No application logic changes.                                                      |
| `Procfile`| Tells the hosting platform to run `gunicorn wsgi:app` with sensible defaults.                  |
| `requirements.txt` | Lists your Python dependencies. `gunicorn` has been added to support production use. |
| `runtime.txt` | Pins the Python version (`python-3.10.14`) to ensure a consistent runtime (optional).      |
| `README.md` | This deployment guide.                                                                       |
| `.env.example` | Documents environment variables your app expects.                                         |

## Environment Variables

Your app uses a secret key for Flask sessions. Hosting providers allow you to
set environment variables through their dashboards. Copy `.env.example` to
create an `.env` file locally or set the variable directly in your provider:

```
FLASK_SECRET=please-change-this
```

Replace `please-change-this` with a long, random value in production.

## Deploy to Render

1. **Create a repository**: Push this folder to a new GitHub repository.
2. **Create a web service**:
   * Sign in to [Render](https://render.com). Click **New → Web Service**.
   * Connect your GitHub repo and choose the main branch.
   * Render automatically detects Python. Leave the build command blank.
   * Confirm the start command shown in the *Procfile*: `gunicorn wsgi:app --workers=2 --threads=4 --timeout=120`.
3. **Set environment variables**: In Render’s **Environment** tab, add `FLASK_SECRET` with a secure value.
4. **Deploy**: Click **Create Web Service**. After the build and deploy finish, Render provides a public URL (e.g. `https://your-service.onrender.com`). Share this with anyone.

## Deploy to Railway

1. **Create a repository**: Push this folder to a new GitHub repository.
2. **Start a project**:
   * Go to [Railway](https://railway.app). Click **New Project → Deploy from GitHub**.
   * Select your repository. Railway auto‑detects Python and uses the `Procfile`.
3. **Set variables**: Add `FLASK_SECRET` under project variables.
4. **Deploy**: Railway will build and launch your app. Copy the generated public domain and share it.

## Local Development

You can still run the app locally as before:

```
python app.py

# or, to test exactly like production:
gunicorn wsgi:app --reload
```

## Quick Temporary Share

If you need to demo the site from your own machine without fully deploying,
you can use a tunnelling tool (e.g. **Cloudflare Tunnel** or **ngrok**) while
your Flask app is running locally. These commands expose your local port
8000 to the internet temporarily:

```
# Cloudflare Tunnel (install cloudflared first)
cloudflared tunnel --url http://127.0.0.1:8000

# or ngrok (install ngrok first)
ngrok http 8000
```

Both options require you to keep your terminal open and your computer on;
they are for temporary testing only. For a permanent, shareable link, use
Render or Railway as described above.