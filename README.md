# reels-ai
Reels AI

## Checking the UI locally
1. From the project root, start a static server: `python3 -m http.server 8000`.
2. Open your browser to `http://localhost:8000/` to view `index.html` and the latest styling.
3. Refresh after edits to verify the updated HindiClip landing page.
4. Stop the server with `Ctrl+C` when you are done previewing.

## Enabling Supabase auth for uploads
1. In `index.html`, set `data-supabase-url` and `data-supabase-anon-key` on the `<body>` tag to your Supabase project values.
2. Ensure Google, Facebook, and Instagram OAuth providers are configured in the Supabase dashboard.
3. Reload the page; clicking the upload area will now prompt social sign-in via Supabase and, once signed in, reveal the file picker.
