# Privacy

<!-- https://learn-inference.com/privacy -->

# Privacy

Nothing here needs your name. No account, no email, nothing following you to the next site. This page is the actual mechanism, not a lawyer’s paraphrase of it.

## Ask AI

Type a question and it travels through Vercel’s AI Gateway, along with whatever page you’re reading, so the answer can be specific to it. It can only read pages already on this site. Not the web, not code execution, not the server’s files.

The server hashes your IP before anything else touches it. That hash stops one browser from hammering the endpoint, nothing else, and the raw address is never written down. Your side of the conversation lives in local storage on your machine. The server keeps a matching copy under the same session, so a reply that was mid-stream when you reloaded can keep going. Clear the conversation and your copy disappears.

## Everything else

Vercel Analytics and Speed Insights count visits and page-load speed in aggregate, no cookie involved. The API and MCP server need no key and rate-limit by IP in memory only, gone the moment the server restarts. Dark mode or light lives in local storage too. Fonts are baked into the site at build time, so even the first load never asks Google for anything.
