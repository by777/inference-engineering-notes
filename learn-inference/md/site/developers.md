# Developers

<!-- https://learn-inference.com/developers -->

# Learn Inference for developers and agents

The chapter and section index behind this site is also a small public JSON API: no account, no API key, and no request quota to buy. It is read-only, so the live endpoints below are already safe to try.

## Quickstart

List every chapter:

```
curl https://learn-inference.com/api/v1/chapters
```

Get one chapter and its sections:

```
curl https://learn-inference.com/api/v1/chapters/inference
```

The full surface is described machine-readably at [/openapi.json](https://learn-inference.com/openapi.json) (OpenAPI 3.1), with typed request and response schemas an agent can load directly into a function-calling tool definition.

## MCP server

The same index is also an MCP server at `https://learn-inference.com/api/mcp`, for MCP-native clients like Claude Desktop or Claude Code. It exposes two tools: `list_chapters` and `get_chapter`, no authentication required.

```
{
  "mcpServers": {
    "learn-inference": {
      "url": "https://learn-inference.com/api/mcp"
    }
  }
}
```

It also publishes an [MCP server card](https://learn-inference.com/.well-known/mcp-server-card) for automatic discovery, per the draft [SEP-2127](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2127) proposal.

## Errors

Every non-2xx response, including an unknown path under `/api/` and an unsupported method on a real one, is JSON in the same shape: a stable `code`, a human-readable `message`, and usually a `hint` naming the next call to make.

```
curl https://learn-inference.com/api/v1/chapters/not-a-real-slug

{
  "error": {
    "code": "chapter_not_found",
    "message": "No chapter matches slug \"not-a-real-slug\".",
    "hint": "GET /api/v1/chapters for the list of valid slugs."
  }
}
```

## Versioning and rate limits

The API is versioned in the URL path (`/api/v1/...`). A breaking change ships under a new version prefix rather than changing this one in place; this version keeps working for at least 90 days after a new one ships, announced with a `Deprecation` response header and a `Sunset` date before removal.

Every response carries `RateLimit-Limit`, `RateLimit-Remaining`, and `RateLimit-Reset` headers. Going over the limit returns `429` with a `Retry-After` header, in the same JSON error shape as above.

## Full text

For reading rather than querying, every page is available as Markdown by appending `.md` to its URL, or by sending `Accept: text/markdown`. The whole book is one document at [/llms-full.txt](https://learn-inference.com/llms-full.txt), indexed at [/llms.txt](https://learn-inference.com/llms.txt).

## Reference

-   [OpenAPI 3.1 spec](https://learn-inference.com/openapi.json)
-   [GET /api/v1/chapters](https://learn-inference.com/api/v1/chapters)
-   [GET /api/v1/chapters/{slug}](https://learn-inference.com/api/v1/chapters/inference)
-   [MCP server](https://learn-inference.com/api/mcp)
-   [llms.txt](https://learn-inference.com/llms.txt)
-   [llms-full.txt](https://learn-inference.com/llms-full.txt)
-   [sitemap.xml](https://learn-inference.com/sitemap.xml)
