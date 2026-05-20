# Envoy MCP Coordination

This project uses Envoy as shared work state, not as an agent runtime.

Codex and Claude should access the shared Envoy space through separate
`envoy-mcp` identity lanes:

```text
Codex  -> envoy-mcp --envoy-bin /Users/avinashvankadaru/.local/bin/envoy --profile codex
Claude -> envoy-mcp --envoy-bin /Users/avinashvankadaru/.local/bin/envoy --profile claude
```

The shared local space used during setup is:

```text
space_name: shared-workspace
space_id: room_1779305260690cfc3lp
```

## Operating Model

Use Envoy for durable coordination state:

- messages and decisions;
- task state and claims;
- evidence, objections, and handoffs;
- roster and authority metadata.

Do not run a custom script that invokes both agents and posts as both Envoy
profiles. Each agent must act through its own configured MCP server/profile.

For substantial work, the owner should seed one or more Envoy task objects
instead of only sending prose messages. Agents should read history, inbox, and
task state, then claim their matching task before mutating repository state.

## Verification

Codex MCP configuration:

```bash
codex mcp get envoy
```

Expected command shape:

```text
/Users/avinashvankadaru/.local/bin/envoy-mcp --envoy-bin /Users/avinashvankadaru/.local/bin/envoy --profile codex
```

Claude MCP configuration:

```bash
claude mcp get envoy
```

Expected command shape:

```text
/Users/avinashvankadaru/.local/bin/envoy-mcp --envoy-bin /Users/avinashvankadaru/.local/bin/envoy --profile claude
```

