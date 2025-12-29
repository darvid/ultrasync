# ultrasync saas hub api — implementation spec

## scope
this document specifies the **server-authoritative hub-and-spoke sync server**
for the managed saas version of ultrasync.  
target consumers: coding agents (claude code, codex).

---

## invariants
- server is the **single source of truth**
- server assigns total order (`server_seq`)
- ops are immutable, idempotent, replayable
- state is derived from `(snapshot, ordered ops)`
- clients treat local lmdb as cache + outbox only

---

## topology
- hub-and-spoke
- one logical hub per `{org, region}`
- long-lived bidirectional stream per client

---

## ordering & consistency
- strong ordering per `{org, project, namespace}`
- read-your-writes after ack
- optimistic local apply allowed on clients
- server reconciliation always wins

---

## op envelope (canonical)
```json
{
  "op_id": "string",                // hash(actor_id, command_id, payload)
  "org_id": "uuid",
  "project_id": "uuid",
  "actor_id": "uuid",
  "server_seq": 0,                  // assigned by server only
  "hlc_ts": "uint64",
  "namespace": "string",
  "key": "string",
  "op_type": "set|del|patch|crdt_op",
  "payload": {},
  "precondition": {
    "expected_version": 0
  }
}
````

rules:

* `server_seq` strictly monotonic
* rejected commands never receive `server_seq`
* ops are append-only

---

## namespaces & semantics

| namespace | consistency | semantics  |
| --------- | ----------- | ---------- |
| presence  | eventual    | lww        |
| settings  | strong      | cas / etag |
| index     | strong      | serialized |
| collab    | mixed       | crdt ops   |
| metadata  | strong      | lww or cas |

---

## client protocol

### connect

```
client → server:
hello {
  org_id,
  project_id,
  client_id,
  last_server_seq
}
```

```
server → client:
ops {
  seq_start,
  ops[]
}
```

---

### command submission

```
client → server:
command {
  command_id,
  namespace,
  key,
  payload,
  precondition?
}
```

---

### ack

```
server → client:
ack {
  command_id,
  server_seq,
  op
}
```

---

### reject

```
server → client:
reject {
  command_id,
  reason,
  current_state?
}
```

client rules:

* may optimistically apply command
* must reconcile on ack/reject

---

## server responsibilities

### validation

* authz (org, project, namespace)
* schema validation per namespace
* precondition checks

### oplog

* append-only
* ordered by `server_seq`
* persisted before ack

### state materialization

* deterministic apply
* no side effects during apply
* derived from snapshots + ops

---

## storage contract (abstract)

### oplog

```
key: (org, project, namespace, server_seq)
value: op_bytes
```

### state

```
key: (org, project, namespace, key)
value: state_bytes
```

### snapshots

```
key: snapshot_id
value: {
  org,
  project,
  namespace,
  last_server_seq,
  state_blob
}
```

### client cursors

```
key: client_id
value: last_acked_server_seq
```

backend must support:

* atomic append
* ordered iteration
* atomic batch commit

---

## batching & performance

* apply ops in 1–10 ms batches
* compress op batches (zstd)
* single-writer apply loop

targets:

* p50 ack < 20 ms
* p99 ack < 100 ms
* 10k+ ops/sec/org

---

## snapshotting & gc

### snapshot

* periodic (time or op count)
* snapshot frontier = `server_seq`

### gc

* compute `min(last_acked_seq across clients)`
* delete ops ≤ min_ack_seq
* configurable retention floor

---

## failure handling

* client disconnect → resume from last_server_seq
* server crash → replay oplog from last snapshot
* duplicate ops → ignored via `server_seq`
* partial writes → disallowed (atomic append only)

---

## security

* tls everywhere
* per-org isolation
* per-project authz
* full audit via oplog

---

## future compatibility

* op envelope compatible with crdt mode
* transport-agnostic
* supports future gossip / multi-hub replication
* no breaking changes required

---

## success criteria

* full state sync via single stream
* deterministic rebuild from oplog
* no client-side conflicts visible to users
* shared core with future on-prem deployment
