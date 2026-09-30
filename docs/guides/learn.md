# Learn from the usage log

Every search and call `toolrank serve` answers goes to its usage log: which tools a request
retrieved, which one the agent then called, and how that went. `toolrank learn` turns that into
better heads for your catalogue, without ever seeing a request's text.

```bash
pip install "toolrank[clm]"                 # torch, for training only; the result runs in numpy
toolrank learn --data data/mytools          # the ingest dir toolrank serve has been serving
toolrank learn --data data/mytools --dry-run   # what the log yields, nothing trained (no torch)
```

## What it learns from

The log holds no request text (see [Serve them to agents](serve.md#the-usage-log)), only a keyed
digest of the request's embedding-cache key. That is enough: the request's backbone vector is in the
ingest dir's cache, found by digesting the cache's keys with the log's key, and the tools' vectors
come from the catalogue's text through the same cache. Nothing is sent to the embedding endpoint
unless a catalogue tool changed since it was served.

A request becomes a training pair when a search of it has calls linked to it:

| The agent | The pair gets |
| --- | --- |
| called a tool and it returned `ok` | a positive |
| called a tool and it returned `tool_error` | a weak positive (the tool was the one to try; `--strict` leaves it out) |
| was shown a tool and never called it | a hard negative |
| got `refused`, `protocol_error`, `timeout`, `unknown_tool` | nothing: that says nothing about the tool |

Searches of the same request merge; requests whose tools left the catalogue are dropped and
counted. Hard negatives the starting heads score like a positive are dropped as well
(`--neg-filter 0.95`): two tools that do the same job are not each other's negatives, a lesson from
training on mined negatives.

## How it decides

The newest 20% of the requests (`--dev-share`) are held out, by time, and never trained on. The
metric is the log's own: the share of those requests whose called tool ranks in the top 5 of the
whole catalogue. The starting heads compete as epoch 0, and a later epoch has to beat them there.

With `--dev DIR` (a benchmark-format set, as `toolrank finetune` takes) the same heads are scored on
it every epoch; an epoch that wins on the log but loses more than `--max-drop` NDCG@10 points there
is not published either. The decision is one of:

- `published`: the heads are written as fp16 `.npz` (default `DATA/heads/learned-<stamp>.npz`, the
  `.pt` next to it) with what they were selected on in their `cfg`. Serve them with
  `TOOLRANK_HEADS=<file>` or `--clm-ckpt <file>`.
- `no improvement`, `benchmark dropped`: nothing is written; the report says the numbers.
- `not enough pairs` (fewer than `--min-pairs`, 20): the log is too young.

The report goes to `results/learn_<name>.json`. `--since` and `--tenant` narrow the log to a period
or to one API key's traffic.

## What to expect

A few dozen requests move Recall@5 in coarse steps and the decision can go either way from run to
run; a few hundred, with varied requests, is where the held-out share starts to mean something.
Heads learned from one catalogue's traffic are for that catalogue: the benchmark guard is there to
catch the case where they stop being good at anything else.
