# TradeDocs Agent

I built an agent that answers questions about Alpaca's Trading API. It searches 17 pages of Alpaca's docs, writes an example API call, and runs it against Alpaca's paper trading sandbox after I approve it. Paper trading uses fake money.


## How it works

1. `python -m tradedocs.ingest` downloads the 17 pages listed in `sources.txt`. Alpaca serves each page as markdown. Guide pages get split into chunks by heading. API reference pages are mostly one large OpenAPI JSON block, so each endpoint gets rewritten as plain text before it is split. The chunks go into a local Chroma database, and the endpoint definitions get saved for the call checker.
2. When a question comes in, search picks the 6 best chunks. It runs two searches and merges the results. One compares meaning with a small embedding model that runs on my machine. The other is BM25 keyword search.
3. Claude gets the chunks and the question. It answers with links to the pages it used. When an example call helps, it uses its one tool, `call_alpaca_api`, to propose a request.
4. The call checker compares the proposed call with the endpoint definitions and with order rules the docs state in words, like "notional orders need time_in_force day". It also gives the call a color. Green means it only reads data. Orange means it places or changes something. Red means it cancels or closes something.
5. The agent stops and waits. I can approve the call, edit it first, or reject it. An edited call gets checked again.
6. An approved call goes to `https://paper-api.alpaca.markets`. Claude reads the JSON that comes back and explains it.

The agent is a LangGraph graph with five steps called retrieve, model, check, approve and run. The pause uses LangGraph's `interrupt`. Claude is called through Anthropic's Python SDK inside the model step. The default model is `claude-opus-5-5` at medium effort. If Claude declines a request, the API retries it on another model, using Anthropic's refusal fallback setting.

The downloaded docs are not in the repo. The ingest step fetches them, so results can change when Alpaca updates its docs.

## Safety

- The Alpaca host is hard coded to the paper trading URL. If `.env` points anywhere else, every call is refused.
- Only paths that start with `/v2/` and the methods GET, POST, PATCH and DELETE are allowed.
- Nothing runs until I approve it. The eval never sends calls at all.
- After 4 calls on one question, any further call is refused and the agent has to finish its answer.

## Setup

I use Python 3.12.

```bash
git clone https://github.com/danish-puri/tradedocs-agent.git
cd tradedocs-agent
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Put three keys in `.env`. The Claude key comes from console.anthropic.com. The two Alpaca keys come from the paper trading section of the Alpaca dashboard.

Build the doc index. The first run also downloads an 80 MB embedding model.

```bash
python -m tradedocs.ingest
```

Run the web page.

```bash
streamlit run app.py
```

Or ask one question in the terminal.

```bash
python -m tradedocs.cli 'How do I buy $50 worth of AAPL?'
```

Run the tests. They use a fake Claude and a fake Alpaca, so they need no keys.

```bash
pytest
```

## The eval

`evals/questions.yaml` has 20 questions. I checked each answer against the docs. For each question the script checks up to four things.

- Search returned at least one of the right pages.
- The answer links to one of the right pages.
- The agent proposed the right API call. Only the fields listed in the question are compared, and prices can be numbers or text.
- The answer contains the key words, like `notional` and `day`.

A question passes when every check that applies to it passes. The grading is plain code, so the same answer always gets the same grade. During the eval every proposed call is recorded and then rejected, so nothing reaches Alpaca.

```bash
python -m evals.run_eval                    # full run, needs ANTHROPIC_API_KEY
python -m evals.run_eval --retrieval-only   # search only, needs no keys
```

Results get saved in `evals/results/`.

## Results

Search only, from `python -m evals.run_eval --retrieval-only` on November 23, 2026.

| Search | Right page in top 6 | Right page in top 3 | Right page first |
|---|---|---|---|
| Embeddings only | 18 of 20 | 18 of 20 | 16 of 20 |
| Embeddings plus BM25 | 19 of 20 | 19 of 20 | 16 of 20 |

There is no agent score yet because I haven't run the full eval.

## What I learned about where it fails

These come from building it and from the search eval. The agent itself hasn't been graded yet.

- The small embedding model missed the "Close a Position" page for both questions about closing a position. The right chunk was in the index but ranked 20th. Adding keyword search fixed one of the two.
- "How do I sell half of my AAPL position in a single request" still fails. The answer is the `percentage` parameter on `DELETE /v2/positions/{symbol}`. That page says "liquidate" and never says "sell half", so neither search connects them.
- The reference pages are mostly JSON. Searching that raw would match on brackets and field names, so I turned each endpoint into plain sentences first.
- Alpaca's docs disagree with each other. The API reference says notional orders only work as market orders. The Fractional Trading page says they work with market, limit, stop and stop limit orders. My checker only enforces the rule both pages agree on, which is time_in_force day.
- The API reference types prices and quantities as strings, but the guide examples send numbers too. A checker that trusted the types would flag correct calls, so mine accepts either one as long as the value is a number.
- My first chunker read Python comments inside code examples, like `# Get our position in AAPL.`, as page headings. I had to make it skip code blocks.

## Project layout

```
sources.txt            the 17 doc pages
tradedocs/ingest.py    downloads pages and builds the index
tradedocs/docs.py      splits pages into chunks and reads endpoint definitions
tradedocs/store.py     hybrid search over the chunks
tradedocs/agent.py     the LangGraph agent
tradedocs/checker.py   the call checker and risk colors
tradedocs/alpaca.py    the one tool, with the paper trading guards
tradedocs/grading.py   grading for the eval
tradedocs/cli.py       terminal version
app.py                 Streamlit page
evals/                 questions, eval script, results
tests/                 pytest tests
```
