"""The RAGAS judge used by evaluate.py, exp2.py and exp3.py: faithfulness and answer relevancy.

config.JUDGE_BACKEND picks the judge LLM:
  dsh     DeepSeek through the DeepSeek Harness (DSH) desktop app, on its signed-in DeepSeek account (no API key), one
          fresh tool-less DSH session per prompt. config.DSH_TRANSPORT picks the route:
            desktop   POST to the optional dsh-lightrag-judge plugin (github.com/Test-Plus-XD/DSH-Discord-AutoReply,
                      lightrag-judge/) inside the running app; the session is created
                      in this repository's DSH workspace and can be watched live, thinking included.
            headless  one `dsh --profile headless` run with the overlay dsh-judge.patch.yml; the session is saved on
                      disk (grouped under config.JUDGE_DIR) but the desktop app does not list it.
  ollama  the original local judge (qwen2.5-14b-judge, or the qwen3.5-9b-judge fallback) through Ollama's /v1 API.
The answer-relevancy embeddings stay on local bge-m3 for both backends; only the judging LLM changes.
Every score is the mean of config.JUDGE_REPEATS judgings (judge_answer); the score caches keep each judging.
"""
import asyncio
import contextvars
import json
import os
import re
import statistics
import urllib.error
import urllib.request
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import ValidationError
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

import config as C

PATCH = Path(__file__).resolve().parent.parent / "configs" / "dsh-judge.patch.yml"

# A short tag such as "naive q030 faithfulness" that the scoring loops set before each metric. It becomes the first
# line of the task, so the DSH session list shows which answer and metric every judging session belongs to.
judge_label = contextvars.ContextVar("judge_label", default="")


# Raised when the judge cannot work at all (DSH missing, not signed in, quota used up). The scoring loops re-raise it
# instead of recording a failed score, so a sign-in problem stops the run rather than caching None for every answer.
class JudgeUnavailableError(RuntimeError):
    pass


# Raised when one judge call fails (a reply that is not valid JSON after every retry, a timeout, a server error). The
# scoring loops record that one score as None, as they already do for a malformed Ollama reply.
class JudgeReplyError(RuntimeError):
    pass


# Mirrors instructor's Mode.JSON instruction (instructor/process_response.py), which the Ollama judge receives
# through RAGAS's llm_factory, so both judges are asked for the same JSON instance of the same schema.
def schema_instruction(response_model):
    schema = json.dumps(response_model.model_json_schema(), indent=2)
    return ("As a genius expert, your task is to understand the content and provide\n"
            "the parsed objects in json that match the following json_schema:\n\n"
            f"{schema}\n\n"
            "Make sure to return an instance of the JSON, not the schema itself")


# Pulls the JSON object out of a reply that may wrap it in a ```json fence or surround it with a sentence; returns
# the text from the first "{" to the last "}", or the stripped reply when it has no braces (validation then fails).
def extract_json(text):
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text.strip()


# A RAGAS "collections" LLM backed by DeepSeek Harness. RAGAS calls agenerate(prompt, response_model) with a Pydantic
# model to fill; each call starts a fresh DSH session (in the desktop app or headless), so no judgement sees another
# one's conversation. Replies that fail validation are retried in a new session with the parser's complaint appended,
# like instructor's re-ask. DSH retries rate limits and server errors itself; this class adds a few whole-run retries.
class DshJudgeLLM(InstructorBaseRagasLLM):
    # Errors whose code means the account cannot be used; retrying would only repeat them. "AUTH" is DSH's code for an
    # HTTP 403 or a permission error, which the account provider does not rewrite to an ACCOUNT_ code.
    FATAL_CODES = ("MISSING_CREDENTIAL", "ACCOUNT_", "QUOTA", "AUTH")

    # cli: the dsh.cmd launcher; patch: the judge overlay; model and effort: exported for the overlay's !!js lines;
    # workdir: the headless session's working folder, which is also how DSH groups those sessions on disk.
    # transport: "desktop" or "headless"; port: the plugin's localhost API; repo: the folder whose DSH workspace shows
    # the desktop sessions (this repository).
    def __init__(self, cli, patch, model, effort, workdir, timeout_s=900, attempts=3, transport="headless",
                 port=18791, repo=None):
        self.cli, self.patch, self.model, self.effort = Path(cli), Path(patch), model, effort
        self.workdir, self.timeout_s, self.attempts = Path(workdir), timeout_s, attempts
        self.transport, self.port = transport, port
        self.repo = Path(repo or Path(__file__).resolve().parent.parent).resolve()

    # RAGAS's collections metrics only call agenerate; this synchronous form exists for completeness and must not be
    # called from inside a running event loop.
    def generate(self, prompt, response_model):
        return asyncio.run(self.agenerate(prompt, response_model))

    # Asks DSH for a JSON instance of response_model and returns the validated object. Raises JudgeReplyError when
    # every attempt fails, or JudgeUnavailableError as soon as DSH reports an unusable account.
    async def agenerate(self, prompt, response_model):
        label = judge_label.get()
        task = (f"{label}\n\n" if label else "") + f"{prompt}\n\n{schema_instruction(response_model)}"
        retry_note, problem = "", ""
        for attempt in range(self.attempts):
            try:
                if self.transport == "desktop":
                    reply = await self._ask_desktop(task + retry_note, label)
                else:
                    reply = await self._ask(task + retry_note)
            except JudgeReplyError as error:
                problem = str(error)
                if attempt + 1 < self.attempts:  # no pointless wait after the last attempt
                    await asyncio.sleep(5 * (attempt + 1))
                continue
            try:
                return response_model.model_validate_json(extract_json(reply))
            except (ValidationError, ValueError) as error:
                problem = f"invalid JSON reply: {str(error)[:300]}"
                retry_note = (f"\n\nA previous reply to this task could not be parsed ({str(error)[:300]}). "
                              "Reply with only the JSON instance.")
        raise JudgeReplyError(f"DSH judge failed after {self.attempts} attempts: {problem}")

    # Sends one task to the dsh-lightrag-judge plugin in the open desktop app and returns its reply text. The plugin
    # creates the session in this repository's workspace, titled with the label, so it streams live in DSH. A closed
    # app, a missing plugin, a rejected request or an unusable account stops the run; other failures retry.
    async def _ask_desktop(self, task, label):
        body = json.dumps({"cwd": str(self.repo), "title": label or "LightRAG judge", "prompt": task,
                           "model": self.model, "reasoningEffort": self.effort or "high"}).encode("utf-8")
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}/judge", data=body, method="POST",
                                         headers={"content-type": "application/json"})

        # Blocking HTTP call, run in a worker thread; the plugin answers when the DSH turn ends.
        def post():
            with urllib.request.urlopen(request, timeout=self.timeout_s + 60) as response:
                return json.load(response)

        try:
            return (await asyncio.to_thread(post))["text"]
        except urllib.error.HTTPError as error:
            try:
                detail = json.load(error)
            except ValueError:
                detail = {}
            code, message = str(detail.get("code") or ""), str(detail.get("error") or error.reason)[:300]
            if any(fatal in code for fatal in self.FATAL_CODES):
                raise JudgeUnavailableError(f"DSH cannot use the DeepSeek account ({code}): {message}. "
                                            "Sign in through the DeepSeek Harness desktop app and run again.")
            if error.code in (400, 403, 404, 415):  # a request the plugin will always refuse
                raise JudgeUnavailableError(f"DSH judge plugin refused the request ({error.code}): {message}")
            raise JudgeReplyError(f"DSH judge plugin error {error.code} {code}: {message}")
        except (urllib.error.URLError, ConnectionError) as error:
            raise JudgeUnavailableError(f"The DSH desktop judge plugin is not reachable on port {self.port} "
                                        f"({error}). Open DeepSeek Harness, or set LIGHTRAG_DSH_TRANSPORT=headless.")
        except TimeoutError:
            raise JudgeReplyError(f"DSH desktop judge did not answer within {self.timeout_s + 60} s")

    # Runs one headless DSH task, sending it on stdin (judge prompts exceed Windows' command-line limit), and returns
    # the final answer text. The --json event stream is parsed line by line; DSH also prints plain log lines there.
    async def _ask(self, task):
        if not self.cli.is_file():
            raise JudgeUnavailableError(f"DeepSeek Harness CLI not found: {self.cli}")
        self.workdir.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "DSH_PERMISSION_MODE": "read-only", "LIGHTRAG_DSH_MODEL": self.model,
               "LIGHTRAG_DSH_EFFORT": self.effort or ""}
        process = await asyncio.create_subprocess_exec(
            str(self.cli), "--profile", "headless", "--patch", str(self.patch), "--json", "-",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=self.workdir, env=env)
        try:
            out, err = await asyncio.wait_for(process.communicate(task.encode("utf-8")), self.timeout_s)
        except TimeoutError:
            await self._kill_tree(process)
            raise JudgeReplyError(f"DSH did not answer within {self.timeout_s} s")
        final, failure, started = None, None, False
        for line in out.decode("utf-8", "replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            started = started or event.get("type") == "session"
            if event.get("type") == "final":
                final = event.get("text", "")
            reason = event.get("reason") or {}
            if event.get("phase") == "turn_end" and reason.get("kind") == "error":
                failure = reason.get("error") or {}
        if failure is not None:
            code, message = str(failure.get("code", "")), str(failure.get("message", ""))[:300]
            if any(fatal in code for fatal in self.FATAL_CODES):
                raise JudgeUnavailableError(f"DSH cannot use the DeepSeek account ({code}): {message}. "
                                            "Sign in through the DeepSeek Harness desktop app and run again.")
            raise JudgeReplyError(f"DSH error {code}: {message}")
        stderr = err.decode("utf-8", "replace").strip()[-300:]
        if not started:  # DSH never opened a session: a broken install or overlay, which every retry would repeat
            raise JudgeUnavailableError(f"DSH failed to start (exit {process.returncode}): {stderr}")
        if process.returncode != 0 or not final:
            raise JudgeReplyError(f"DSH exited with {process.returncode} and no answer: {stderr}")
        return final

    # Ends a timed-out run. dsh.cmd runs under cmd.exe and starts "DeepSeek Harness.exe" as its child, so killing only
    # the direct child would leave DSH running on the account and holding the pipes (asyncio's wait() would then block
    # until it exits). taskkill /T ends the whole tree; the final wait is bounded in case a handle still lingers.
    @staticmethod
    async def _kill_tree(process):
        killer = await asyncio.create_subprocess_exec("taskkill", "/T", "/F", "/PID", str(process.pid),
                                                      stdout=asyncio.subprocess.DEVNULL,
                                                      stderr=asyncio.subprocess.DEVNULL)
        await killer.wait()
        try:
            await asyncio.wait_for(process.wait(), 30)
        except TimeoutError:
            pass


# Builds the two RAGAS scorers on the configured judge: (Faithfulness, AnswerRelevancy). For Ollama this is the
# original setup: temperature 0, max_tokens 3000, and reasoning_effort "none" for the Qwen 3.5 fallback judge, whose
# hidden reasoning would otherwise use up max_tokens (nothing extra is sent to qwen2.5). DSH exposes no temperature or
# output cap for headless runs, so DeepSeek runs with its own defaults; see README "Evaluation".
def make_judges():
    client = AsyncOpenAI(base_url=f"{C.OLLAMA_HOST}/v1", api_key="ollama")  # Ollama ignores the key
    if C.JUDGE_BACKEND == "dsh":
        llm = DshJudgeLLM(C.DSH_CLI, PATCH, C.DSH_JUDGE_MODEL, C.DSH_JUDGE_EFFORT, C.JUDGE_DIR / "dsh_workdir",
                          transport=C.DSH_TRANSPORT, port=C.DSH_PORT, repo=C.ROOT)
    else:
        reasoning = {} if C.JUDGE_THINK is None else {"reasoning_effort": "medium" if C.JUDGE_THINK else "none"}
        llm = llm_factory(C.JUDGE_MODEL, client=client, temperature=0, max_tokens=3000, **reasoning)
    print(f"[judge] {C.JUDGE_NAME}; scores in {C.JUDGE_DIR}", flush=True)
    embeddings = embedding_factory("openai", "bge-m3", client=client)
    return Faithfulness(llm=llm), AnswerRelevancy(llm=llm, embeddings=embeddings)


### Repeated judging. One judging of the same answer is noisy (two DeepSeek judgings moved faithfulness by 0.155 on
### average in the smoke test), so every RAGAS score is the mean of config.JUDGE_REPEATS independent judgings, each in
### its own fresh judge session. A score record keeps the individual judgings (<metric>_runs) beside their mean, the
### spread (<metric>_sd) and the number of judgings the mean was taken over (judge_repeats). The caches are therefore
### resumable at the level of single judgings: a rerun, or a run with a higher repeat count, only adds the judgings
### still missing, and a judging that failed is simply tried again next time.
METRICS = ("faithfulness", "answer_relevancy")


# Returns the successful judgings cached for one metric of a score record (an empty list for no record). A record
# written before repeated judging has only the single value, which counts as its first judging, so old caches are
# topped up rather than judged again from scratch.
def cached_runs(record, metric):
    if record is None:
        return []
    runs = record.get(f"{metric}_runs")
    if runs is None:
        runs = [record.get(metric)]
    return [value for value in runs if value is not None]


# True when a cached record needs no work: it names the current judge, every metric has at least JUDGE_REPEATS
# judgings, and its stored mean was taken over exactly JUDGE_REPEATS of them. A record averaged under another repeat
# count, or written before records named their judge, is recomputed, which costs judge calls only when it has too few
# judgings.
def judging_complete(record):
    return (record is not None and record.get("judge_model") == C.JUDGE_MODEL_ID
            and record.get("judge_repeats") == C.JUDGE_REPEATS
            and all(len(cached_runs(record, metric)) >= C.JUDGE_REPEATS for metric in METRICS))


# Judges one answer with both RAGAS metrics until each has JUDGE_REPEATS judgings, starting from those cached in
# previous, and returns the judge fields of its score record. The mean uses the first JUDGE_REPEATS judgings, so every
# answer in a comparison is averaged over the same number; extra cached judgings are kept but not used. Each session is
# labelled "<label> <metric> <n>/<repeats>" for the DSH session list. An unusable judge (JudgeUnavailableError) stops
# the run; any other failure loses only that judging, and a metric with no judging at all scores None, as before.
async def judge_answer(question, answer, contexts, faith, relevancy, label, previous=None):
    # Judgings by another judge model must never be averaged with this one's. That can only happen in the Ollama
    # folder, which both Ollama judges share, so such a record is judged again from scratch rather than topped up.
    if previous is not None and previous.get("judge_model") not in (None, C.JUDGE_MODEL_ID):
        print(f"  {label}: cached judgings are by {previous['judge_model']}, not {C.JUDGE_MODEL_ID}; judging again")
        previous = None
    scorers = {"faithfulness": lambda: faith.ascore(user_input=question, response=answer,
                                                    retrieved_contexts=contexts or [""]),
               "answer_relevancy": lambda: relevancy.ascore(user_input=question, response=answer)}
    fields = {}
    for metric, score in scorers.items():
        runs = cached_runs(previous, metric)
        for number in range(len(runs) + 1, C.JUDGE_REPEATS + 1):
            judge_label.set(f"{label} {metric} {number}/{C.JUDGE_REPEATS}")
            try:
                runs.append((await score()).value)
            except JudgeUnavailableError:
                raise
            except Exception as error:  # a malformed judge reply loses one judging, not the whole run
                print(f"  {metric} judging {number} failed for {label}: {str(error)[:150]}")
        used = runs[:C.JUDGE_REPEATS]
        fields[metric] = statistics.fmean(used) if used else None
        fields[f"{metric}_sd"] = statistics.stdev(used) if len(used) > 1 else None
        fields[f"{metric}_runs"] = runs
    fields["judge_model"], fields["judge_repeats"] = C.JUDGE_MODEL_ID, C.JUDGE_REPEATS
    return fields


# Writes one score record to a JSONL cache, replacing the line with the same key (a topped-up record) in place or
# appending a new one. A replacement rewrites the file through a temporary copy, so an interrupted run never leaves a
# half-written cache, and every reader keeps seeing one line per answer.
def save_score(path, record, key_fields):
    key = tuple(record[field] for field in key_fields)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    records = [json.loads(line) for line in lines if line.strip()]
    position = next((index for index, old in enumerate(records)
                     if tuple(old.get(field) for field in key_fields) == key), None)
    if position is None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return
    records[position] = record
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(json.dumps(old, ensure_ascii=False) + "\n" for old in records), encoding="utf-8")
    os.replace(temporary, path)
