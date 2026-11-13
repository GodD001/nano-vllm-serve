"""Acceptance demo: the official `openai` client against a running nanovllm.serve.
Completions + chat, streaming + non-streaming, usage, a Chinese/emoji prompt, a stop string.

    python benchmarks/openai_client_demo.py --base-url http://127.0.0.1:8000/v1 --model Qwen3-0.6B
"""
import argparse

import openai


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="Qwen3-0.6B")
    args = ap.parse_args()
    client = openai.OpenAI(base_url=args.base_url, api_key="unused")

    print("== models.list()")
    print([m.id for m in client.models.list()])

    print("\n== completions.create (non-streaming, greedy)")
    c = client.completions.create(model=args.model, prompt="The capital of France is", max_tokens=16, temperature=0)
    print(repr(c.choices[0].text), c.choices[0].finish_reason, c.usage)

    print("\n== completions.create (streaming, include_usage)")
    stream = client.completions.create(model=args.model, prompt="Count from one to five:", max_tokens=24,
                                       temperature=0, stream=True, stream_options={"include_usage": True})
    n_chunks, text, usage = 0, "", None
    for ch in stream:
        if ch.choices:
            n_chunks += 1
            text += ch.choices[0].text
            reason = ch.choices[0].finish_reason
        if ch.usage:
            usage = ch.usage
    print(repr(text), "| token chunks:", n_chunks, "| finish:", reason, "|", usage)

    print("\n== chat.completions.create (non-streaming)")
    chat = client.chat.completions.create(model=args.model, messages=[{"role": "user", "content": "用一句话介绍你自己。🙂"}],
                                          max_tokens=64, temperature=0,
                                          extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    print(repr(chat.choices[0].message.content), chat.choices[0].finish_reason, chat.usage)

    print("\n== chat.completions.create (streaming, include_usage, stop string)")
    stream = client.chat.completions.create(model=args.model, messages=[{"role": "user", "content": "List three fruits, one per line."}],
                                            max_tokens=128, temperature=0, stop=["3."], stream=True,
                                            stream_options={"include_usage": True},
                                            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    text, reason, usage = "", None, None
    for ch in stream:
        if ch.choices:
            text += ch.choices[0].delta.content or ""
            reason = ch.choices[0].finish_reason or reason
        if ch.usage:
            usage = ch.usage
    print(repr(text), "| finish:", reason, "|", usage)


if __name__ == "__main__":
    main()
