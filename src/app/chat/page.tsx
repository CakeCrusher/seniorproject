"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";

interface Message {
  id?: number;
  role: "user" | "assistant";
  content: string;
}

interface Conversation {
  id: number;
  title: string;
  updated_at: string;
}

function headers() {
  const tok = localStorage.getItem("token") ?? "";
  return { Authorization: `Bearer ${tok}`, "Content-Type": "application/json" };
}

export default function ChatPage() {
  const router = useRouter();
  const [convos, setConvos] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<number | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [sidebar, setSidebar] = useState(true);
  const [modelInfo, setModelInfo] = useState<{ model: string; url: string } | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!localStorage.getItem("token")) {
      router.replace("/login");
      return;
    }
    fetchConvos();
    fetch("/api/model").then((r) => r.json()).then(setModelInfo).catch(() => {});
  }, [router]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function fetchConvos() {
    const res = await fetch("/api/conversations", { headers: headers() });
    if (res.status === 401) {
      localStorage.removeItem("token");
      router.replace("/login");
      return;
    }
    setConvos(await res.json());
  }

  async function openConvo(id: number) {
    const res = await fetch(`/api/conversations/${id}/messages`, { headers: headers() });
    setMessages(await res.json());
    setActiveId(id);
  }

  async function newConvo() {
    const res = await fetch("/api/conversations", { method: "POST", headers: headers() });
    const c = await res.json();
    setConvos((prev) => [c, ...prev]);
    setActiveId(c.id);
    setMessages([]);
  }

  async function send() {
    const text = input.trim();
    if (!text || isStreaming) return;

    let cid = activeId;
    if (!cid) {
      const res = await fetch("/api/conversations", { method: "POST", headers: headers() });
      const c = await res.json();
      setConvos((prev) => [c, ...prev]);
      cid = c.id;
      setActiveId(c.id);
    }

    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setInput("");
    setIsStreaming(true);

    if (inputRef.current) inputRef.current.style.height = "auto";

    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: headers(),
        body: JSON.stringify({ conversationId: cid, message: text }),
      });

      if (!res.ok) {
        const body = await res.json();
        setMessages((prev) => {
          const copy = [...prev];
          copy[copy.length - 1] = { role: "assistant", content: `Error: ${body.error}` };
          return copy;
        });
        return;
      }

      const reader = res.body!.getReader();
      const dec = new TextDecoder();
      let soFar = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = dec.decode(value, { stream: true });
        for (const line of text.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          const payload = line.slice(6);
          if (payload === "[DONE]") continue;

          try {
            const { content } = JSON.parse(payload);
            if (content) {
              soFar += content;
              const snapshot = soFar;
              setMessages((prev) => {
                const copy = [...prev];
                copy[copy.length - 1] = { role: "assistant", content: snapshot };
                return copy;
              });
            }
          } catch { /* incomplete json, ignore */ }
        }
      }

      fetchConvos();
    } catch (err) {
      console.error(err);
      setMessages((prev) => {
        const copy = [...prev];
        copy[copy.length - 1] = { role: "assistant", content: "Something went wrong. Try again?" };
        return copy;
      });
    } finally {
      setIsStreaming(false);
    }
  }

  return (
    <div className="flex h-screen bg-[#0c0c0f] text-[#ede9e3] font-mono">

      {/* sidebar */}
      <div className={`${sidebar ? "w-64" : "w-0"} bg-[#131316] border-r border-white/[0.06] flex flex-col transition-all duration-200 overflow-hidden`}>
        <div className="p-3 border-b border-white/[0.06]">
          <button
            onClick={newConvo}
            className="w-full py-2 px-3 bg-transparent border border-white/10 hover:border-[#c9aa71] hover:text-[#c9aa71] text-[#908c86] rounded-md text-xs transition-colors"
          >
            + new conversation
          </button>
        </div>
        <div className="flex-1 overflow-y-auto">
          {convos.map((c) => (
            <button
              key={c.id}
              onClick={() => openConvo(c.id)}
              className={`w-full text-left px-4 py-2 text-xs truncate transition-colors border-l-2 ${
                activeId === c.id
                  ? "border-[#c9aa71] text-[#e8d5a8] bg-[#c9aa71]/[0.04]"
                  : "border-transparent text-[#6b6760] hover:bg-[#1a1a1f] hover:text-[#ede9e3]"
              }`}
            >
              {c.title}
            </button>
          ))}
        </div>
        <div className="p-3 border-t border-white/[0.06]">
          <button
            onClick={() => {
              localStorage.clear();
              router.replace("/login");
            }}
            className="w-full py-2 text-xs text-[#6b6760] hover:text-[#ede9e3] transition-colors"
          >
            log out
          </button>
        </div>
      </div>

      {/* main */}
      <div className="flex-1 flex flex-col min-w-0">

        {/* topbar */}
        <div className="h-12 border-b border-white/[0.06] bg-[#0e0e12] flex items-center px-4 gap-3 flex-shrink-0">
          <button onClick={() => setSidebar(!sidebar)} className="text-[#6b6760] hover:text-[#ede9e3] transition-colors">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <span className="text-xs text-[#6b6760] flex-1 truncate italic" style={{ fontFamily: "'Fraunces', Georgia, serif", fontWeight: 300 }}>
            {activeId ? convos.find((c) => c.id === activeId)?.title : "pick a conversation or start one"}
          </span>
          {isStreaming && (
            <span className="text-[10px] text-[#8ecf72] bg-[#0f1f12] border border-[#50a050]/20 px-2.5 py-1 rounded">
              ● streaming
            </span>
          )}
          {modelInfo && (
            <a
              href={modelInfo.url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 text-[10px] bg-[#1a1a1f] border border-white/[0.06] rounded px-2.5 py-1 text-[#6b6760] hover:text-[#ede9e3] hover:border-white/20 transition-colors shrink-0"
            >
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-[#8ecf72]" />
              {modelInfo.model.split("/").pop()}
            </a>
          )}
        </div>

        {/* messages */}
        <div className="flex-1 overflow-y-auto px-4 py-6">
          <div className="max-w-3xl mx-auto space-y-5">
            {messages.length === 0 && (
              <div className="text-center mt-32">
                <p
                  className="text-4xl text-[#c9aa71] opacity-40 mb-3"
                  style={{ fontFamily: "'Fraunces', Georgia, serif", fontStyle: "italic", fontWeight: 200 }}
                >
                  Dialogue
                </p>
                <p className="text-xs text-[#6b6760]">what can I help with?</p>
                {modelInfo && (
                  <a
                    href={modelInfo.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 mt-3 text-xs text-[#6b6760] hover:text-[#ede9e3] transition-colors"
                  >
                    powered by <span className="text-[#908c86]">{modelInfo.model}</span>
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                    </svg>
                  </a>
                )}
              </div>
            )}
            {messages.map((msg, i) => (
              <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-[80%] text-xs leading-relaxed whitespace-pre-wrap px-4 py-2.5 ${
                    msg.role === "user"
                      ? "bg-[#0f1f12] border border-[#50a050]/20 text-[#8ecf72] rounded-[8px_2px_8px_8px]"
                      : "bg-[#1a1a1f] border border-white/[0.08] text-[#ede9e3] rounded-[2px_8px_8px_8px]"
                  }`}
                >
                  {msg.content || <span className="text-[#6b6760] animate-pulse">···</span>}
                </div>
              </div>
            ))}
            <div ref={bottomRef} />
          </div>
        </div>

        {/* input */}
        <div className="border-t border-white/[0.06] p-4 flex-shrink-0">
          <div className="max-w-3xl mx-auto flex gap-2 items-end">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                e.target.style.height = "auto";
                e.target.style.height = Math.min(e.target.scrollHeight, 200) + "px";
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
              }}
              placeholder="send a message…"
              rows={1}
              className="flex-1 resize-none bg-[#1a1a1f] border border-white/[0.08] focus:border-[#c9aa71] rounded-lg px-4 py-2.5 text-xs text-[#ede9e3] placeholder-[#6b6760] outline-none transition-colors font-mono"
            />
            <button
              onClick={send}
              disabled={!input.trim() || isStreaming}
              className="px-4 py-2.5 bg-[#c9aa71] hover:bg-[#e8d5a8] disabled:opacity-30 disabled:cursor-not-allowed rounded-lg transition-colors flex-shrink-0"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#0c0c0f]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19V5m0 0l-7 7m7-7l7 7" />
              </svg>
            </button>
          </div>
        </div>

      </div>
    </div>
  );
}