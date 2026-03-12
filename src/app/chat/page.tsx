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

    // placeholder for the incoming response
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

      // read SSE stream
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
              const snapshot = soFar; // capture for closure
              setMessages((prev) => {
                const copy = [...prev];
                copy[copy.length - 1] = { role: "assistant", content: snapshot };
                return copy;
              });
            }
          } catch { /* incomplete json, ignore */ }
        }
      }

      fetchConvos(); // refresh sidebar titles
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
    <div className="flex h-screen">
      {/* sidebar */}
      <div className={`${sidebar ? "w-64" : "w-0"} bg-gray-900 border-r border-gray-800 flex flex-col transition-all duration-200 overflow-hidden`}>
        <div className="p-3 border-b border-gray-800">
          <button onClick={newConvo} className="w-full py-2 px-3 bg-blue-600 hover:bg-blue-700 rounded text-sm font-medium transition">
            + New Chat
          </button>
        </div>
        <div className="flex-1 overflow-y-auto">
          {convos.map((c) => (
            <button
              key={c.id}
              onClick={() => openConvo(c.id)}
              className={`w-full text-left px-3 py-2 text-sm truncate hover:bg-gray-800 transition ${
                activeId === c.id ? "bg-gray-800 text-white" : "text-gray-400"
              }`}
            >
              {c.title}
            </button>
          ))}
        </div>
        <div className="p-3 border-t border-gray-800">
          <button
            onClick={() => {
              localStorage.clear();
              router.replace("/login");
            }}
            className="w-full py-2 text-sm text-gray-400 hover:text-white transition"
          >
            Log out
          </button>
        </div>
      </div>

      {/* main */}
      <div className="flex-1 flex flex-col">
        <div className="h-12 border-b border-gray-800 flex items-center px-4 gap-3">
          <button onClick={() => setSidebar(!sidebar)} className="text-gray-400 hover:text-white">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <span className="text-sm text-gray-400 flex-1 truncate">
            {activeId ? convos.find((c) => c.id === activeId)?.title : "Pick a conversation or start one"}
          </span>
          {modelInfo && (
            <a
              href={modelInfo.url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 text-xs bg-gray-800 border border-gray-700 rounded-full px-3 py-1 text-gray-400 hover:text-white hover:border-gray-500 transition shrink-0"
            >
              <span className="inline-block w-2 h-2 rounded-full bg-green-500" />
              {modelInfo.model.split("/").pop()}
            </a>
          )}
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-6">
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.length === 0 && (
              <div className="text-center text-gray-500 mt-32">
                <p className="text-lg">What can I help with?</p>
                {modelInfo && (
                  <a
                    href={modelInfo.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 mt-3 text-sm text-gray-400 hover:text-white transition"
                  >
                    Powered by <span className="font-medium text-gray-300">{modelInfo.model}</span>
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
                  className={`max-w-[80%] rounded-lg px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
                    msg.role === "user" ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-100"
                  }`}
                >
                  {msg.content || <span className="animate-pulse text-gray-400">...</span>}
                </div>
              </div>
            ))}
            <div ref={bottomRef} />
          </div>
        </div>

        <div className="border-t border-gray-800 p-4">
          <div className="max-w-3xl mx-auto flex gap-2">
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
              placeholder="Send a message..."
              rows={1}
              className="flex-1 resize-none bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-sm focus:outline-none focus:border-blue-500 placeholder-gray-500"
            />
            <button
              onClick={send}
              disabled={!input.trim() || isStreaming}
              className="px-4 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg transition"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19V5m0 0l-7 7m7-7l7 7" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
