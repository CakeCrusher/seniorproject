import { NextResponse } from "next/server";
import { getDb } from "@/lib/db";
import { getUserFromToken } from "@/lib/auth";

export async function POST(request: Request) {
  const user = await getUserFromToken(request);
  if (!user) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const { conversationId, message } = await request.json();
  if (!conversationId || !message) {
    return NextResponse.json({ error: "conversationId and message required" }, { status: 400 });
  }

  const sql = getDb();

  // make sure this convo belongs to the user
  const check = await sql`
    SELECT id FROM conversations WHERE id = ${conversationId} AND user_id = ${user.id}
  `;
  if (check.length === 0) {
    return NextResponse.json({ error: "Conversation not found" }, { status: 404 });
  }

  await sql`
    INSERT INTO messages (conversation_id, role, content)
    VALUES (${conversationId}, 'user', ${message})
  `;

  // grab full history so the model has context
  const history = await sql`
    SELECT role, content FROM messages
    WHERE conversation_id = ${conversationId}
    ORDER BY created_at ASC
  `;

  const chatMessages = history.map((row) => ({ role: row.role, content: row.content }));

  const hfRes = await fetch(`${process.env.HF_BASE_URL}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.HF_API_TOKEN}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: process.env.HF_MODEL,
      messages: [
        { role: "system", content: "You are a helpful assistant." },
        ...chatMessages,
      ],
      max_tokens: 1024,
      stream: true,
    }),
  });

  if (!hfRes.ok) {
    const txt = await hfRes.text();
    console.error(`HF returned ${hfRes.status}: ${txt}`);
    return NextResponse.json({ error: "Could not reach the model" }, { status: 502 });
  }

  // stream tokens back to client via SSE
  const enc = new TextEncoder();
  let answer = "";

  const stream = new ReadableStream({
    async start(controller) {
      const reader = hfRes.body!.getReader();
      const dec = new TextDecoder();
      let buffer = "";

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += dec.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || ""; // keep the incomplete line

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const payload = line.slice(6);
            if (payload === "[DONE]") continue;

            try {
              const json = JSON.parse(payload);
              const token = json.choices?.[0]?.delta?.content;
              if (token) {
                answer += token;
                controller.enqueue(enc.encode(`data: ${JSON.stringify({ content: token })}\n\n`));
              }
            } catch {
              // malformed chunk, just skip it
            }
          }
        }

        // persist the full response
        if (answer) {
          await sql`
            INSERT INTO messages (conversation_id, role, content)
            VALUES (${conversationId}, 'assistant', ${answer})
          `;

          // auto-title based on first user message
          const count = await sql`
            SELECT COUNT(*)::int as n FROM messages WHERE conversation_id = ${conversationId}
          `;
          if (count[0].n <= 2) {
            const title = message.length > 40 ? message.slice(0, 40) + "..." : message;
            await sql`UPDATE conversations SET title = ${title}, updated_at = NOW() WHERE id = ${conversationId}`;
          } else {
            await sql`UPDATE conversations SET updated_at = NOW() WHERE id = ${conversationId}`;
          }
        }

        controller.enqueue(enc.encode("data: [DONE]\n\n"));
        controller.close();
      } catch (e) {
        console.error("streaming blew up:", e);
        controller.error(e);
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
