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

  const history = await sql`
    SELECT role, content FROM messages
    WHERE conversation_id = ${conversationId}
    ORDER BY created_at ASC
  `;

  // Convert flat history to completed (user, assistant) turn pairs.
  // Exclude the last row (current user message, no reply yet).
  const turns: [string, string][] = [];
  const prior = history.slice(0, -1);
  for (let i = 0; i < prior.length - 1; i++) {
    if (prior[i].role === "user" && prior[i + 1].role === "assistant") {
      turns.push([prior[i].content, prior[i + 1].content]);
      i++;
    }
  }

  const hfRes = await fetch(process.env.HF_ENDPOINT_URL!, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.HF_TOKEN}`,
    },
    body: JSON.stringify({ inputs: { turns, message } }),
  });

  if (!hfRes.ok) {
    const txt = await hfRes.text();
    console.error(`HF endpoint returned ${hfRes.status}: ${txt}`);
    return NextResponse.json({ error: "Could not reach the model" }, { status: 502 });
  }

  const { reply } = await hfRes.json() as { reply: string };

  const enc = new TextEncoder();

  const stream = new ReadableStream({
    async start(controller) {
      try {
        controller.enqueue(enc.encode(`data: ${JSON.stringify({ content: reply })}\n\n`));

        if (reply) {
          await sql`
            INSERT INTO messages (conversation_id, role, content)
            VALUES (${conversationId}, 'assistant', ${reply})
          `;

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
