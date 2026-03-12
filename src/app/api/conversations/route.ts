import { NextResponse } from "next/server";
import { getDb } from "@/lib/db";
import { getUserFromToken } from "@/lib/auth";

export async function GET(request: Request) {
  const user = await getUserFromToken(request);
  if (!user) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const sql = getDb();
  const rows = await sql`
    SELECT id, title, created_at, updated_at FROM conversations
    WHERE user_id = ${user.id}
    ORDER BY updated_at DESC
  `;
  return NextResponse.json(rows);
}

export async function POST(request: Request) {
  const user = await getUserFromToken(request);
  if (!user) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const sql = getDb();
  const rows = await sql`
    INSERT INTO conversations (user_id, title) VALUES (${user.id}, 'New Chat')
    RETURNING id, title, created_at, updated_at
  `;
  return NextResponse.json(rows[0]);
}
