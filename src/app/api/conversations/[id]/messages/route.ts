import { NextResponse } from "next/server";
import { getDb } from "@/lib/db";
import { getUserFromToken } from "@/lib/auth";

export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const user = await getUserFromToken(request);
  if (!user) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const { id } = await params;
  const convId = parseInt(id);
  const sql = getDb();

  // make sure this convo actually belongs to the user
  const check = await sql`
    SELECT id FROM conversations WHERE id = ${convId} AND user_id = ${user.id}
  `;
  if (check.length === 0) {
    return NextResponse.json({ error: "Conversation not found" }, { status: 404 });
  }

  const rows = await sql`
    SELECT id, role, content, created_at FROM messages
    WHERE conversation_id = ${convId}
    ORDER BY created_at ASC
  `;
  return NextResponse.json(rows);
}
