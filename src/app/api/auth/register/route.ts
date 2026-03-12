import { NextResponse } from "next/server";
import { getDb } from "@/lib/db";
import bcrypt from "bcryptjs";
import { v4 as uuidv4 } from "uuid";

export async function POST(request: Request) {
  const { username, password } = await request.json();

  if (!username || !password) {
    return NextResponse.json({ error: "Missing username or password" }, { status: 400 });
  }

  if (username.length < 3) {
    return NextResponse.json({ error: "Username too short (min 3)" }, { status: 400 });
  }
  if (password.length < 6) {
    return NextResponse.json({ error: "Password too short (min 6)" }, { status: 400 });
  }

  const sql = getDb();

  const existing = await sql`SELECT id FROM users WHERE username = ${username}`;
  if (existing.length > 0) {
    return NextResponse.json({ error: "That username is taken" }, { status: 409 });
  }

  const hash = await bcrypt.hash(password, 10);
  const rows = await sql`
    INSERT INTO users (username, password_hash)
    VALUES (${username}, ${hash})
    RETURNING id, username
  `;

  const user = rows[0];
  const token = uuidv4();
  await sql`INSERT INTO sessions (user_id, token) VALUES (${user.id}, ${token})`;

  return NextResponse.json({ user: { id: user.id, username: user.username }, token });
}
