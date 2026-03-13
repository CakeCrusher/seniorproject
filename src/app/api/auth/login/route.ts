import { NextResponse } from "next/server";
import { getDb } from "@/lib/db";
import bcrypt from "bcryptjs";
import { v4 as uuidv4 } from "uuid";

// ── Remove this block once the DB is live ───────────────────────────────
const MOCK_USERS = [
  { id: "mock-1", username: "admin", password: "admin", role: "admin" },
];

async function mockLogin(username: string, password: string) {
  const user = MOCK_USERS.find(
    (u) => u.username === username && u.password === password
  );
  if (!user) return null;
  return { id: user.id, username: user.username, token: uuidv4() };
}
// ─────────────────────────────────────────────────────────────────────────────

export async function POST(request: Request) {
  const body = await request.json();

  if (!body.username || !body.password) {
    return NextResponse.json({ error: "Missing fields" }, { status: 400 });
  }

  // ── Try  DB first fall back to mock─────────────────
  try {
    const sql = getDb();
    const rows = await sql`
      SELECT id, username, password_hash FROM users WHERE username = ${body.username}
    `;

    if (rows.length === 0) {
      return NextResponse.json(
        { error: "Wrong username or password" },
        { status: 401 }
      );
    }

    const user = rows[0];
    const match = await bcrypt.compare(body.password, user.password_hash);
    if (!match) {
      return NextResponse.json(
        { error: "Wrong username or password" },
        { status: 401 }
      );
    }

    const token = uuidv4();
    await sql`INSERT INTO sessions (user_id, token) VALUES (${user.id}, ${token})`;

    return NextResponse.json({
      user: { id: user.id, username: user.username },
      token,
    });
  } catch (dbError) {
    // ── DB unavailable , use mock credentials ─────────────────────────────────
    console.warn("[auth] DB unavailable, falling back to mock login:", dbError);

    const mock = await mockLogin(body.username, body.password);
    if (!mock) {
      return NextResponse.json(
        { error: "Wrong username or password" },
        { status: 401 }
      );
    }

    return NextResponse.json({
      user: { id: mock.id, username: mock.username },
      token: mock.token,
    });
    // ───────────────────────────────────────────────────────
  }
}
