import { getDb } from "./db";

export async function getUserFromToken(request: Request) {
  const token = request.headers.get("Authorization")?.replace("Bearer ", "");
  if (!token) return null;

  const sql = getDb();
  const rows = await sql`
    SELECT u.id, u.username FROM users u
    JOIN sessions s ON s.user_id = u.id
    WHERE s.token = ${token}
  `;
  return rows[0] || null;
}
