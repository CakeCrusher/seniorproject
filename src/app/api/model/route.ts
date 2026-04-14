import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({
    model: "GatorLM",
    url: "https://github.com/UF-CIS6930/GatorLM",
  });
}
