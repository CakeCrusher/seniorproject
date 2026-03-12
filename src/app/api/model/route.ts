import { NextResponse } from "next/server";

export async function GET() {
  const model = process.env.HF_MODEL || "unknown";
  return NextResponse.json({
    model,
    url: `https://huggingface.co/${model}`,
  });
}
