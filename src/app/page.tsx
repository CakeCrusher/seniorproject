"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function Home() {
  const router = useRouter();

  useEffect(() => {
    const token = localStorage.getItem("token");
    if (token) {
      router.replace("/chat");
    } else {
      router.replace("/login");
    }
  }, [router]);

  return (
    <div className="flex items-center justify-center h-screen">
      <div className="animate-pulse text-gray-400">Loading...</div>
    </div>
  );
}
