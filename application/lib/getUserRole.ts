// lib/getUserRole.ts
import { auth } from "@clerk/nextjs/server";
import { prisma } from "./db";

export async function getUserRole() {
  const { userId } = await auth();

  if (!userId) {
    return null; // User is not authenticated
  }

  try {
    const user = await prisma.user.findUnique({
      where: { clerkId: userId },
      select: { role: true },
    });

    return user?.role || null;
  } catch (error) {
    console.error("Error fetching user role:", error);
    return null;
  }
}
