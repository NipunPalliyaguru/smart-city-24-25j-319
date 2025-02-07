import { getUserRole } from "./getUserRole"
import { redirect } from "next/navigation"

export async function adminAuth() {
  const role = await getUserRole()
  if (role !== "ADMIN") {
    redirect("/dashboard")
  }
  // If the user is an admin, we don't redirect, just return
  return role
}