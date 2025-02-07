import { getUserRole } from "@/lib/getUserRole"
import { redirect } from "next/navigation"

export default async function Page() {
  const role = await getUserRole()

  if (role === "ADMIN") {
    redirect("/admin")
  } else {
    redirect("/dashboard")
  }
}