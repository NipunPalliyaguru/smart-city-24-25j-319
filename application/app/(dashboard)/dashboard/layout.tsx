import { getUserRole } from "@/lib/getUserRole";
import { redirect } from "next/navigation";

export default async function UserDashboardLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const role = await getUserRole()

  if (role === "ADMIN") {
    redirect("/admin")
  }
  return <>{children}</>
}
export const dynamic = "force-dynamic"