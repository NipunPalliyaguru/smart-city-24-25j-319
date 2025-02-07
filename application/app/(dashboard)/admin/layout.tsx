import { adminAuth } from "@/lib/adminAuth";

export default async function AdminDashboardLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  await adminAuth()
  return <>{children}</>;
}
export const dynamic = "force-dynamic"