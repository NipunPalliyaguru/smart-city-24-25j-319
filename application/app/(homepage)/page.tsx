import { UserButton } from "@clerk/nextjs";
import { currentUser } from "@clerk/nextjs/server";

export default async function Home() {
  const getUserRole = async () => {
    const metaData = await currentUser();
    const roleName = metaData!.publicMetadata.role;
    console.log(roleName);
  };

  await getUserRole();

  return (
    <div className="flex min-h-screen flex-col items-center justify-center">
      This is the Homepage 👋
      <UserButton />
    </div>
  );
}
