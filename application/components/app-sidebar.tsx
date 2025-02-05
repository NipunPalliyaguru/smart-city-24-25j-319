"use client";

import * as React from "react";
import {
  CarTaxiFront,
  CircleParking,
  GalleryVerticalEnd,
  LayoutDashboard,
  LifeBuoy,
  PersonStanding,
  Send,
  Trash,
} from "lucide-react";

import { NavMain } from "@/components/nav-main";
import { NavUser } from "@/components/nav-user";
import { TeamSwitcher } from "@/components/team-switcher";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarRail,
} from "@/components/ui/sidebar";
import { NavSecondary } from "./nav-secondary";

const baseURL =  "dashboard"

// This is sample data.
const data = {
  user: {
    name: "shadcn",
    email: "m@example.com",
    avatar: "/avatars/shadcn.jpg",
  },
  teams: [
    {
      name: "Smart City",
      logo: GalleryVerticalEnd,
      plan: "Enterprise",
    }
  ],
  navMain: [
    {
      title: "Dashboard",
      url: `/${baseURL}`,
      icon: LayoutDashboard,
    },
    {
      title: "Surveillance Enhancement",
      url: `/${baseURL}/surveillance-enhancement`,
      icon: PersonStanding,
    },
    {
      title: "Waste Management",
      url: `/${baseURL}/waste-management`,
      icon: Trash,
    },
    {
      title: "Accident Detection",
      url: `/${baseURL}/accident-detection`,
      icon: CarTaxiFront,
    },
    {
      title: "Parking Management",
      url: `/${baseURL}/parking-management`,
      icon: CircleParking,
    },
  ],
  navSecondary: [
    {
      title: "Support",
      url: "#",
      icon: LifeBuoy,
    },
    {
      title: "Feedback",
      url: "#",
      icon: Send,
    },
  ],
};

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  return (
    <Sidebar collapsible="icon" {...props}>
      <SidebarHeader>
        <TeamSwitcher teams={data.teams} />
      </SidebarHeader>
      <SidebarContent>
        <NavMain items={data.navMain} />
        <NavSecondary items={data.navSecondary} className="mt-auto" />
      </SidebarContent>
      <SidebarFooter>
        <NavUser user={data.user} />
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  );
}
