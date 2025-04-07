"use client";
import Link from "next/link";
import { useState } from "react";
import { Menu, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";

export default function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <nav className="bg-white shadow-md px-6 py-3 flex justify-between items-center">
      {/* Left - Title */}
      <h1 className="text-xl font-bold">Scheduler</h1>

      {/* Right - Links */}
      <div className="hidden md:flex space-x-6">
        <Button variant="ghost" asChild>
          <Link href="https://github.com">GitHub</Link>
        </Button>
        <Button variant="ghost" asChild>
          <Link href="/docs">Docs</Link>
        </Button>
        <Button variant="ghost" asChild>
          <Link href="/contact">Contact</Link>
        </Button>

        {/* User Dropdown */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" className="w-8 h-8 rounded-full bg-gray-300"></Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem>Profile</DropdownMenuItem>
            <DropdownMenuItem>Settings</DropdownMenuItem>
            <DropdownMenuItem>Logout</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Mobile Menu Button */}
      <Button variant="ghost" className="md:hidden" onClick={() => setMenuOpen(!menuOpen)}>
        {menuOpen ? <X size={24} /> : <Menu size={24} />}
      </Button>

      {/* Mobile Menu */}
      {menuOpen && (
        <div className="absolute top-16 right-6 bg-white shadow-lg rounded-md p-4 flex flex-col space-y-4 md:hidden">
          <Button variant="ghost" asChild>
            <Link href="https://github.com">GitHub</Link>
          </Button>
          <Button variant="ghost" asChild>
            <Link href="/docs">Docs</Link>
          </Button>
          <Button variant="ghost" asChild>
            <Link href="/contact">Contact</Link>
          </Button>
          <Button variant="ghost" className="w-8 h-8 rounded-full bg-gray-300"></Button>
        </div>
      )}
    </nav>
  );
}
