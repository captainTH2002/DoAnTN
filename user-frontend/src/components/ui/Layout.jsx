// src/components/Layout.jsx
import Navbar from "./Navbar"
import { Outlet } from "react-router-dom"

export default function Layout() {
  return (
    <div className="flex flex-col min-h-screen">
      <Navbar />
      <main className="flex-grow bg-gray-50 px-4 py-6">
        <Outlet />
      </main>
    </div>
  )
}
